"""
summarizer.py
--------------
Job: condense a full lecture into a short "core content only" video --
definitions and main topic explanations kept, examples/anecdotes/
tangents/administrative talk cut -- fitting within a target duration
budget (TARGET_SUMMARY_MIN_MINUTES to TARGET_SUMMARY_MAX_MINUTES).

This is a different job from rag_backend.process_query(): that answers
a SPECIFIC question by searching for relevant segments. This looks at
the ENTIRE lecture and classifies every segment by importance, which
is what a broad "summarize this" request actually needs -- a vague
query like "summary" matches almost everything in semantic search, so
process_query() alone can't produce a genuinely condensed video.

Approach: split the transcript into batches (to stay under Groq's
free-tier tokens-per-minute limit), score every segment in each batch
1-10 on how essential it is, then combine scores across all batches and
greedily keep the highest-scored segments until adding the next one
would push total kept duration past the max-minutes budget -- so the
result is always under budget, biased toward the most essential
content rather than an arbitrary chronological or per-batch cutoff.

CHANGES vs the previous version:
  1. Retry with backoff on every Groq call (_score_batch). A single
     rate-limit or network blip used to kill the entire run and lose
     all scoring work done so far -- now it retries up to 3x with
     exponential backoff before giving up on just that one batch.
  2. Failed batches no longer fail silently. If a batch can't be
     scored even after retries, that's now counted and surfaced in
     the returned stats (`batches_failed`) instead of just quietly
     producing a worse summary with no signal that something broke.
  3. Batches now carry a small overlap (last 2 segments of the
     previous batch) purely as CONTEXT for the next batch's prompt --
     not re-scored, just shown so the model isn't blind to what was
     said right before a batch boundary. This fixes definitions that
     start in one batch and finish in the next getting scored twice,
     out of context, by two calls that couldn't see each other.
"""

import json
import time
from pathlib import Path

from groq import Groq

from config import (
    GROQ_API_KEY, GROQ_MODEL, TRANSCRIPT_DIR,
    TARGET_SUMMARY_MIN_MINUTES, TARGET_SUMMARY_MAX_MINUTES,
)
from video_cutter import cut_and_stitch

_client = None

# Groq's free tier caps llama-3.3-70b-versatile at 12,000 tokens/minute
# PER REQUEST, and that budget is shared across input + output. A full
# 1-hour lecture transcript alone can be 13,000+ tokens, well over that
# in a single call -- so we split the transcript into batches that each
# stay safely under the limit (leaving headroom for the model's
# response), score each batch's segments independently, then combine
# scores across all batches afterward.
BATCH_TARGET_CHARS = 16000  # ~4000 tokens of transcript text per batch, leaving room for prompt scaffolding + response
SECONDS_BETWEEN_BATCHES = 8  # spreads calls out so consecutive batches don't stack into the same TPM window
OVERLAP_SEGMENTS = 2  # segments carried from the end of one batch into the next, as context only (not re-scored)
MAX_RETRIES = 3


def _get_client() -> Groq:
    global _client
    if _client is None:
        _client = Groq(api_key=GROQ_API_KEY)
    return _client


def _load_segments(video_id: str) -> list[dict]:
    transcript_path = TRANSCRIPT_DIR / f"{video_id}.json"
    if not transcript_path.exists():
        raise FileNotFoundError(f"No transcript found for video_id={video_id}.")
    return json.loads(transcript_path.read_text())["segments"]


def _batch_segments(segments: list[dict], overlap: int = OVERLAP_SEGMENTS) -> list[list[tuple[int, dict, bool]]]:
    """
    Groups (global_index, segment, is_context) triples into batches
    whose combined text stays under BATCH_TARGET_CHARS, without
    splitting a segment across batches.

    is_context=True marks segments carried over from the tail of the
    previous batch purely so the model has continuity -- these are
    shown in the prompt but must NOT be scored (they're already
    handled, either as real scores from the previous batch or about
    to be superseded once we see them again in-context).
    """
    batches = []
    current: list[tuple[int, dict, bool]] = []
    current_chars = 0

    for i, seg in enumerate(segments):
        seg_chars = len(seg["text"]) + 40  # + rough overhead for the timestamp/index prefix
        if current and current_chars + seg_chars > BATCH_TARGET_CHARS:
            batches.append(current)
            # Carry the tail of this batch forward as non-scored context.
            tail = current[-overlap:] if overlap else []
            current = [(idx, s, True) for idx, s, _ in tail]
            current_chars = sum(len(s["text"]) + 40 for _, s, _ in current)
        current.append((i, seg, False))
        current_chars += seg_chars

    if current:
        batches.append(current)

    return batches


def _score_batch(batch: list[tuple[int, dict, bool]], max_retries: int = MAX_RETRIES) -> dict[int, int]:
    """
    Asks the LLM to score every SCORABLE (non-context) segment in this
    batch from 1-10 on how essential it is for a "core definitions and
    main topics" summary. Segments marked as context-only are shown
    for continuity but the prompt explicitly tells the model not to
    score them.

    Retries with exponential backoff on failure (rate limits, network
    errors, malformed responses that even the regex fallback can't
    parse) rather than propagating the exception and losing every
    other batch's work along with it. Returns {} only after all
    retries are exhausted -- the caller is responsible for treating
    that as a "this batch failed" signal, not silent success.

    Absolute 1-10 scores (rather than a per-batch ranked list) are what
    make it valid to compare segments from DIFFERENT batches against
    each other later -- an ordinal rank of "1st in this batch" doesn't
    tell you how it compares to "1st in batch 3".
    """
    context_indices = {i for i, _, is_ctx in batch if is_ctx}

    numbered = "\n".join(
        f"[{i}] ({seg['start']:.1f}s-{seg['end']:.1f}s){' [CONTEXT ONLY -- do not score]' if is_ctx else ''} {seg['text']}"
        for i, seg, is_ctx in batch
    )

    prompt = f"""You are scoring part of a lecture transcript for a condensed summary video.
This is a SECTION of a longer lecture -- segments are numbered with their
original position in the full lecture, so numbers won't start at 0.

Some lines are marked [CONTEXT ONLY -- do not score]. These are the tail
end of the previous section, shown only so you understand what was being
discussed right before this section starts. Do NOT include them in your
scores.

{numbered}

Score each SCORABLE segment (i.e. NOT marked context-only) from 1-10 on
how essential it is for a summary that should contain ONLY: definitions,
key concepts, and main topics that are ACTUALLY EXPLAINED in this
transcript section.

Give LOW scores or OMIT ENTIRELY (don't include in your response) any segment that is:
  - an example, anecdote, or illustrative story
  - administrative talk (attendance, assignments, logistics)
  - repeated/restated points or filler
  - a topic that is merely MENTIONED or RECAPPED from a previous lecture
    without actually being explained here (e.g. "as we covered last
    class, X works like...") -- only score a topic highly if THIS
    transcript section contains the real explanation, not just a reference to it.

Return ONLY a JSON object mapping segment index (as a string) to score,
for segments worth including at all. Example:
{{"12": 9, "15": 7, "23": 8}}
No explanation, no markdown formatting.
"""

    client = _get_client()
    last_error = None

    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model=GROQ_MODEL,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1,
            )
            raw = response.choices[0].message.content.strip()

            try:
                scores = {int(k): int(v) for k, v in json.loads(raw).items()}
            except (json.JSONDecodeError, ValueError):
                import re
                pairs = re.findall(r'"?(\d+)"?\s*:\s*(\d+)', raw)
                scores = {int(k): int(v) for k, v in pairs}

            # Belt-and-suspenders: strip out anything the model scored
            # despite the context-only instruction, so a prompt-following
            # slip can't leak a context segment into the real results.
            return {idx: score for idx, score in scores.items() if idx not in context_indices}

        except Exception as e:
            last_error = e
            if attempt < max_retries - 1:
                wait = SECONDS_BETWEEN_BATCHES * (2 ** attempt)
                print(f"Batch scoring error ({e}), retrying in {wait}s... (attempt {attempt + 1}/{max_retries})")
                time.sleep(wait)

    print(f"Batch scoring failed after {max_retries} attempts, giving up on this batch: {last_error}")
    return {}


def _score_all_segments(segments: list[dict]) -> tuple[dict[int, int], int]:
    """
    Batches the full transcript, scores each batch, and merges the
    results into one global {segment_index: score} map.

    Returns (scores, batches_failed) -- batches_failed lets the caller
    surface "part of this lecture couldn't be scored" instead of
    silently producing a summary that's missing a chunk with no
    explanation why.
    """
    batches = _batch_segments(segments)
    all_scores: dict[int, int] = {}
    batches_failed = 0

    for batch_num, batch in enumerate(batches):
        batch_scores = _score_batch(batch)
        if not batch_scores:
            batches_failed += 1
        all_scores.update(batch_scores)

        # Only pause between batches, not after the last one.
        if batch_num < len(batches) - 1:
            time.sleep(SECONDS_BETWEEN_BATCHES)

    return all_scores, batches_failed


def _select_within_budget(
    segments: list[dict], scores: dict[int, int], max_minutes: float
) -> list[int]:
    """
    Sorts scored segments from highest to lowest importance, then
    accumulates them until the next one would push total kept duration
    past max_minutes. Returns the kept indices, chronologically sorted
    (ready for merging/cutting).
    """
    budget_seconds = max_minutes * 60

    # Highest score first; ties broken by original order (earlier = kept first).
    ranked = sorted(scores.items(), key=lambda item: (-item[1], item[0]))

    kept = []
    total = 0.0
    for idx, _score in ranked:
        seg = segments[idx]
        seg_duration = seg["end"] - seg["start"]
        if total + seg_duration > budget_seconds:
            continue  # skip this one, but keep checking lower-priority (possibly shorter) ones
        kept.append(idx)
        total += seg_duration

    kept.sort()
    return kept


def generate_summary_video(video_id: str, video_path: Path, output_path: Path) -> tuple[Path | None, dict]:
    """
    Produces a condensed "core content only" video for video_id.
    Returns (output_path_or_None, stats_dict).
    """
    segments = _load_segments(video_id)
    if not segments:
        return None, {"error": "No transcript segments found."}

    original_duration = segments[-1]["end"] - segments[0]["start"]

    scores, batches_failed = _score_all_segments(segments)
    if not scores:
        return None, {"error": "The model didn't identify any core-content segments.", "batches_failed": batches_failed}

    kept_indices = _select_within_budget(segments, scores, TARGET_SUMMARY_MAX_MINUTES)
    kept_segments = [segments[i] for i in kept_indices]
    kept_duration = sum(s["end"] - s["start"] for s in kept_segments)

    timestamps = [{"start": s["start"], "end": s["end"]} for s in kept_segments]
    result_path = cut_and_stitch(video_path, timestamps, output_path)

    stats = {
        "original_minutes": round(original_duration / 60, 1),
        "summary_minutes": round(kept_duration / 60, 1),
        "segments_kept": len(kept_segments),
        "segments_total": len(segments),
        "under_target_min": kept_duration < TARGET_SUMMARY_MIN_MINUTES * 60,
        "batches_failed": batches_failed,
    }
    return result_path, stats


if __name__ == "__main__":
    import sys
    from config import VIDEO_DIR, DATA_DIR

    video_id = sys.argv[1]
    video_path = VIDEO_DIR / f"{video_id}.mp4"
    output_path = DATA_DIR / f"summary_{video_id}.mp4"

    path, stats = generate_summary_video(video_id, video_path, output_path)
    print(stats)
    if path:
        print(f"Done: {path}")