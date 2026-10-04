"""
summarizer.py
--------------
Job: condense a full lecture into a short "core content only" video.

CHANGE vs the previous version: added a post-processing silence-removal
pass on the STITCHED summary output, not the original lecture. This is
a separate step from segment selection -- selecting the right segments
doesn't remove a pause sitting in the middle of a kept segment (e.g. a
speaker pausing mid-explanation), so without this, that silence was
still ending up in the final video.

Can't reuse whisper_gaps for this step: once segments are cut and
re-stitched into a new timeline, the original transcript's word
timestamps no longer correspond to anything in the new video. So this
step extracts fresh audio from the stitched output and runs Silero VAD
directly on the waveform, which doesn't depend on any prior timeline.
If that fails for any reason, falls back to the raw stitched version
rather than losing the whole summary over one post-processing step,
and flags the fallback in stats (post_silence_removed: False).
"""

import json
import shutil
import subprocess
import tempfile
import time
from pathlib import Path

from groq import Groq

from config import (
    GROQ_API_KEY, GROQ_MODEL, TRANSCRIPT_DIR,
    TARGET_SUMMARY_MIN_MINUTES, TARGET_SUMMARY_MAX_MINUTES,
)
from video_cutter import cut_and_stitch
from silence_remover import remove_silence

_client = None

WINDOW_TARGET_CHARS = 16000
SECONDS_BETWEEN_CALLS = 8
OVERLAP_SEGMENTS = 2
MAX_RETRIES = 3

TOPIC_MIN_ALLOCATION_SECONDS = 20.0
MIN_SEGMENTS_TO_SCORE = 4
MAX_FALLBACK_ASSIGN_DISTANCE_SECONDS = 60.0

POST_SILENCE_METHOD = "silero"  # doesn't depend on the original lecture's transcript timeline, which no longer aligns once segments are cut and re-stitched


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


def _extract_json_array(raw: str) -> list:
    text = raw.strip()
    if text.startswith("```"):
        text = text.strip("`")
        if text.lower().startswith("json"):
            text = text[4:]
        text = text.strip()

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    start = text.find("[")
    end = text.rfind("]")
    if start != -1 and end != -1 and end > start:
        return json.loads(text[start:end + 1])

    raise ValueError("Could not extract a JSON array from the response.")


def _extract_temp_audio(video_path: Path, tmp_dir: Path) -> Path:
    """
    Pulls a mono 16kHz WAV out of a video for silence detection only --
    no drift correction needed here (unlike ingest.py's extract_audio),
    since this audio and video came from the same ffmpeg encode moments
    ago and are already in sync.
    """
    audio_path = tmp_dir / "temp_audio.wav"
    subprocess.run(
        ["ffmpeg", "-y", "-i", str(video_path), "-ac", "1", "-ar", "16000", "-vn", str(audio_path)],
        check=True, capture_output=True,
    )
    return audio_path


def _get_duration_seconds(path: Path) -> float:
    result = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration",
         "-of", "default=noprint_wrappers=1:nokey=1", str(path)],
        capture_output=True, text=True,
    )
    try:
        return float(result.stdout.strip())
    except ValueError:
        return 0.0


def _window_segments(segments: list[dict], overlap: int = OVERLAP_SEGMENTS) -> list[list[tuple[int, dict, bool]]]:
    windows = []
    current: list[tuple[int, dict, bool]] = []
    current_chars = 0

    for i, seg in enumerate(segments):
        seg_chars = len(seg["text"]) + 40
        if current and current_chars + seg_chars > WINDOW_TARGET_CHARS:
            windows.append(current)
            tail = current[-overlap:] if overlap else []
            current = [(idx, s, True) for idx, s, _ in tail]
            current_chars = sum(len(s["text"]) + 40 for _, s, _ in current)
        current.append((i, seg, False))
        current_chars += seg_chars

    if current:
        windows.append(current)

    return windows


def _propose_topics_for_window(window: list[tuple[int, dict, bool]], max_retries: int = MAX_RETRIES) -> list[dict]:
    numbered = "\n".join(
        f"[{i}] ({seg['start']:.1f}s-{seg['end']:.1f}s){' [CONTEXT ONLY]' if is_ctx else ''} {seg['text']}"
        for i, seg, is_ctx in window
    )

    prompt = f"""You are identifying distinct topics discussed in part of a lecture transcript.
This is a SECTION of a longer lecture -- segments are numbered with their
original position in the full lecture and shown with timestamps.

Some lines are marked [CONTEXT ONLY]. These are the tail end of the
previous section, shown only so you understand what was already being
discussed when this section starts. Do NOT propose a topic based only
on context lines -- only propose topics actually covered by the
non-context lines below.

{numbered}

List the distinct topics covered in the NON-CONTEXT lines above. For each:
- title: a short (3-6 word) topic name
- start: the timestamp (seconds) where this topic starts being discussed
- end: the timestamp (seconds) where this topic's discussion ends
- key_points: one sentence on what's actually explained

Skip administrative talk (attendance, assignments, logistics) and skip
examples/tangents that don't introduce a new topic on their own.

Return ONLY a JSON array. No explanation. No markdown code fences.
[{{"title": "...", "start": 123.4, "end": 210.0, "key_points": "..."}}]
"""

    client = _get_client()
    last_error = None

    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model=GROQ_MODEL,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.2,
            )
            raw = response.choices[0].message.content.strip()
            topics = _extract_json_array(raw)
            return [
                t for t in topics
                if isinstance(t, dict) and {"title", "start", "end"} <= t.keys()
            ]
        except Exception as e:
            last_error = e
            if attempt < max_retries - 1:
                wait = SECONDS_BETWEEN_CALLS * (2 ** attempt)
                print(f"Topic proposal error ({e}), retrying in {wait}s... (attempt {attempt + 1}/{max_retries})")
                time.sleep(wait)

    print(f"Topic proposal failed after {max_retries} attempts, skipping this window: {last_error}")
    return []


def _propose_all_topics(windows: list[list[tuple[int, dict, bool]]]) -> tuple[list[dict], int]:
    all_topics = []
    windows_failed = 0

    for w_num, window in enumerate(windows):
        topics = _propose_topics_for_window(window)
        if not topics:
            windows_failed += 1
        all_topics.extend(topics)

        if w_num < len(windows) - 1:
            time.sleep(SECONDS_BETWEEN_CALLS)

    return all_topics, windows_failed


def _merge_topics_fallback(raw_topics: list[dict]) -> list[dict]:
    if not raw_topics:
        return []

    merged = [dict(raw_topics[0])]
    for topic in raw_topics[1:]:
        last = merged[-1]
        if topic["start"] <= last["end"]:
            last["end"] = max(last["end"], topic["end"])
        else:
            merged.append(dict(topic))
    return merged


def _merge_topics(raw_topics: list[dict], max_retries: int = MAX_RETRIES) -> list[dict]:
    raw_topics = sorted(raw_topics, key=lambda t: t["start"])
    if not raw_topics:
        return []

    prompt = f"""Here is a list of topics proposed from different sections of the same
lecture, in chronological order. Adjacent sections overlapped slightly,
so the same topic may appear more than once with slightly different
boundaries, or a topic may have been split across two entries because
it crossed a section boundary.

{json.dumps(raw_topics)}

Merge these into one clean, chronological list of DISTINCT topics --
combine duplicate/overlapping entries into one, using the widest
start/end range that's actually correct and one clear title. Return
ONLY a JSON array in the same shape as the input. No explanation. No
markdown code fences.
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
            merged = _extract_json_array(raw)
            cleaned = [
                t for t in merged
                if isinstance(t, dict) and {"title", "start", "end"} <= t.keys()
            ]
            if cleaned:
                return sorted(cleaned, key=lambda t: t["start"])
        except Exception as e:
            last_error = e
            if attempt < max_retries - 1:
                wait = SECONDS_BETWEEN_CALLS * (2 ** attempt)
                print(f"Topic merge error ({e}), retrying in {wait}s... (attempt {attempt + 1}/{max_retries})")
                time.sleep(wait)

    print(f"Topic merge failed after {max_retries} attempts, falling back to rule-based merge: {last_error}")
    return _merge_topics_fallback(raw_topics)


def _assign_segments_to_topics(segments: list[dict], topics: list[dict]) -> tuple[list[list[int]], list[int]]:
    assignments = [[] for _ in topics]
    unassigned = []

    for idx, seg in enumerate(segments):
        midpoint = (seg["start"] + seg["end"]) / 2
        placed = False
        for t_idx, topic in enumerate(topics):
            if topic["start"] <= midpoint < topic["end"]:
                assignments[t_idx].append(idx)
                placed = True
                break
        if not placed:
            unassigned.append((idx, midpoint))

    uncovered = []
    for idx, midpoint in unassigned:
        best_t, best_dist = None, float("inf")
        for t_idx, topic in enumerate(topics):
            if midpoint < topic["start"]:
                dist = topic["start"] - midpoint
            elif midpoint > topic["end"]:
                dist = midpoint - topic["end"]
            else:
                dist = 0
            if dist < best_dist:
                best_dist = dist
                best_t = t_idx

        if best_t is not None and best_dist <= MAX_FALLBACK_ASSIGN_DISTANCE_SECONDS:
            assignments[best_t].append(idx)
        else:
            uncovered.append(idx)

    return assignments, uncovered


def _allocate_topic_budgets(topics: list[dict], total_budget_seconds: float) -> list[float]:
    total_original = sum(t["end"] - t["start"] for t in topics) or 1.0
    raw = [
        max(TOPIC_MIN_ALLOCATION_SECONDS, (t["end"] - t["start"]) / total_original * total_budget_seconds)
        for t in topics
    ]
    total_raw = sum(raw)
    if total_raw > total_budget_seconds:
        scale = total_budget_seconds / total_raw
        raw = [r * scale for r in raw]
    return raw


def _score_topic_segment_batch(topic: dict, batch_indices: list[int], segments: list[dict], max_retries: int = MAX_RETRIES) -> dict[int, int]:
    numbered = "\n".join(
        f"[{i}] ({segments[i]['start']:.1f}s-{segments[i]['end']:.1f}s) {segments[i]['text']}"
        for i in batch_indices
    )

    prompt = f"""You are selecting which parts of a lecture segment on the topic
"{topic['title']}" should be kept for a condensed summary video.

{numbered}

Score each segment 1-10 on how well it explains or represents this
specific topic. Give LOW scores to examples, tangents, repeated points,
silence-heavy stretches, or administrative talk mixed into this stretch.

Return ONLY a JSON object mapping segment index (as a string) to score.
Example: {{"12": 9, "15": 7}}
No explanation, no markdown formatting.
"""

    client = _get_client()
    last_error = None
    batch_index_set = set(batch_indices)

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
            return {idx: score for idx, score in scores.items() if idx in batch_index_set}
        except Exception as e:
            last_error = e
            if attempt < max_retries - 1:
                wait = SECONDS_BETWEEN_CALLS * (2 ** attempt)
                print(f"Topic segment scoring error ({e}), retrying in {wait}s... (attempt {attempt + 1}/{max_retries})")
                time.sleep(wait)

    print(f"Scoring failed after {max_retries} attempts for topic '{topic['title']}' batch: {last_error}")
    return {}


def _score_topic_segments(topic: dict, seg_indices: list[int], segments: list[dict]) -> dict[int, int]:
    sub_batches = []
    current: list[int] = []
    current_chars = 0

    for idx in seg_indices:
        seg_chars = len(segments[idx]["text"]) + 40
        if current and current_chars + seg_chars > WINDOW_TARGET_CHARS:
            sub_batches.append(current)
            current, current_chars = [], 0
        current.append(idx)
        current_chars += seg_chars

    if current:
        sub_batches.append(current)

    all_scores: dict[int, int] = {}
    for b_num, batch_indices in enumerate(sub_batches):
        all_scores.update(_score_topic_segment_batch(topic, batch_indices, segments))
        if b_num < len(sub_batches) - 1:
            time.sleep(SECONDS_BETWEEN_CALLS)

    return all_scores


def _select_within_budget(indices: list[int], segments: list[dict], scores: dict[int, int], budget_seconds: float) -> list[int]:
    ranked = sorted(indices, key=lambda i: (-scores.get(i, 0), i))
    kept, total = [], 0.0
    for idx in ranked:
        dur = segments[idx]["end"] - segments[idx]["start"]
        if total + dur > budget_seconds:
            continue
        kept.append(idx)
        total += dur
    return kept


def generate_summary_video(video_id: str, video_path: Path, output_path: Path) -> tuple[Path | None, dict]:
    segments = _load_segments(video_id)
    if not segments:
        return None, {"error": "No transcript segments found."}

    original_duration = segments[-1]["end"] - segments[0]["start"]

    windows = _window_segments(segments)
    raw_topics, windows_failed = _propose_all_topics(windows)
    if not raw_topics:
        return None, {"error": "No topics could be identified.", "windows_failed": windows_failed}

    topics = _merge_topics(raw_topics)
    if not topics:
        return None, {"error": "Topic merge produced no usable topics.", "windows_failed": windows_failed}

    topic_assignments, uncovered_indices = _assign_segments_to_topics(segments, topics)

    total_budget = TARGET_SUMMARY_MIN_MINUTES * 60
    topic_budgets = _allocate_topic_budgets(topics, total_budget)

    kept_indices = []
    topics_scoring_failed = 0

    for topic, seg_indices, budget in zip(topics, topic_assignments, topic_budgets):
        if not seg_indices:
            continue

        if len(seg_indices) <= MIN_SEGMENTS_TO_SCORE:
            scores = {i: 10 for i in seg_indices}
        else:
            scores = _score_topic_segments(topic, seg_indices, segments)
            if not scores:
                topics_scoring_failed += 1
                scores = {i: 5 for i in seg_indices}

        kept_indices.extend(_select_within_budget(seg_indices, segments, scores, budget))

    kept_indices = sorted(set(kept_indices))
    kept_segments = [segments[i] for i in kept_indices]

    if not kept_segments:
        return None, {"error": "No segments were selected for the summary."}

    timestamps = [{"start": s["start"], "end": s["end"]} for s in kept_segments]

    # --- cut/stitch, then a post-processing silence-removal pass on the
    # STITCHED output (see module docstring for why whisper_gaps can't
    # be used here) ---
    post_silence_ok = True
    with tempfile.TemporaryDirectory(prefix="lecturelens_summary_") as tmp_dir_str:
        tmp_dir = Path(tmp_dir_str)
        raw_stitched_path = tmp_dir / "raw_summary.mp4"

        stitched = cut_and_stitch(video_path, timestamps, raw_stitched_path)
        if stitched is None:
            return None, {"error": "Cutting/stitching the summary failed."}

        try:
            temp_audio_path = _extract_temp_audio(stitched, tmp_dir)
            final = remove_silence(stitched, temp_audio_path, output_path, method=POST_SILENCE_METHOD)
            if final is None:
                raise RuntimeError("remove_silence returned no output")
        except Exception as e:
            print(f"Post-processing silence removal failed ({e}), using the raw stitched summary instead.")
            shutil.copy(stitched, output_path)
            post_silence_ok = False

    kept_duration = _get_duration_seconds(output_path)

    stats = {
        "original_minutes": round(original_duration / 60, 1),
        "summary_minutes": round(kept_duration / 60, 1),
        "segments_kept": len(kept_segments),
        "segments_total": len(segments),
        "segments_uncovered": len(uncovered_indices),
        "topics_found": len(topics),
        "windows_failed": windows_failed,
        "topics_scoring_failed": topics_scoring_failed,
        "post_silence_removed": post_silence_ok,
        "under_target_min": kept_duration < TARGET_SUMMARY_MIN_MINUTES * 60,
    }
    return output_path, stats


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