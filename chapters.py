import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal

import numpy as np
from dotenv import load_dotenv
from groq import BadRequestError as GroqBadRequestError
from langchain_groq import ChatGroq
from pydantic import BaseModel, Field, create_model

from chunking import QDRANT_PATH, load_chunks_for_video
from topics import TOPICS_QDRANT_PATH, get_topics_for_subject

load_dotenv()

# =============================
# CONFIG
# =============================
CHAPTERS_JSON_PATH = Path(__file__).parent / "chapter.json"

# Checkpoint dump written right after clusters get a topic assigned, but
# BEFORE the per-segment importance filter runs -- lets you eyeball the
# clustering + topic-assignment quality before the (slower, per-segment)
# filtering pass runs on top of it.
TOPICS_REVIEW_JSON_PATH = Path(__file__).parent / "topics_review.json"

# llama-3.3-70b-versatile was deprecated by Groq on 2026-06-17 and is
# scheduled to stop working entirely on 2026-08-16. openai/gpt-oss-120b is
# Groq's recommended production replacement.
GROQ_MODEL = "openai/gpt-oss-120b"

# openai/gpt-oss-120b is currently unreliable about honoring forced
# tool-calling on Groq: it sometimes answers in plain text instead of
# calling the structured-output tool, which Groq's API rejects with a
# 400 "tool_use_failed" error rather than retrying itself (known
# provider/model issue, not something wrong with our prompts/schemas --
# see langchain-ai/langchain#33995 and #34155, and Groq's own community
# forum thread "Structured Outputs ignored by openai/gpt-oss-120b" --
# their `response_format: json_schema` mode is *also* unreliable for
# this model, so switching structured-output "method" doesn't fix it).
# This is a real, currently-unresolved model/provider limitation, not
# something retries alone can fully paper over. So there are two layers:
#   1. Retry the forced tool call a few times with an extra nudge
#      (handles the common transient case).
#   2. If that's still failing, fall back to a plain, tool-free prompt
#      that asks for raw JSON and parses/validates it by hand -- a
#      structurally different path that doesn't depend on Groq's forced
#      tool-calling at all, so it can succeed even when (1) can't.
# Only if BOTH layers fail do we fall back to a safe default rather than
# letting one flaky generation crash the whole run.
STRUCTURED_OUTPUT_MAX_RETRIES = 3
JSON_FALLBACK_MAX_RETRIES = 2

# Similarity-drop segmentation tuning
BREAKPOINT_PERCENTILE = 92
SMOOTHING_WINDOW = 3
SUPPRESSION_RADIUS = 2
MIN_SEGMENT_CHUNKS = 6
MIN_SEGMENT_SECONDS = 45.0
TARGET_SEGMENT_RANGE = (6, 20)

# How many chunks (in start-time order) from a cluster to sample when
# asking the LLM to classify/name it. Cheaper than sending the whole
# cluster, and clusters are merged up to at least MIN_SEGMENT_CHUNKS.
SAMPLE_CHUNKS_PER_CLUSTER = 6

# The per-segment "is this core content" call (is_segment_important) is a
# harder, more context-dependent judgment than the coarse Important/Filler
# cluster classification, so it gets its own model + reasoning_effort you
# can tune independently -- this is the one lever to reach for if you want
# to try a different model just for that step (e.g. a stronger model, or
# the same model with "high" effort) without touching classification or
# topic-assignment. Defaults to the same model as everything else.
SEGMENT_GROQ_MODEL = GROQ_MODEL
SEGMENT_REASONING_EFFORT = "medium"

# reasoning_format="parsed" and reasoning_effort="low" are load-bearing,
# not just tuning knobs: Groq's docs state reasoning_format must be
# "parsed" or "hidden" (never the default "raw") when a reasoning model
# is forced into tool calls / structured output -- "raw" interleaves
# <think>...</think> content into the response instead of a clean tool
# call, which is what was causing every single call to fail on the
# first attempt with tool_use_failed. "parsed" keeps the reasoning
# trace available in additional_kwargs.reasoning_content if you ever
# want to inspect it; switch to "hidden" if you don't need that.
llm = ChatGroq(
    model=GROQ_MODEL,
    temperature=0.2,
    api_key=os.getenv("GROQ_API_KEY"),
    reasoning_format="parsed",
    reasoning_effort="low",
)

# See SEGMENT_GROQ_MODEL / SEGMENT_REASONING_EFFORT above -- separate
# instance so the per-segment importance call can be tuned (or swapped to
# a different model) independently of classification/topic-assignment.
segment_llm = ChatGroq(
    model=SEGMENT_GROQ_MODEL,
    temperature=0.2,
    api_key=os.getenv("GROQ_API_KEY"),
    reasoning_format="parsed",
    reasoning_effort=SEGMENT_REASONING_EFFORT,
)


# =============================
# Structured output schemas
# =============================
class ClusterCategory(BaseModel):
    category: Literal["Important", "Filler"] = Field(
        description=(
            "Important = concepts, explanations, definitions, algorithms, "
            "examples, derivations, formulas, reasoning. Filler = greetings, "
            "jokes, introductions, announcements, pauses, administrative talk, "
            "repeated statements, attendance, off-topic discussion, scolding "
            "the students."
        )
    )


class SegmentImportance(BaseModel):
    importance: Literal["high", "medium", "low"] = Field(
        description=(
            "high = the core statement of the topic itself -- the actual "
            "definition, formula, derivation step, or explanation of the "
            "concept.\n"
            "medium = still genuinely part of teaching the topic, even if "
            "it isn't the core statement -- a supporting detail, "
            "clarification, restatement in different words, elaboration, "
            "or a transitional step of reasoning that helps build the "
            "explanation. When in doubt between medium and low, prefer "
            "medium: it's better to keep a borderline-relevant segment "
            "than to lose real teaching content.\n"
            "low = a worked example used purely to illustrate the concept, "
            "a tangent, filler (greetings, administrative talk, pauses), "
            "or a near-exact repeat of something already covered earlier "
            "in this same cluster."
        )
    )


structured_llm_category = llm.with_structured_output(ClusterCategory)
structured_llm_importance = segment_llm.with_structured_output(SegmentImportance)

# Dynamic per-subject "must pick exactly one of these topics" models are
# built lazily and cached, since the allowed topic list only depends on
# subject_code and is fixed for the lifetime of one generate_chapters() call.
_topic_model_cache: dict[tuple, type[BaseModel]] = {}


def _get_topic_choice_model(topics: list[str]) -> type[BaseModel]:
    key = tuple(sorted(topics))
    if key not in _topic_model_cache:
        _topic_model_cache[key] = create_model(
            "TopicChoice",
            topic=(
                Literal[tuple(topics)],
                Field(description="The single best-matching topic from the allowed list for this content."),
            ),
        )
    return _topic_model_cache[key]


# =============================
# Retry wrapper for flaky forced tool-calling (see STRUCTURED_OUTPUT_MAX_RETRIES above)
# =============================
def _extract_json_object(text: str) -> str:
    """
    Best-effort cleanup of a model reply that's supposed to be raw JSON
    but may be wrapped in markdown code fences or have stray text around
    it (reasoning models especially like to add a stray sentence before
    or after the JSON even when told not to).
    """
    text = text.strip()

    if "```" in text:
        # Grab the contents of the first fenced block, if any.
        parts = text.split("```")
        if len(parts) >= 2:
            fenced = parts[1]
            if fenced.lower().startswith("json"):
                fenced = fenced[4:]
            text = fenced.strip()

    # If there's still leading/trailing prose, narrow to the outermost
    # {...} span -- the schema is always a single JSON object here.
    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1 and end > start:
        text = text[start : end + 1]

    return text


def _manual_json_fallback(
    raw_llm,
    schema: type[BaseModel],
    prompt: str,
    *,
    max_retries: int = JSON_FALLBACK_MAX_RETRIES,
):
    """
    Last-resort fallback for when forced tool-calling has failed every
    retry (see _invoke_structured_with_retry below). Rather than trying
    the same forced-tool-call path again -- which is exactly the path
    that's been failing -- this calls the model with NO tool binding at
    all and simply asks it to reply with raw JSON matching the schema,
    then parses and validates that JSON by hand with Pydantic. It's a
    structurally different request, so it isn't subject to the same
    "tool_use_failed" 400 that Groq raises for forced tool calls, and it
    gives genuinely flaky generations one more, different-shaped chance
    to succeed before the caller's safe default is used.

    Returns a validated instance of `schema`, or None if every attempt
    still failed to produce parseable, schema-valid JSON.
    """
    schema_json = json.dumps(schema.model_json_schema(), indent=2)
    current_prompt = (
        f"{prompt}\n\n"
        f"Respond with ONLY a single JSON object matching this JSON schema "
        f"-- no markdown code fences, no explanation, no text before or "
        f"after it:\n{schema_json}"
    )

    for attempt in range(1, max_retries + 1):
        try:
            response = raw_llm.invoke(current_prompt)
            content = response.content if isinstance(response.content, str) else str(response.content)
            data = json.loads(_extract_json_object(content))
            return schema.model_validate(data)
        except (json.JSONDecodeError, ValueError) as e:
            print(
                f"    [json-fallback {attempt}/{max_retries}] reply wasn't "
                f"parseable/valid JSON for the schema: {e}"
            )
            current_prompt = (
                f"{current_prompt}\n\n"
                f"Your previous reply could not be parsed as valid JSON matching "
                f"the schema. Reply again with ONLY the JSON object and nothing else."
            )

    return None


def _invoke_structured_with_retry(
    structured_llm,
    prompt: str,
    *,
    fallback,
    raw_llm,
    schema: type[BaseModel],
    max_retries: int = STRUCTURED_OUTPUT_MAX_RETRIES,
):
    """
    Calls structured_llm.invoke(prompt), retrying with an extra nudge if
    Groq rejects the response with a 400 "tool_use_failed" error (the
    model answered in plain text instead of calling the tool). Any
    other error is re-raised immediately -- this only swallows the one
    known-flaky failure mode.

    If every forced-tool-call retry still fails, this drops down to
    _manual_json_fallback (a tool-free, plain-JSON request against the
    same `raw_llm`/`schema`) before giving up entirely. Only if that
    *also* fails is `fallback` returned, so the caller can decide what
    "give up gracefully" means for that particular call site.
    """
    # NOTE: the nudge below is included starting on attempt 1, not just
    # after a failure. In practice openai/gpt-oss-120b almost never
    # complies with a forced tool call on the very first, un-nudged
    # attempt -- it reliably needs this explicit instruction -- so
    # waiting for a failure before adding it was just paying for a
    # doomed extra API call on nearly every single invocation.
    tool_call_nudge = (
        f"\n\nIMPORTANT: You must respond ONLY by calling the provided "
        f"function/tool with valid arguments. Do not reply with plain "
        f"text, commentary, or any other format."
    )
    current_prompt = prompt + tool_call_nudge
    last_error: Exception | None = None

    for attempt in range(1, max_retries + 1):
        try:
            return structured_llm.invoke(current_prompt)
        except GroqBadRequestError as e:
            body = getattr(e, "body", None) or {}
            error_code = (body.get("error") or {}).get("code") if isinstance(body, dict) else None

            if error_code != "tool_use_failed":
                raise  # a different error -- don't mask it, let it surface

            last_error = e
            print(
                f"    [retry {attempt}/{max_retries}] model replied with plain text "
                f"instead of calling the tool -- retrying."
            )

    print(
        f"    Forced tool-calling failed after {max_retries} attempt(s) "
        f"({last_error}) -- trying a tool-free plain-JSON fallback."
    )
    result = _manual_json_fallback(raw_llm, schema, prompt)
    if result is not None:
        print("    [json-fallback] succeeded.")
        return result

    print("    JSON fallback also failed -- using the safe default for this call.")
    return fallback


# =============================
# LLM calls
# =============================
def classify_cluster(sample_text: str) -> str:
    """Returns 'important' or 'filler' (lowercase)."""
    prompt = f"""You are analyzing a cluster of consecutive lecture transcript segments.

Determine whether this cluster contains IMPORTANT educational content
(concepts, explanations, definitions, algorithms, examples, derivations,
formulas, reasoning) or is FILLER (greetings, jokes, introductions,
announcements, pauses, administrative talk, repeated statements,
off-topic discussion).

TEXT:
{sample_text}
"""
    # Fallback: treat unresolvable clusters as Filler (dropped). Losing
    # a genuinely-important cluster to a flaky retry is recoverable
    # (rerun with force=True later); crashing the whole video isn't.
    result = _invoke_structured_with_retry(
        structured_llm_category,
        prompt,
        fallback=ClusterCategory(category="Filler"),
        raw_llm=llm,
        schema=ClusterCategory,
    )
    return result.category.lower()


def assign_topic(sample_text: str, allowed_topics: list[str]) -> str | None:
    """
    Forces the LLM to pick exactly one topic from `allowed_topics` --
    the full topic list stored for this subject_code in topics.py's
    Qdrant DB. No topic outside that list can ever be returned, since
    the response schema's Literal is built directly from the list.

    Returns None if the model still won't comply after retries -- there
    is no safe "default topic" to fall back to, so the caller is
    responsible for skipping this cluster rather than mislabeling it.
    """
    Model = _get_topic_choice_model(allowed_topics)
    structured = llm.with_structured_output(Model)

    prompt = f"""You are labeling a cluster of consecutive lecture transcript segments
with the single best-matching topic from a fixed, allowed list.

You MUST choose exactly one topic from this list -- do not invent a new
one, do not modify the wording:
{json.dumps(allowed_topics, indent=2)}

Pick whichever listed topic this text is most centrally about.

TEXT:
{sample_text}
"""
    result = _invoke_structured_with_retry(
        structured,
        prompt,
        fallback=None,
        raw_llm=llm,
        schema=Model,
    )
    return result.topic if result is not None else None


def _build_marked_cluster_context(cluster: list[dict], marked_idx: int) -> str:
    """
    Renders the full cluster as one block of text with the segment at
    marked_idx wrapped in >>> <<< markers, so the model judges that one
    segment WITH the surrounding material visible, instead of the bare
    isolated sentence it would otherwise see.
    """
    lines = []
    for i, c in enumerate(cluster):
        lines.append(f">>> {c['text']} <<<" if i == marked_idx else c["text"])
    return "\n".join(lines)


def is_segment_important(segment_text: str, topic: str, cluster_context: str) -> bool:
    """
    Per-segment pass INSIDE an already topic-assigned cluster: keep
    segments that are core teaching content for that topic. Judged with
    the surrounding cluster as context (see _build_marked_cluster_context)
    rather than the bare sentence in isolation -- a single short
    transcript chunk very often only reads as "central" once you can see
    what leads into and out of it; judging chunks with zero context is
    what was causing this filter to keep only 1-2 segments per topic even
    when a cluster's explanation clearly spanned several consecutive
    chunks. Worked examples/illustrations are still excluded, per spec
    ("no need to add examples") -- only the surrounding-context awareness
    changed, not that rule.

    Uses a three-tier "high / medium / low" importance judgment rather
    than a single binary keep/drop call -- both "high" (the core
    statement of the topic) and "medium" (supporting detail,
    clarification, elaboration, transitional reasoning) count as
    important and are kept; only "low" (examples, tangents, filler,
    near-exact repeats) is dropped. A plain yes/no tends to collapse
    everything that isn't the single most central sentence down to "no",
    under-keeping genuinely relevant supporting material; the middle
    tier gives the model room to say "this is part of teaching the
    topic, just not the core statement" instead of being forced to call
    it filler, so more real teaching content survives the filter.
    """
    prompt = f"""You are looking at one specific segment (marked with >>> <<<) inside
a larger cluster of consecutive lecture transcript segments, all grouped
under the topic "{topic}". The rest of the cluster is shown for context so
you can judge the marked segment correctly -- you are deciding about the
MARKED segment only, not the whole cluster.

FULL CLUSTER:
{cluster_context}

MARKED SEGMENT:
{segment_text}

Rate how important the MARKED segment is to teaching "{topic}":

- high: the core statement itself -- a definition, explanation, derivation
  step, or formula for the concept.
- medium: still genuinely part of teaching the topic even though it isn't
  the core statement -- a supporting detail, clarification, restatement,
  elaboration, or transitional step of reasoning. If you're unsure between
  medium and low, choose medium.
- low: only for a worked example used purely to illustrate the concept, a
  tangent, filler (greetings, administrative talk, pauses), or a
  near-exact repeat of something already covered earlier in this same
  cluster.

Segments naturally build on each other, so several consecutive segments in
the same cluster can all be high or medium.
"""
    # Fallback (used only if BOTH the forced tool call and the manual
    # JSON fallback fail -- a technical failure, not a judgment call):
    # treat it as "low"/dropped, consistent with "when a call can't be
    # resolved at all, don't guess -- drop it and let a rerun catch it."
    result = _invoke_structured_with_retry(
        structured_llm_importance,
        prompt,
        fallback=SegmentImportance(importance="low"),
        raw_llm=segment_llm,
        schema=SegmentImportance,
    )
    return result.importance in ("high", "medium")


# =============================
# Similarity-drop segmentation (clusters chunks into topical groups)
# =============================
def cosine_sim(a, b):
    a = np.asarray(a)
    b = np.asarray(b)
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))


def windowed_embedding(chunks, center_idx, window, side):
    if side == "left":
        lo = max(0, center_idx - window + 1)
        group = chunks[lo : center_idx + 1]
    else:
        hi = min(len(chunks), center_idx + 1 + window)
        group = chunks[center_idx + 1 : hi]
    return np.mean([c["embedding"] for c in group], axis=0)


def compute_distance_signal(chunks, window):
    distances = []
    for i in range(len(chunks) - 1):
        left = windowed_embedding(chunks, i, window, "left")
        right = windowed_embedding(chunks, i, window, "right")
        distances.append(1 - cosine_sim(left, right))
    return distances


def find_breakpoints(distances, percentile, suppression_radius):
    if not distances:
        return set()

    threshold = np.percentile(distances, percentile)
    candidates = [i for i, d in enumerate(distances) if d > threshold]

    kept = []
    for idx in candidates:
        lo = max(0, idx - suppression_radius)
        hi = min(len(distances), idx + suppression_radius + 1)
        window_slice = distances[lo:hi]
        if distances[idx] == max(window_slice):
            if not kept or idx - kept[-1] > suppression_radius:
                kept.append(idx)

    return {i + 1 for i in kept}


def build_clusters(chunks, breakpoints):
    clusters = []
    current = [chunks[0]]
    for i in range(1, len(chunks)):
        if i in breakpoints:
            clusters.append(current)
            current = [chunks[i]]
        else:
            current.append(chunks[i])
    if current:
        clusters.append(current)
    return clusters


def cluster_duration(cluster):
    return cluster[-1]["end"] - cluster[0]["start"]


def merge_small_clusters(clusters, min_chunks, min_seconds):
    if len(clusters) <= 1:
        return clusters

    merged = [list(c) for c in clusters]
    changed = True

    while changed:
        changed = False
        for idx, c in enumerate(merged):
            too_small = len(c) < min_chunks or cluster_duration(c) < min_seconds
            if not too_small or len(merged) == 1:
                continue

            if idx == 0:
                merged[idx + 1] = c + merged[idx + 1]
                merged.pop(idx)
            elif idx == len(merged) - 1:
                merged[idx - 1] = merged[idx - 1] + c
                merged.pop(idx)
            else:
                c_mean = np.mean([x["embedding"] for x in c], axis=0)
                prev_mean = np.mean([x["embedding"] for x in merged[idx - 1]], axis=0)
                next_mean = np.mean([x["embedding"] for x in merged[idx + 1]], axis=0)

                if cosine_sim(c_mean, prev_mean) >= cosine_sim(c_mean, next_mean):
                    merged[idx - 1] = merged[idx - 1] + c
                else:
                    merged[idx + 1] = c + merged[idx + 1]
                merged.pop(idx)

            changed = True
            break

    return merged


def cluster_transcript(chunks, percentile, window, suppression_radius, min_chunks, min_seconds):
    distances = compute_distance_signal(chunks, window)
    breakpoints = find_breakpoints(distances, percentile, suppression_radius)
    raw_clusters = build_clusters(chunks, breakpoints)
    return merge_small_clusters(raw_clusters, min_chunks, min_seconds)


def auto_tune_clustering(chunks, target_range, **kwargs):
    lo_target, hi_target = target_range
    percentile = kwargs.pop("percentile")
    best = None

    for _ in range(15):
        clusters = cluster_transcript(chunks, percentile=percentile, **kwargs)
        count = len(clusters)

        if lo_target <= count <= hi_target:
            return clusters

        mid = (lo_target + hi_target) / 2
        if best is None or abs(count - mid) < abs(best[1] - mid):
            best = (clusters, count)

        if count > hi_target:
            percentile = min(99, percentile + 2)
        else:
            percentile = max(50, percentile - 2)

    return best[0]


# =============================
# JSON DB helpers (shared shape for both chapter.json and the review file)
# =============================
def _load_json_db(path) -> dict:
    if not os.path.exists(path):
        return {}
    with open(path, "r", encoding="utf-8") as f:
        try:
            return json.load(f)
        except json.JSONDecodeError:
            print(f"WARNING: {path} was not valid JSON -- starting fresh.")
            return {}


def _save_json_db(db: dict, path) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(db, f, indent=2, ensure_ascii=False)


# =============================
# Main entry point
# =============================
def generate_chapters(
    video_url: str,
    subject_code: str,
    force: bool = False,
    qdrant_path: str = QDRANT_PATH,
    topics_qdrant_path: str = TOPICS_QDRANT_PATH,
    chapters_path=CHAPTERS_JSON_PATH,
    review_path=TOPICS_REVIEW_JSON_PATH,
):
    """
    Pipeline for one video
    -----------------------
    1. Load this video's chunks from Qdrant (chunking.py's collection).
    2. Similarity-drop clustering to group chunks into topical clusters.
    3. Classify each cluster Important/Filler. Filler clusters are
       dropped entirely -- only Important clusters continue.
    4. For each Important cluster, force the LLM to pick exactly one
       topic from `subject_code`'s allowed topic list (topics.py). No
       topic outside that list can ever be produced.
    5. Checkpoint: dump every (topic, cluster) assignment made so far to
       `topics_review.json`, keyed by video_url, so you can review/
       sanity-check clustering + topic-assignment quality before the
       (slower) per-segment filtering pass below runs.
    6. Per-segment filter: within each Important, topic-assigned
       cluster, ask the LLM about EVERY individual chunk -- shown with
       its cluster as surrounding context, not in isolation -- and keep
       it if it's core teaching content for that topic (examples/
       tangents/filler are dropped, never kept).
    7. Group surviving segments by topic name (the same topic can
       legitimately recur in separate clusters spread across a video --
       e.g. revisited later in a recap), and save the final result to
       `chapter.json`, keyed by video_url:

           {
             "<video_url>": [
               {
                 "topic": "...",
                 "segments": [
                   {"id": 3, "start": 41.2, "end": 63.8},
                   ...
                 ]
               },
               ...
             ]
           }

       Only "id"/"start"/"end" are stored per segment -- the segment
       text itself already lives in Qdrant (chunking.py), keyed by the
       same video_url + segment id, so it isn't duplicated here.

    Parameters
    ----------
    video_url : str
        Key used to look up chunks in Qdrant and to store/find this
        video's entry in chapter.json.
    subject_code : str
        Selects which allowed topic list (from topics.py's Qdrant DB) to
        force cluster topic-naming against. Raises if nothing is stored
        for this subject_code.
    force : bool
        Regenerate even if video_url already has a chapter.json entry.
    """
    chapters_db = _load_json_db(chapters_path)
    if video_url in chapters_db and not force:
        print(f"'{video_url}' already has chapters generated -- skipping.")
        print("Pass force=True to regenerate.")
        return chapters_db[video_url]

    allowed_topics = get_topics_for_subject(subject_code, topics_qdrant_path)
    if not allowed_topics:
        raise ValueError(
            f"No topics found for subject_code={subject_code!r}. "
            f"Run topics.py (upload_topics) to seed that subject's topic list first."
        )

    chunks = load_chunks_for_video(video_url, qdrant_path)
    if not chunks:
        raise ValueError(
            f"No chunks found in Qdrant for video_url={video_url!r}. "
            f"Make sure chunking.chunk_transcript() has already been run for this video."
        )

    clusters = auto_tune_clustering(
        chunks,
        target_range=TARGET_SEGMENT_RANGE,
        percentile=BREAKPOINT_PERCENTILE,
        window=SMOOTHING_WINDOW,
        suppression_radius=SUPPRESSION_RADIUS,
        min_chunks=MIN_SEGMENT_CHUNKS,
        min_seconds=MIN_SEGMENT_SECONDS,
    )
    print(f"Clustered {len(chunks)} chunk(s) into {len(clusters)} cluster(s).")

    # ----------------------------------
    # Classify + assign topic (Important clusters only)
    # ----------------------------------
    review_entries = []
    important_clusters = []  # list[(topic, cluster_chunks)]
    skipped_clusters = 0

    for cluster in clusters:
        sample = "\n".join(c["text"] for c in cluster[:SAMPLE_CHUNKS_PER_CLUSTER])

        category = classify_cluster(sample)
        if category != "important":
            continue

        topic = assign_topic(sample, allowed_topics)
        if topic is None:
            # Model never complied with the forced tool call even after
            # retries -- skip this cluster rather than mislabel it.
            skipped_clusters += 1
            print(
                f"  Skipping a cluster ({cluster[0]['start']:.1f}s-{cluster[-1]['end']:.1f}s): "
                f"could not get a topic assignment after retries."
            )
            continue

        important_clusters.append((topic, cluster))

        review_entries.append(
            {
                "topic": topic,
                "start": round(cluster[0]["start"], 2),
                "end": round(cluster[-1]["end"], 2),
                "chunk_ids": [c["id"] for c in cluster],
            }
        )

    print(f"{len(important_clusters)} of {len(clusters)} cluster(s) classified Important.")
    if skipped_clusters:
        print(f"{skipped_clusters} cluster(s) skipped due to unresolvable topic-assignment failures.")

    # ----------------------------------
    # Checkpoint dump -- BEFORE per-segment filtering, so it reflects
    # raw clustering + topic-assignment quality only.
    # ----------------------------------
    review_db = _load_json_db(review_path)
    review_db[video_url] = review_entries
    _save_json_db(review_db, review_path)
    print(f"Saved topic-assignment review data to {review_path}")

    # ----------------------------------
    # Per-segment importance filter, inside each Important cluster
    # ----------------------------------
    topic_to_segments: dict[str, list[dict]] = {}

    for topic, cluster in important_clusters:
        for i, chunk in enumerate(cluster):
            context = _build_marked_cluster_context(cluster, i)
            if not is_segment_important(chunk["text"], topic, context):
                continue
            topic_to_segments.setdefault(topic, []).append(
                {
                    "id": chunk["id"],
                    "start": round(chunk["start"], 2),
                    "end": round(chunk["end"], 2),
                }
            )

    kept_count = sum(len(v) for v in topic_to_segments.values())
    total_count = sum(len(cluster) for _, cluster in important_clusters)
    print(f"Kept {kept_count} of {total_count} chunk(s) after per-segment importance filtering.")

    # Same topic can legitimately recur across separate (non-adjacent)
    # clusters -- e.g. a concept revisited in a later recap -- so all
    # segments for a topic are grouped together into one entry rather
    # than kept as separate per-cluster entries.
    final_chapters = [
        {
            "topic": topic,
            "segments": sorted(segments, key=lambda s: s["start"]),
        }
        for topic, segments in topic_to_segments.items()
    ]
    final_chapters.sort(key=lambda c: c["segments"][0]["start"] if c["segments"] else float("inf"))

    chapters_db[video_url] = final_chapters
    _save_json_db(chapters_db, chapters_path)

    print(f"Saved {len(final_chapters)} topic(s) to {chapters_path} for {video_url}")

    return final_chapters