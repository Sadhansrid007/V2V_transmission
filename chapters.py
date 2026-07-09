import os
import re
import json
import numpy as np
import chromadb

from datetime import datetime, timezone
from typing import Literal

from pathlib import Path

from dotenv import load_dotenv
from pydantic import BaseModel, Field
from langchain_groq import ChatGroq

load_dotenv()

# =============================
# CONFIG
# =============================
# These MUST match the path/collection name used in chunking.py, or this
# script will find zero chunks for every video_url.
CHROMA_PATH = "./chroma_db"
COLLECTION_NAME = "lecture_transcript"


CHAPTERS_JSON_PATH = Path(__file__).parent / "chapters.json"

# Fallback source of silence data ONLY for standalone/CLI use. main.py's
# intended flow calls ingestion() once and passes its "silences" list
# straight into generate_chapters(silences=...), so this file is never
# touched in that flow. This constant only matters if generate_chapters()
# is called without an explicit `silences` list.
SILENCE_JSON_PATH = "silence.json"

# Fixed label for pure-silence stretches (no speech transcribed at all --
# "the mic wasn't even picking anything up"). Assigned directly, never
# passed through the LLM, so it's byte-identical across every video and a
# frontend can filter/style on it without fuzzy matching.
MIC_OFF_TOPIC = "Mic Off"

# llama-3.3-70b-versatile was deprecated by Groq on 2026-06-17 and is
# scheduled to stop working entirely on 2026-08-16. openai/gpt-oss-120b is
# Groq's recommended production replacement.
GROQ_MODEL = "openai/gpt-oss-120b"

# Segmentation tuning (unchanged from your working version)
BREAKPOINT_PERCENTILE = 92
SMOOTHING_WINDOW = 3
SUPPRESSION_RADIUS = 2
MIN_SEGMENT_CHUNKS = 6
MIN_SEGMENT_SECONDS = 45.0
TARGET_SEGMENT_RANGE = (6, 20)

llm = ChatGroq(model=GROQ_MODEL, temperature=0.2, api_key=os.getenv("GROQ_API_KEY"))


# =============================
# Structured output schema for classification
# =============================
class SegmentClassification(BaseModel):
    category: Literal["Important", "Filler"] = Field(
        description=(
            "Important = concepts, explanations, definitions, algorithms, "
            "examples, derivations, formulas, reasoning. Filler = greetings, "
            "jokes, introductions, announcements, pauses, administrative talk, "
            "repeated statements,attendence, off-topic discussion, scolding the students."
        )
    )
    topic: str = Field(
        description=(
            "A precise, standardized topic name, Title Case, 2-6 words, no "
            "punctuation.\n"
            "\n"
            "If category is Important: name the SPECIFIC concept, technique, "
            "algorithm, or formula being taught -- the term a textbook index "
            "or course syllabus would use for it. Do not use vague/filler "
            "words ('Overview', 'Basics', 'Introduction To', 'Discussion') "
            "unless the section genuinely surveys many unrelated concepts at "
            "once with no single dominant one. Do not name the speaker's "
            "phrasing, the course name, or the broad subject area if a more "
            "specific concept applies.\n"
            "  Good: 'Binary Search Trees', 'Preprocessor Directives', "
            "\"Dijkstra's Shortest Path\", 'Recursive Base Cases'.\n"
            "  Bad: 'Talking About Trees Today', 'Hash Include Stuff', "
            "'Graph Algorithm Example', 'Today's Topic'.\n"
            "\n"
            "If category is Filler: a short descriptive label for what kind "
            "of filler it is, e.g. 'Class Announcements', 'Attendance Check', "
            "'Student Small Talk'."
        )
    )


# with_structured_output uses Groq's tool-calling under the hood, so this
# reliably returns a SegmentClassification object instead of text you have
# to regex-parse.
structured_llm = llm.with_structured_output(SegmentClassification)


# =============================
# Load one video's chunks from ChromaDB
# =============================
def load_chunks_for_video(video_url: str):
    """
    Pulls only this video's stored chunks (filtered by the video_url
    metadata set in chunking.py), restores chronological order, and
    returns them with their embeddings attached as numpy arrays.
    """
    client = chromadb.PersistentClient(path=CHROMA_PATH)
    collection = client.get_collection(COLLECTION_NAME)

    data = collection.get(
        where={"video_url": video_url},
        include=["embeddings", "documents", "metadatas"],
    )

    documents = data["documents"]
    metadatas = data["metadatas"]

    if not documents:
        return []

    embeddings = np.array(data["embeddings"])

    order = sorted(
        range(len(documents)),
        key=lambda i: (metadatas[i]["start_time"], metadatas[i].get("chunk_index", 0)),
    )

    return [
        {
            "text": documents[i],
            "start_time": round(metadatas[i]["start_time"], 2),
            "end_time": round(metadatas[i]["end_time"], 2),
            "duration": round(metadatas[i]["end_time"] - metadatas[i]["start_time"], 2),
            "chunk_index": metadatas[i].get("chunk_index"),
            "embedding": embeddings[i],
        }
        for i in order
    ]


# =============================
# Silence ranges -> filler chapters
# =============================
def _load_silences_from_path(path: str) -> list:
    """
    Fallback loader for standalone/CLI use, reading the shape ingest.py's
    __main__ block writes to silence.json. Not used when `silences` is
    passed directly to generate_chapters() (the main.py flow).

    Per your workflow, silence.json is deleted after each video is
    processed, so whatever silence.json exists on disk at call time is
    assumed to belong to the video currently being processed -- this does
    NOT filter/match by video_url internally.
    """
    if not path or not os.path.exists(path):
        return []
    with open(path, "r", encoding="utf-8") as f:
        try:
            data = json.load(f)
        except json.JSONDecodeError:
            print(f"WARNING: {path} was not valid JSON -- proceeding with no silence data.")
            return []
    return data.get("silences", [])


def _silence_to_chapters(silences: list) -> list:
    """
    Converts recorded silence ranges (stretches where no speech was
    transcribed at all -- the mic was effectively off) into chapter-shaped
    filler entries, so they show up in the same chronological chapter list
    as real content instead of being an invisible gap in the timeline.

    No LLM classification happens here: type and topic are fixed, and
    `chunks` is empty since no transcript chunk exists for pure silence.
    """
    chapters = []
    for s in silences:
        start = round(s["start"], 2)
        end = round(s["end"], 2)
        chapters.append(
            {
                "type": "filler",
                "topic": MIC_OFF_TOPIC,
                "start_time": start,
                "end_time": end,
                "duration": round(end - start, 2),
                "chunks": [],
            }
        )
    return chapters


# =============================
# Silence-aware hard boundaries
# =============================
def _gap_overlaps_silence(prev_end: float, next_start: float, silences: list) -> bool:
    """
    True if the time gap between two chronologically consecutive chunks
    overlaps a recorded silence range at all. Any positive overlap counts --
    a silence range is, by construction, a stretch with no transcribed
    speech, so if part of it falls in the gap between two chunks, that gap
    is a real silence break and must not be bridged by a segment.
    """
    gap_start, gap_end = prev_end, next_start
    if gap_end <= gap_start:
        return False
    for s in silences:
        overlap = min(gap_end, s["end"]) - max(gap_start, s["start"])
        if overlap > 0:
            return True
    return False


def split_chunks_by_silence(chunks: list, silences: list) -> list:
    """
    Partitions the chronological chunk list into contiguous groups such
    that no group's chunks straddle a silence gap.

    This exists because segment_transcript()/build_segments() previously
    only looked at chunk-to-chunk embedding similarity, with zero awareness
    of wall-clock gaps. Two chunks separated by a two-minute silence could
    still end up in the same segment if their embeddings looked similar,
    producing a segment whose start_time/end_time span (min/max over its
    chunks) silently swallowed the silence in between -- and could even
    overlap a neighboring segment's/silence chapter's time range in the
    final chapter list.

    By splitting BEFORE segmentation, each group is later segmented
    independently (see generate_chapters), so:
      - the embedding smoothing window (windowed_embedding) never averages
        across a silence gap,
      - merge_small_segments never merges a too-small segment into a
        neighbor on the other side of a silence gap,
      - a segment's start/end can never span a silence range.

    Returns a list of chunk-lists (each list itself in chronological
    order); concatenating them back together reconstructs `chunks`.
    """
    if not chunks:
        return []
    if not silences:
        return [chunks]

    groups = [[chunks[0]]]
    for i in range(1, len(chunks)):
        prev_end = chunks[i - 1]["end_time"]
        next_start = chunks[i]["start_time"]
        if _gap_overlaps_silence(prev_end, next_start, silences):
            groups.append([chunks[i]])
        else:
            groups[-1].append(chunks[i])

    return groups


def _scaled_target_range(group_size: int, total_size: int, target_range: tuple) -> tuple:
    """
    TARGET_SEGMENT_RANGE (e.g. 6-20) is calibrated for a whole video. Once
    the video is split into several silence-separated groups, applying that
    same absolute range to each group would over-segment small groups (a
    2-minute group doesn't need 6+ segments). Instead scale the target
    proportionally to how much of the video's total chunk count this group
    holds, with a floor of 1 segment.
    """
    lo, hi = target_range
    if total_size <= 0 or group_size <= 0:
        return (1, 1)

    frac = group_size / total_size
    scaled_lo = max(1, round(lo * frac))
    scaled_hi = max(scaled_lo, round(hi * frac))
    return (scaled_lo, scaled_hi)


# =============================
# Similarity-drop segmentation
# =============================
def cosine_sim(a, b):
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


def build_segments(chunks, breakpoints):
    segments = []
    current = [chunks[0]]
    for i in range(1, len(chunks)):
        if i in breakpoints:
            segments.append(current)
            current = [chunks[i]]
        else:
            current.append(chunks[i])
    if current:
        segments.append(current)
    return segments


def segment_duration(seg):
    return seg[-1]["end_time"] - seg[0]["start_time"]


def merge_small_segments(segments, min_chunks, min_seconds):
    if len(segments) <= 1:
        return segments

    merged = [list(seg) for seg in segments]
    changed = True

    while changed:
        changed = False
        for idx, seg in enumerate(merged):
            too_small = len(seg) < min_chunks or segment_duration(seg) < min_seconds
            if not too_small or len(merged) == 1:
                continue

            if idx == 0:
                merged[idx + 1] = seg + merged[idx + 1]
                merged.pop(idx)
            elif idx == len(merged) - 1:
                merged[idx - 1] = merged[idx - 1] + seg
                merged.pop(idx)
            else:
                seg_mean = np.mean([c["embedding"] for c in seg], axis=0)
                prev_mean = np.mean([c["embedding"] for c in merged[idx - 1]], axis=0)
                next_mean = np.mean([c["embedding"] for c in merged[idx + 1]], axis=0)

                if cosine_sim(seg_mean, prev_mean) >= cosine_sim(seg_mean, next_mean):
                    merged[idx - 1] = merged[idx - 1] + seg
                else:
                    merged[idx + 1] = seg + merged[idx + 1]
                merged.pop(idx)

            changed = True
            break

    return merged


def segment_transcript(chunks, percentile, window, suppression_radius, min_chunks, min_seconds):
    distances = compute_distance_signal(chunks, window)
    breakpoints = find_breakpoints(distances, percentile, suppression_radius)
    raw_segments = build_segments(chunks, breakpoints)
    return merge_small_segments(raw_segments, min_chunks, min_seconds)


def auto_tune_segmentation(chunks, target_range, **kwargs):
    lo_target, hi_target = target_range
    percentile = kwargs.pop("percentile")
    best = None

    for _ in range(15):
        segs = segment_transcript(chunks, percentile=percentile, **kwargs)
        count = len(segs)

        if lo_target <= count <= hi_target:
            return segs

        mid = (lo_target + hi_target) / 2
        if best is None or abs(count - mid) < abs(best[1] - mid):
            best = (segs, count)

        if count > hi_target:
            percentile = min(99, percentile + 2)
        else:
            percentile = max(50, percentile - 2)

    return best[0]


def segment_transcript_with_silence_boundaries(
    chunks,
    silences,
    target_range,
    percentile,
    window,
    suppression_radius,
    min_chunks,
    min_seconds,
):
    """
    Silence-aware replacement for calling auto_tune_segmentation() directly
    on the full chunk list. Splits chunks into silence-separated groups
    first (see split_chunks_by_silence), then runs the normal
    similarity-drop segmentation independently within each group and
    concatenates the results in chronological order.

    This guarantees no returned segment's start_time/end_time can span a
    silence gap, and that no two returned segments can overlap in time
    (since every group's chunks are a disjoint, contiguous chronological
    slice of `chunks`).
    """
    groups = split_chunks_by_silence(chunks, silences)
    total_size = len(chunks)

    all_segments = []
    for group in groups:
        if not group:
            continue

        if len(group) == 1:
            # A lone chunk between two silences (or at a video boundary) --
            # nothing to segment, it's its own segment.
            all_segments.append(group)
            continue

        group_target = _scaled_target_range(len(group), total_size, target_range)
        segs = auto_tune_segmentation(
            group,
            target_range=group_target,
            percentile=percentile,
            window=window,
            suppression_radius=suppression_radius,
            min_chunks=min_chunks,
            min_seconds=min_seconds,
        )
        all_segments.extend(segs)

    return all_segments


# =============================
# LLM classification (langchain-groq, structured output)
# =============================
def classify_segment(sample_text: str) -> SegmentClassification:
    prompt = f"""You are analyzing lecture transcript segments.

Determine whether this segment contains IMPORTANT educational content
(concepts, explanations, definitions, algorithms, examples, derivations,
formulas, reasoning) or is FILLER (greetings, jokes, introductions,
announcements, pauses, administrative talk, repeated statements,
off-topic discussion).

Also give the topic a standard, consistent name, Title Case, 2-6 words,
no punctuation.

If IMPORTANT: name the SPECIFIC concept/technique/algorithm/formula being
taught -- the term a textbook index or course syllabus would use. Avoid
vague words like "Overview", "Basics", "Introduction To", or "Discussion"
unless the segment truly surveys many unrelated concepts with no single
dominant one. Name the concept itself, not the speaker's phrasing, not the
course name.
  Good examples: "Binary Search Trees", "Preprocessor Directives",
  "Dijkstra's Shortest Path", "Recursive Base Cases".
  Bad examples: "Talking About Trees Today", "Hash Include Stuff",
  "Graph Algorithm Example", "Today's Topic".

If FILLER: a short descriptive label for the kind of filler, e.g.
"Class Announcements", "Attendance Check", "Student Small Talk".

TEXT:
{sample_text}
"""
    return structured_llm.invoke(prompt)


def standardize_topics(raw_topics: list) -> dict:
    """
    Asks the LLM to collapse near-duplicate topic labels (e.g. "Loops In
    Python" vs "Python Loop Basics") into one canonical name each. This uses
    a plain (non-structured) call since the output is a dynamic-keys JSON
    object, which doesn't map cleanly onto a fixed schema. Falls back to the
    original names if parsing fails.

    Only pass LLM-classified content/filler topics here -- NOT the fixed
    MIC_OFF_TOPIC label, which is injected after this step so it stays
    byte-identical across every video instead of risking being renamed or
    merged into an unrelated filler bucket.
    """
    unique_topics = sorted(set(raw_topics))
    if not unique_topics:
        return {}

    prompt = f"""Here is a list of topic labels extracted from lecture transcript segments.
Some may refer to the SAME underlying concept but are worded differently.

Produce a single standardized, canonical name for each group of equivalent
topics. Canonical names must be Title Case, 2-6 words, and use the precise
technical/academic term for the concept (the way a textbook index or
course syllabus would name it) rather than a vague or paraphrased label.

Return ONLY a JSON object mapping every ORIGINAL topic string (exactly as
given) to its STANDARDIZED topic string. No text outside the JSON, no
markdown code fences.

TOPICS:
{json.dumps(unique_topics, indent=2)}
"""
    raw = llm.invoke(prompt).content
    raw = re.sub(r"^```(json)?", "", raw, flags=re.IGNORECASE).strip()
    raw = re.sub(r"```$", "", raw).strip()

    try:
        mapping = json.loads(raw)
        return {t: mapping.get(t, t) for t in unique_topics}
    except (json.JSONDecodeError, TypeError):
        print("WARNING: topic standardization response wasn't valid JSON — keeping original topic names.")
        return {t: t for t in unique_topics}


# =============================
# Build one chronological chapter list (important + filler + silence)
# =============================
def build_chapters(chunks, segments, silences=None):
    # BUGFIX: timeline_start used to be hardcoded to 0.0, while
    # timeline_end was correctly computed from the real chunk data
    # (max end_time). That asymmetry meant a video whose real content
    # (and first chunk) only starts partway in — e.g. 10 minutes of
    # silence/pre-roll before the lecture begins — always reported
    # timeline_start=0.0 regardless of where the content actually
    # starts. Compute it the same way timeline_end already is: from
    # the earliest chunk's real start_time.
    timeline_start = round(min(c["start_time"] for c in chunks), 2)
    timeline_end = round(max(c["end_time"] for c in chunks), 2)

    content_chapters = []
    for seg_chunks in segments:
        sample = "\n".join(c["text"] for c in seg_chunks[:5])
        result = classify_segment(sample)

        starts = [c["start_time"] for c in seg_chunks]
        ends = [c["end_time"] for c in seg_chunks]

        content_chapters.append(
            {
                "type": result.category.lower(),  # "important" | "filler"
                "topic": result.topic,
                "start_time": round(min(starts), 2),
                "end_time": round(max(ends), 2),
                "duration": round(max(ends) - min(starts), 2),
                "chunks": [c["chunk_index"] for c in seg_chunks],
            }
        )

    # Standardize topic names across LLM-classified chapters only.
    topic_map = standardize_topics([c["topic"] for c in content_chapters])
    for c in content_chapters:
        c["topic"] = topic_map.get(c["topic"], c["topic"])

    # Silence -> "Mic Off" filler chapters, added after standardization so
    # the fixed label can't be touched by the LLM.
    silence_chapters = _silence_to_chapters(silences or [])

    all_chapters = content_chapters + silence_chapters
    all_chapters.sort(key=lambda c: c["start_time"])

    # Re-number chronologically now that content and silence are merged,
    # so segment_id is one clean, gap-free index over the whole timeline
    # instead of only counting LLM-classified segments.
    for idx, c in enumerate(all_chapters):
        c["segment_id"] = idx

    # Put segment_id first for readability in the saved JSON.
    all_chapters = [
        {
            "segment_id": c["segment_id"],
            "type": c["type"],
            "topic": c["topic"],
            "start_time": c["start_time"],
            "end_time": c["end_time"],
            "duration": c["duration"],
            "chunks": c["chunks"],
        }
        for c in all_chapters
    ]

    return all_chapters, timeline_start, timeline_end


# =============================
# Chapters DB — single JSON file keyed by video_url
# =============================
def _load_chapters_db(path: str) -> dict:
    if not os.path.exists(path):
        return {}
    with open(path, "r", encoding="utf-8") as f:
        try:
            return json.load(f)
        except json.JSONDecodeError:
            print(f"WARNING: {path} was not valid JSON — starting a fresh database.")
            return {}


def _save_chapters_db(db: dict, path: str):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(db, f, indent=2, ensure_ascii=False)


# =============================
# Main entry point
# =============================
def generate_chapters(
    video_url: str,
    force: bool = False,
    path: str = CHAPTERS_JSON_PATH,
    silences: list = None,
    silence_path: str = SILENCE_JSON_PATH,
):
    """
    Generates (or reuses) chapter data for one video and stores it in a
    single JSON file keyed by video_url — the shape a frontend can pull
    straight from:

        {
          "<video_url>": {
            "video_url": ...,
            "timeline_start": 0.0,
            "timeline_end": ...,
            "generated_at": "2026-07-06T12:34:56+00:00",
            "chapters": [
              {
                "segment_id": 0,
                "type": "important" | "filler",
                "topic": "...",   # "Mic Off" for pure-silence chapters
                "start_time": ..., "end_time": ..., "duration": ...,
                "chunks": [...]   # empty list for "Mic Off" chapters
              },
              ...
            ]
          },
          "<another_video_url>": { ... }
        }

    Parameters
    ----------
    video_url : str  -> key used to look this video up later, and to detect
                         whether it's already been processed
    force     : bool -> regenerate even if video_url already has an entry
    path      : str  -> where the chapters JSON lives
    silences  : list[dict] | None
        Silence ranges for THIS video, in the exact
        {"start": ..., "end": ..., "duration": ...} shape returned by
        ingestion()["silences"]. This is the expected path from main.py:
        call ingestion() once, get transcript + silences together, pass
        silences straight in here -- no silence.json round-trip needed.
        These silences are now also used as HARD segmentation boundaries
        (see split_chunks_by_silence) so a segment can never span a
        silence gap.
    silence_path : str
        Fallback used only when `silences` is not passed directly: reads
        the shape ingest.py's __main__ block writes to silence.json. Since
        you delete silence.json after each video is processed, whatever
        file is present when this runs is assumed to belong to `video_url`
        -- it is not filtered/matched by video_url internally.

    Returns
    -------
    dict: this video's entry (freshly generated, or the existing one if skipped)
    """
    db = _load_chapters_db(path)

    if video_url in db and not force:
        print(f"'{video_url}' already has chapters generated — skipping.")
        print("Pass force=True to regenerate.")
        return db[video_url]

    chunks = load_chunks_for_video(video_url)
    if not chunks:
        raise ValueError(
            f"No chunks found in ChromaDB for video_url={video_url!r}. "
            f"Make sure chunking.py has already been run for this video, and "
            f"that CHROMA_PATH/COLLECTION_NAME here match the ones used there."
        )

    if silences is None:
        silences = _load_silences_from_path(silence_path)

    segments = segment_transcript_with_silence_boundaries(
        chunks,
        silences=silences,
        target_range=TARGET_SEGMENT_RANGE,
        percentile=BREAKPOINT_PERCENTILE,
        window=SMOOTHING_WINDOW,
        suppression_radius=SUPPRESSION_RADIUS,
        min_chunks=MIN_SEGMENT_CHUNKS,
        min_seconds=MIN_SEGMENT_SECONDS,
    )

    chapters, timeline_start, timeline_end = build_chapters(chunks, segments, silences=silences)

    entry = {
        "video_url": video_url,
        "timeline_start": timeline_start,
        "timeline_end": round(timeline_end, 2),
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "chapters": chapters,
    }

    db[video_url] = entry
    _save_chapters_db(db, path)

    important_count = sum(1 for c in chapters if c["type"] == "important")
    mic_off_count = sum(1 for c in chapters if c["topic"] == MIC_OFF_TOPIC)
    filler_count = len(chapters) - important_count
    print(
        f"Generated {len(chapters)} chapters "
        f"({important_count} important, {filler_count} filler, "
        f"{mic_off_count} of which are Mic Off) for {video_url}"
    )
    print(f"Saved to {path}")

    return entry