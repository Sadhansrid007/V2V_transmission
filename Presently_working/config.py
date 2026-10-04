"""
Central configuration for the semantic topic segmentation pipeline.

Every tunable knob lives here so you don't have to hunt through
segment_embeddings.py / topic_segmenter.py / run_segmentation.py to
change behavior. See the "what to tune" section at the bottom of
run_segmentation.py's printed summary, and the accompanying
explanation, for guidance on which of these to touch first if the
output has too many/too few topics.
"""

import hashlib
import json
import os

# ------------------------------------------------------------------
# Multi-video keying helpers
# ------------------------------------------------------------------
# Every JSON artifact the pipeline writes (transcript.json, silence.json,
# topic_segmentation.json, boundary_scores.json, topic_analysis.json,
# topic_time_summary.json, pipeline_state.json) is a single file shared
# across every video you've ever run, structured as
#     { url_key(video_url): <that video's content>, ... }
# so re-running the pipeline on a new video_url never clobbers another
# video's results in the same file. ChromaDB collections and the
# boundary_scores.png plot (which can't live inside a JSON dict) are
# namespaced the same way, via collection_name_for_url() / url_scoped_path().

def url_key(video_url: str) -> str:
    """Short, stable, filesystem/collection-name-safe key derived from a
    video URL. Use this (not the raw URL) as the dict key / filename
    suffix everywhere a JSON file or ChromaDB collection is namespaced
    per-video, so odd characters in URLs never leak into paths or
    collection names."""
    return hashlib.sha256(video_url.strip().encode("utf-8")).hexdigest()[:16]


def url_scoped_path(base_path: str, video_url: str) -> str:
    """Derives a per-video filename for outputs that can't be stored as an
    entry inside a shared JSON dict (currently just the boundary_scores.png
    plot, and retreive_topics.py's embedding cache files)."""
    stem, ext = os.path.splitext(base_path)
    return f"{stem}__{url_key(video_url)}{ext}"


def load_url_keyed_json(path: str) -> dict:
    """Loads a shared, url-keyed JSON file. Returns {} if it doesn't exist
    yet or is unreadable, so callers can always treat the result as a dict."""
    if not os.path.exists(path):
        return {}
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except (json.JSONDecodeError, OSError):
        return {}
    return data if isinstance(data, dict) else {}


def load_url_keyed_entry(path: str, video_url: str):
    """Returns just this video's entry from a shared url-keyed JSON file,
    or None if that file or this video's entry doesn't exist yet."""
    return load_url_keyed_json(path).get(url_key(video_url))


def save_url_keyed_entry(path: str, video_url: str, value) -> None:
    """Loads the existing url-keyed JSON dict at `path` (or starts a new
    one), sets this video's entry, and writes the whole dict back -- so
    saving one video's results never overwrites any other video's entry
    living in the same file."""
    data = load_url_keyed_json(path)
    data[url_key(video_url)] = value
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


# ------------------------------------------------------------------
# Paths
# ------------------------------------------------------------------
TRANSCRIPT_PATH = "transcript.json"

CHROMA_PATH = "./chroma_db"
# Base name for the per-video ChromaDB collection. The actual collection
# used at runtime is always collection_name_for_url(video_url) below --
# never this constant directly -- so each video's segment embeddings live
# in their own collection and can never collide with another video's.
COLLECTION_NAME = "transcript_segment_embeddings"


def collection_name_for_url(video_url: str) -> str:
    """One ChromaDB collection per video, so embeddings from different
    lectures/videos are fully isolated from each other (no metadata
    filtering needed, no risk of one video's ids colliding with another's)."""
    return f"{COLLECTION_NAME}_{url_key(video_url)}"


OUTPUT_TOPIC_JSON = "topic_segmentation.json"
OUTPUT_BOUNDARY_JSON = "boundary_scores.json"
OUTPUT_PLOT_PNG = "boundary_scores.png"

# ------------------------------------------------------------------
# Embedding model
# ------------------------------------------------------------------
# Swap this to compare against another embedding model without
# touching any other file. Must be loadable via
# sentence_transformers.SentenceTransformer(EMBEDDING_MODEL).
EMBEDDING_MODEL = "BAAI/bge-m3"

# Batch size for the one-time embedding pass over all segments.
EMBEDDING_BATCH_SIZE = 32

# ------------------------------------------------------------------
# Sliding-window boundary detection
# ------------------------------------------------------------------
# Number of segments on EACH side of a candidate boundary used to
# build that side's contextual "window embedding" (see step 5/6 of
# the spec). A boundary between segment i-1 and i compares
# segments [i-WINDOW_SIZE, i-1] (LEFT) against [i, i+WINDOW_SIZE-1]
# (RIGHT) -- these two windows never overlap, which is the whole
# point: overlapping windows would trivially look similar regardless
# of any real topic change.
WINDOW_SIZE = 10

# How far the boundary pointer advances between consecutive
# evaluations. STRIDE = 1 means every possible boundary position is
# scored (the recommended default -- this is cheap since it reuses
# precomputed segment embeddings, not re-encoding text).
STRIDE = 1

# ------------------------------------------------------------------
# Smoothing
# ------------------------------------------------------------------
# Width (in boundary positions) of the moving-average filter applied
# to the raw boundary-score curve before peak detection. Larger =
# smoother curve = fewer, more confident boundaries. 1 disables
# smoothing entirely.
SMOOTHING_WINDOW = 3

# ------------------------------------------------------------------
# Candidate boundary detection (scipy.signal.find_peaks on the
# smoothed boundary-score curve)
# ------------------------------------------------------------------
# Minimum number of boundary-score positions between two accepted
# peaks. Since STRIDE = 1, this is directly in units of transcript
# segments. Prevents two adjacent, noisy local maxima from both being
# reported as separate boundaries.
MIN_BOUNDARY_DISTANCE = 5

# Minimum required topic length in segments, enforced AFTER peak
# detection as a second safety net (a peak that is far enough from
# its neighboring peak per MIN_BOUNDARY_DISTANCE can still be very
# close to the transcript's start/end, or MIN_BOUNDARY_DISTANCE could
# be set looser than this for experimentation -- this guarantees no
# topic ever ends up shorter than MIN_TOPIC_LENGTH regardless).
MIN_TOPIC_LENGTH = 5

# Minimum "prominence" (scipy's peak-prominence metric: how much a
# peak stands out from the surrounding valley floor, not just its
# absolute height) required for a local maximum to count as a
# candidate boundary. Set to None to auto-derive it as
# BOUNDARY_PROMINENCE_STD_MULTIPLIER * std(smoothed_scores) --
# adaptive to each lecture's own score distribution rather than a
# fixed number that only works for one recording's noise level.
BOUNDARY_PROMINENCE = None
BOUNDARY_PROMINENCE_STD_MULTIPLIER = 0.5

# ------------------------------------------------------------------
# Optional short/filler-segment down-weighting (step 16)
# ------------------------------------------------------------------
# Disabled by default -- the first implementation uses a plain mean
# over each window's segment embeddings. Flip this on to instead use
# a weighted mean that reduces the influence of very short segments
# ("Yes.", "Okay.", "Right.") on a window's semantic representation,
# on the theory that a one-word filler segment shouldn't be able to
# single-handedly shift a 10-segment window's centroid.
ENABLE_SHORT_SEGMENT_DOWNWEIGHTING = False

# A segment with fewer than this many words is considered "short".
SHORT_SEGMENT_WORD_THRESHOLD = 3

# Weight applied to short segments when downweighting is enabled.
# Normal segments always have weight 1.0.
SHORT_SEGMENT_WEIGHT = 0.3

# ------------------------------------------------------------------
# Retrieval-time LLM relevance judge (retreive_topics.py)
# ------------------------------------------------------------------
# retreive_topics.py first ranks topics by embedding similarity to the
# query, then hands the top TOP_K_EMBEDDING_CANDIDATES of those (with
# their ACTUAL transcript text, not just the generated description) to an
# LLM and asks it to act as the final human-like judge of which of those
# candidates are actually relevant/useful for the query -- it's free to
# pick one, several, or none of them, since embedding similarity alone
# often surfaces topics that are only superficially related.

# How many embedding-ranked candidates to pass to the LLM judge. Higher
# = better recall (less chance the right topic gets cut before the LLM
# ever sees it) at the cost of more tokens per query.
TOP_K_EMBEDDING_CANDIDATES = 8

# Reuse the same Groq model family as the analysis stage unless you want
# a different one specifically for retrieval judging.
RETRIEVAL_LLM_MODEL = "openai/gpt-oss-120b"
RETRIEVAL_LLM_TEMPERATURE = 0.1
RETRIEVAL_LLM_REASONING_EFFORT = "low"

# Safety cap on how much transcript text is sent per candidate topic to
# the judge, so an unusually long topic doesn't blow up the prompt when
# TOP_K_EMBEDDING_CANDIDATES candidates are all sent in a single call.
RETRIEVAL_MAX_TRANSCRIPT_CHARS_PER_TOPIC = 4000
