"""
pipeline.py

End-to-end orchestrator for the lecture pipeline:

    video_url
        |
    [1] injestion.ingestion()        -> transcript.json, silence.json
        |
    [2] run_segmentation.main()      -> topic_segmentation.json,
        |                               boundary_scores.json, boundary_scores.png
    [3] analyze_topics.main()        -> topic_analysis.json
        |
    [4] build_time_summary()         -> topic_time_summary.json
                                         { config.url_key(video_url):
                                             {video_url, important/filler/
                                              uncertain periods + total
                                              duration} }

Every JSON output above is a single file SHARED across every video_url
you've ever run, keyed by config.url_key(video_url) (see config.py), so
running the pipeline on a new video never overwrites another video's
results in the same file. ChromaDB gets one collection per video
(config.collection_name_for_url), and the boundary_scores.png plot gets
one file per video (config.url_scoped_path) since a PNG can't live inside
a shared JSON dict.

Each stage is skipped automatically if this video's outputs already exist
and its inputs/parameters haven't changed since the last run, so
re-running this script on the same video is cheap:

    - Stage 1 (ingestion) is skipped if this video already has an entry in
      both transcript.json and silence.json.
    - Stage 2 (segmentation) is skipped if this video already has an entry
      in topic_segmentation.json / boundary_scores.json AND its
      boundary_scores.png exists AND the transcript hasn't changed AND
      none of the segmentation parameters in config.py have changed since
      the last successful run for this video (tracked per-video in
      pipeline_state.json). Note: even when this stage DOES run, its own
      embedding step (segment_embeddings.get_or_compute_embeddings) has its
      own finer-grained cache in this video's ChromaDB collection and
      reuses embeddings whenever the segment text is unchanged.
    - Stage 3 (topic analysis) always runs, but analyze_topics.py's own
      RESUME logic already skips any topic already present in this video's
      entry in topic_analysis.json with matching boundaries -- so a re-run
      only spends Groq API calls on topics that are new or changed.
    - Stage 4 (time summary) is cheap (pure JSON aggregation) and always
      recomputed, so it stays in sync with whatever is in this video's
      topic_analysis.json entry.

Run:
    python pipeline.py "https://example.com/lecture.mp4"

or set VIDEO_URL below and just run:
    python pipeline.py

Requires everything injestion.py / run_segmentation.py / analyze_topics.py
individually require:
    pip install sentence-transformers chromadb scipy numpy matplotlib \
                groq pydantic python-dotenv torch pydub silero-vad
plus ffmpeg on your PATH, and GROQ_API_KEY set in your environment.
"""

from __future__ import annotations

import hashlib
import json
import os
import sys
from typing import Optional

import config
import injestion
import run_segmentation
import analyze_topics


# ---------------------------------------------------------------------------
# Pipeline-level configuration
# ---------------------------------------------------------------------------

# Leave blank to be prompted, or pass the URL as a command-line argument:
#   python pipeline.py "https://..."
VIDEO_URL = ""

PIPELINE_STATE_FILE = "pipeline_state.json"
TIME_SUMMARY_FILE = "topic_time_summary.json"

# Force a stage to re-run even if its cache looks valid. Useful after
# manually editing an intermediate file, or to sanity-check a cache hit.
FORCE_INGESTION = False
FORCE_SEGMENTATION = False


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------

def _hash_file(path: str) -> Optional[str]:
    if not os.path.exists(path):
        return None
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _load_json(path: str):
    if not os.path.exists(path):
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError):
        return None


def _load_pipeline_state() -> dict:
    """pipeline_state.json is itself url-keyed (config.url_key(video_url) ->
    {"segmentation": {...}}), so caching state for one video never clobbers
    another video's cache in the same file."""
    return _load_json(PIPELINE_STATE_FILE) or {}


def _save_pipeline_state(state: dict) -> None:
    with open(PIPELINE_STATE_FILE, "w", encoding="utf-8") as f:
        json.dump(state, f, indent=2, ensure_ascii=False)


def _segmentation_config_snapshot() -> dict:
    """Every config.py value that changes segmentation output, so a stage-2
    cache hit only counts if none of these have changed since last time."""
    return {
        "EMBEDDING_MODEL": config.EMBEDDING_MODEL,
        "EMBEDDING_BATCH_SIZE": config.EMBEDDING_BATCH_SIZE,
        "WINDOW_SIZE": config.WINDOW_SIZE,
        "STRIDE": config.STRIDE,
        "SMOOTHING_WINDOW": config.SMOOTHING_WINDOW,
        "MIN_BOUNDARY_DISTANCE": config.MIN_BOUNDARY_DISTANCE,
        "MIN_TOPIC_LENGTH": config.MIN_TOPIC_LENGTH,
        "BOUNDARY_PROMINENCE": config.BOUNDARY_PROMINENCE,
        "BOUNDARY_PROMINENCE_STD_MULTIPLIER": config.BOUNDARY_PROMINENCE_STD_MULTIPLIER,
        "ENABLE_SHORT_SEGMENT_DOWNWEIGHTING": config.ENABLE_SHORT_SEGMENT_DOWNWEIGHTING,
        "SHORT_SEGMENT_WORD_THRESHOLD": config.SHORT_SEGMENT_WORD_THRESHOLD,
        "SHORT_SEGMENT_WEIGHT": config.SHORT_SEGMENT_WEIGHT,
    }


def _banner(title: str) -> None:
    print("\n" + "=" * 70)
    print(title)
    print("=" * 70)


# ---------------------------------------------------------------------------
# Stage 1: ingestion
# ---------------------------------------------------------------------------

def run_ingestion_stage(video_url: str) -> None:
    _banner("STAGE 1/4: INGESTION (download audio, VAD silence removal, Whisper transcription)")
    print("Parameters:")
    print(f"  video_url               = {video_url}")
    print(f"  MIN_SILENCE_SECONDS     = {injestion.MIN_SILENCE_SECONDS}")
    print(f"  MERGE_GAP_SECONDS       = {injestion.MERGE_GAP_SECONDS}")
    print(f"  MAX_STRETCH_DRIFT       = {injestion.MAX_STRETCH_DRIFT}")
    print(f"  MAX_SENTENCE_WORDS      = {injestion.MAX_SENTENCE_WORDS}")
    print(f"  MAX_CHUNK_BYTES         = {injestion.MAX_CHUNK_BYTES}")

    transcript_path = config.TRANSCRIPT_PATH
    silence_path = "silence.json"

    # Both files are shared across every video, keyed by config.url_key
    # (see config.py), so this only checks THIS video's entry.
    existing_transcript_entry = config.load_url_keyed_entry(transcript_path, video_url)
    existing_silence_entry = config.load_url_keyed_entry(silence_path, video_url)

    if (
        not FORCE_INGESTION
        and existing_transcript_entry is not None
        and existing_silence_entry is not None
        and existing_silence_entry.get("video_url") == video_url
    ):
        print(f"\nSkipping ingestion: {transcript_path} and {silence_path} already have "
              f"an entry for this exact video_url (key {config.url_key(video_url)}).")
        print("Set FORCE_INGESTION = True to re-run anyway.")
        return

    print("\nRunning ingestion (this downloads audio, runs VAD, and calls Groq Whisper)...\n")
    transcript = injestion.ingestion(video_url=video_url)
    # Stamp the source URL onto the entry itself too, so a reader of
    # transcript.json doesn't have to reverse the hash to know which video
    # a given entry came from.
    transcript = {**transcript, "video_url": video_url}

    config.save_url_keyed_entry(transcript_path, video_url, transcript)

    silence_data = {
        "video_url": video_url,
        "max_duration": transcript["max_duration"],
        "min_silence_seconds": injestion.MIN_SILENCE_SECONDS,
        "silences": transcript["silences"],
    }
    config.save_url_keyed_entry(silence_path, video_url, silence_data)

    print(f"\nSaved transcript ({len(transcript['segments'])} sentences) to "
          f"{transcript_path} (key {config.url_key(video_url)})")
    print(f"Saved {len(transcript['silences'])} silence range(s) to "
          f"{silence_path} (key {config.url_key(video_url)})")


# ---------------------------------------------------------------------------
# Stage 2: segmentation
# ---------------------------------------------------------------------------

def run_segmentation_stage(video_url: str) -> None:
    _banner("STAGE 2/4: TOPIC SEGMENTATION (embeddings + sliding-window boundary detection)")
    print("Parameters:")
    for key, value in _segmentation_config_snapshot().items():
        print(f"  {key:<38} = {value}")

    url_k = config.url_key(video_url)
    state = _load_pipeline_state()
    # The transcript itself is now one entry inside a shared, url-keyed
    # transcript.json rather than the whole file, so hash just this
    # video's entry (not the entire multi-video file, which would falsely
    # invalidate the cache whenever ANY other video's transcript changes).
    transcript_entry = config.load_url_keyed_entry(config.TRANSCRIPT_PATH, video_url)
    transcript_hash = (
        hashlib.sha256(json.dumps(transcript_entry, sort_keys=True).encode("utf-8")).hexdigest()
        if transcript_entry is not None else None
    )
    current_snapshot = _segmentation_config_snapshot()

    plot_path = config.url_scoped_path(config.OUTPUT_PLOT_PNG, video_url)
    outputs_exist = (
        config.load_url_keyed_entry(config.OUTPUT_TOPIC_JSON, video_url) is not None
        and config.load_url_keyed_entry(config.OUTPUT_BOUNDARY_JSON, video_url) is not None
        and os.path.exists(plot_path)
    )
    prior = state.get(url_k, {}).get("segmentation", {})
    cache_matches = (
        outputs_exist
        and transcript_hash is not None
        and prior.get("transcript_hash") == transcript_hash
        and prior.get("config_snapshot") == current_snapshot
    )

    if not FORCE_SEGMENTATION and cache_matches:
        print(f"\nSkipping segmentation: {config.OUTPUT_TOPIC_JSON} already reflects the "
              f"current transcript and current config.py parameters for this video "
              f"(key {url_k}).")
        print("Set FORCE_SEGMENTATION = True to re-run anyway.")
        return

    print("\nRunning segmentation...\n")
    run_segmentation.main(video_url)

    state.setdefault(url_k, {})["segmentation"] = {
        "transcript_hash": transcript_hash,
        "config_snapshot": current_snapshot,
    }
    _save_pipeline_state(state)


# ---------------------------------------------------------------------------
# Stage 3: topic analysis
# ---------------------------------------------------------------------------

def run_analysis_stage(video_url: str) -> None:
    _banner("STAGE 3/4: LLM TOPIC ANALYSIS (Groq classification + description generation)")
    print("Parameters:")
    print(f"  GROQ_MODEL            = {analyze_topics.GROQ_MODEL}")
    print(f"  GROQ_TEMPERATURE      = {analyze_topics.GROQ_TEMPERATURE}")
    print(f"  REASONING_EFFORT      = {analyze_topics.REASONING_EFFORT}")
    print(f"  RESUME                = {analyze_topics.RESUME}  "
          f"(topics already in {analyze_topics.OUTPUT_FILE} for this video are skipped automatically)")
    print(f"  USE_NEIGHBOR_CONTEXT  = {analyze_topics.USE_NEIGHBOR_CONTEXT}")
    print()

    analyze_topics.main(video_url)


# ---------------------------------------------------------------------------
# Stage 4: url-keyed important/filler/uncertain time-period summary
# ---------------------------------------------------------------------------

def build_time_summary(video_url: str, analysis_path: str, video_duration: Optional[float]) -> dict:
    """
    Groups this video's topic_analysis.json entries (looked up by
    config.url_key(video_url) -- see config.py) into contiguous same-type
    blocks (adjacent topics of the same type are merged into one period)
    and returns:

        {
          "important": {"total_seconds": ..., "periods": [
              {"start": .., "end": .., "duration": .., "topic_ids": [..]}
          ]},
          "filler": {...},
          "uncertain": {...}
        }
    """
    topics = config.load_url_keyed_entry(analysis_path, video_url)
    if not topics:
        raise FileNotFoundError(
            f"No topic analysis found for this video_url in {analysis_path} "
            f"(key {config.url_key(video_url)}); run stage 3 first."
        )

    topics = sorted(topics, key=lambda t: t.get("start_time", 0.0))

    by_type = {"important": [], "filler": [], "uncertain": []}
    current_block = None

    for topic in topics:
        ttype = topic.get("type")
        if ttype not in by_type:
            continue

        if (
            current_block is not None
            and current_block["type"] == ttype
            and abs(current_block["end"] - topic["start_time"]) < 1e-6
        ):
            current_block["end"] = topic["end_time"]
            current_block["topic_ids"].append(topic["topic_id"])
        else:
            if current_block is not None:
                by_type[current_block["type"]].append(current_block)
            current_block = {
                "type": ttype,
                "start": topic["start_time"],
                "end": topic["end_time"],
                "topic_ids": [topic["topic_id"]],
            }

    if current_block is not None:
        by_type[current_block["type"]].append(current_block)

    summary = {}
    for ttype, blocks in by_type.items():
        periods = [
            {
                "start": round(b["start"], 2),
                "end": round(b["end"], 2),
                "duration": round(b["end"] - b["start"], 2),
                "topic_ids": b["topic_ids"],
            }
            for b in blocks
        ]
        summary[ttype] = {
            "total_seconds": round(sum(p["duration"] for p in periods), 2),
            "periods": periods,
        }

    summary["video_duration_seconds"] = round(video_duration, 2) if video_duration else None
    return summary


def run_time_summary_stage(video_url: str) -> None:
    _banner("STAGE 4/4: TIME-PERIOD SUMMARY (url-keyed important/filler/uncertain periods)")

    silence_entry = config.load_url_keyed_entry("silence.json", video_url) or {}
    video_duration = silence_entry.get("max_duration")

    new_entry = build_time_summary(video_url, analyze_topics.OUTPUT_FILE, video_duration)
    # Stamp the source URL onto the entry itself too, so a reader of
    # topic_time_summary.json doesn't have to reverse the hash key.
    new_entry = {"video_url": video_url, **new_entry}

    config.save_url_keyed_entry(TIME_SUMMARY_FILE, video_url, new_entry)

    print(f"Important: {new_entry['important']['total_seconds']:.1f}s across "
          f"{len(new_entry['important']['periods'])} period(s)")
    print(f"Filler:    {new_entry['filler']['total_seconds']:.1f}s across "
          f"{len(new_entry['filler']['periods'])} period(s)")
    print(f"Uncertain: {new_entry['uncertain']['total_seconds']:.1f}s across "
          f"{len(new_entry['uncertain']['periods'])} period(s)")
    print(f"\nSaved:\n    {TIME_SUMMARY_FILE}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    video_url = (sys.argv[1] if len(sys.argv) > 1 else "").strip() or VIDEO_URL.strip()
    if not video_url:
        video_url = input("Enter video URL: ").strip()
    if not video_url:
        print("ERROR: no video URL provided.")
        sys.exit(1)

    if not os.environ.get("GROQ_API_KEY"):
        print("ERROR: GROQ_API_KEY environment variable is not set "
              "(needed for both transcription and topic analysis).")
        sys.exit(1)

    run_ingestion_stage(video_url)
    run_segmentation_stage(video_url)
    run_analysis_stage(video_url)
    run_time_summary_stage(video_url)

    _banner("PIPELINE COMPLETE")
    url_k = config.url_key(video_url)
    print(f"  transcript          -> {config.TRANSCRIPT_PATH}         (key {url_k})")
    print(f"  silence info        -> silence.json               (key {url_k})")
    print(f"  topic segmentation  -> {config.OUTPUT_TOPIC_JSON}   (key {url_k})")
    print(f"  boundary scores     -> {config.OUTPUT_BOUNDARY_JSON} (key {url_k})")
    print(f"  boundary plot       -> {config.url_scoped_path(config.OUTPUT_PLOT_PNG, video_url)}")
    print(f"  topic analysis      -> {analyze_topics.OUTPUT_FILE}     (key {url_k})")
    print(f"  time-period summary -> {TIME_SUMMARY_FILE}       (key {url_k})")


if __name__ == "__main__":
    main()