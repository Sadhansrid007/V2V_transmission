"""
clip_and_merge.py

Cuts clips out of a video based on the topic boundaries in topic_analysis.json
(the output of analyze_topics.py) and stitches the selected clips back
together into a single trimmed video.

By default this keeps only topics classified as "important" and drops
"filler" / "uncertain" topics - i.e. it produces a version of the lecture
video with the low-value parts removed.

Requires ffmpeg and ffprobe to be installed and on your PATH:
    - Windows: https://www.gyan.dev/ffmpeg/builds/ (add the bin/ folder to PATH)
    - macOS:   brew install ffmpeg
    - Linux:   sudo apt install ffmpeg   (or your distro's package manager)

This script does not require any Python video libraries - it drives ffmpeg
directly via subprocess.

Run:
    python clip_and_merge.py
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import tempfile
from typing import List


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

VIDEO_FILE = "video_download.mp4"
ANALYSIS_FILE = "topic_analysis.json"
OUTPUT_FILE = "video_important_clips.mp4"

# Which topic types to keep, in the sense of "include in the final video".
INCLUDE_TYPES = ["important"]

# Optional padding (in seconds) added before/after each clip's transcript
# boundary, so cuts don't clip off the first/last word. Set to 0 to disable.
# Padding is automatically clamped so it never runs past the video's start,
# end, or into a neighboring excluded clip.
PAD_SECONDS = 0.3

# Re-encoding settings. Re-encoding (rather than stream-copying) gives
# frame-accurate cuts at arbitrary timestamps, which matters here since our
# cut points come from transcript timing, not video keyframes.
VIDEO_CODEC = "libx264"
VIDEO_PRESET = "fast"
VIDEO_CRF = "18"          # lower = higher quality/bigger file; 18-23 is typical
AUDIO_CODEC = "aac"
AUDIO_BITRATE = "192k"

# If True, keep the individual per-topic clip files (and the concat list)
# in a folder next to the output instead of deleting them after merging.
KEEP_TEMP_CLIPS = False
TEMP_CLIPS_DIR = "temp_clips"


# ---------------------------------------------------------------------------
# ffmpeg / ffprobe helpers
# ---------------------------------------------------------------------------

def check_tools_available() -> None:
    missing = [tool for tool in ("ffmpeg", "ffprobe") if shutil.which(tool) is None]
    if missing:
        print(f"ERROR: required tool(s) not found on PATH: {', '.join(missing)}")
        print("Install ffmpeg (which includes ffprobe) and make sure it's on your PATH.")
        sys.exit(1)


def get_video_duration(video_path: str) -> float:
    result = subprocess.run(
        [
            "ffprobe", "-v", "error",
            "-show_entries", "format=duration",
            "-of", "default=noprint_wrappers=1:nokey=1",
            video_path,
        ],
        capture_output=True, text=True,
    )
    if result.returncode != 0 or not result.stdout.strip():
        raise RuntimeError(f"ffprobe failed to read duration for {video_path}: {result.stderr.strip()}")
    return float(result.stdout.strip())


def clip_segment(video_path: str, start: float, end: float, out_path: str) -> None:
    """Extract [start, end) from video_path into out_path, re-encoding for
    frame-accurate cuts. -ss before -i gives fast input seeking; combined
    with re-encoding, ffmpeg still lands on the exact requested frame."""
    duration = max(0.0, end - start)
    cmd = [
        "ffmpeg", "-y",
        "-ss", f"{start:.3f}",
        "-i", video_path,
        "-t", f"{duration:.3f}",
        "-c:v", VIDEO_CODEC,
        "-preset", VIDEO_PRESET,
        "-crf", VIDEO_CRF,
        "-c:a", AUDIO_CODEC,
        "-b:a", AUDIO_BITRATE,
        "-movflags", "+faststart",
        out_path,
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"ffmpeg failed to clip [{start:.2f}, {end:.2f}]: {result.stderr[-800:]}")


def merge_clips(clip_paths: List[str], list_file_path: str, output_path: str) -> None:
    with open(list_file_path, "w", encoding="utf-8") as f:
        for path in clip_paths:
            # ffmpeg concat demuxer format; escape single quotes in paths defensively.
            safe_path = os.path.abspath(path).replace("'", "'\\''")
            f.write(f"file '{safe_path}'\n")

    cmd = [
        "ffmpeg", "-y",
        "-f", "concat", "-safe", "0",
        "-i", list_file_path,
        "-c", "copy",
        output_path,
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"ffmpeg failed to merge clips: {result.stderr[-800:]}")


# ---------------------------------------------------------------------------
# Topic selection
# ---------------------------------------------------------------------------

def load_selected_topics(analysis_path: str, video_duration: float) -> List[dict]:
    with open(analysis_path, "r", encoding="utf-8") as f:
        topics = json.load(f)

    if not isinstance(topics, list) or not topics:
        raise ValueError("Expected a non-empty JSON array in topic_analysis.json")

    # Keep only requested types, preserve original topic order (already
    # chronological, but sort defensively by start_time just in case).
    selected = [t for t in topics if t.get("type") in INCLUDE_TYPES]
    selected.sort(key=lambda t: t.get("start_time", 0.0))

    # Apply padding, clamped to video bounds and to avoid overlapping a
    # neighboring (also selected) clip.
    for i, topic in enumerate(selected):
        start = max(0.0, topic["start_time"] - PAD_SECONDS)
        end = min(video_duration, topic["end_time"] + PAD_SECONDS)

        if i > 0:
            prev_end = selected[i - 1]["_padded_end"]
            start = max(start, prev_end)
        topic["_padded_start"] = start
        topic["_padded_end"] = end

    return selected


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    check_tools_available()

    if not os.path.exists(VIDEO_FILE):
        print(f"ERROR: {VIDEO_FILE} not found in the current directory.")
        sys.exit(1)
    if not os.path.exists(ANALYSIS_FILE):
        print(f"ERROR: {ANALYSIS_FILE} not found in the current directory.")
        sys.exit(1)

    print(f"Reading {ANALYSIS_FILE}...")
    print(f"Checking {VIDEO_FILE}...")
    video_duration = get_video_duration(VIDEO_FILE)
    print(f"Video duration: {video_duration:.1f}s")

    selected = load_selected_topics(ANALYSIS_FILE, video_duration)
    if not selected:
        print(f"No topics matched INCLUDE_TYPES={INCLUDE_TYPES}. Nothing to do.")
        sys.exit(0)

    total_kept = sum(t["_padded_end"] - t["_padded_start"] for t in selected)
    print(f"Selected {len(selected)} topic(s) with type in {INCLUDE_TYPES}")
    print(f"Total kept duration: {total_kept:.1f}s (of {video_duration:.1f}s)\n")

    clips_dir = tempfile.mkdtemp(prefix="clip_and_merge_") if not KEEP_TEMP_CLIPS else TEMP_CLIPS_DIR
    os.makedirs(clips_dir, exist_ok=True)

    clip_paths = []
    try:
        for i, topic in enumerate(selected, start=1):
            start, end = topic["_padded_start"], topic["_padded_end"]
            out_path = os.path.join(clips_dir, f"clip_{i:03d}_topic{topic['topic_id']}.mp4")
            name = topic.get("name", f"Topic {topic['topic_id']}")
            print(f"[{i}/{len(selected)}] Clipping topic {topic['topic_id']} "
                  f"({start:.1f}s - {end:.1f}s): {name}")
            clip_segment(VIDEO_FILE, start, end, out_path)
            clip_paths.append(out_path)

        print("\nMerging clips...")
        list_file_path = os.path.join(clips_dir, "concat_list.txt")
        merge_clips(clip_paths, list_file_path, OUTPUT_FILE)

        print(f"\nFinished.\nSaved:\n    {OUTPUT_FILE}")

    finally:
        if not KEEP_TEMP_CLIPS:
            shutil.rmtree(clips_dir, ignore_errors=True)
        else:
            print(f"\nKept individual clips and concat list in: {clips_dir}")


if __name__ == "__main__":
    main()