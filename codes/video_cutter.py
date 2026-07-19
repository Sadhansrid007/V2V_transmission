"""
video_cutter.py
-----------------
Job: given a video file and a list of {"start", "end"} timestamp
ranges, cut those ranges out and stitch them into one summary clip.

CHANGE vs the previous version -- this fixes a real accuracy bug, not
just a style change:

The old version used "-ss" BEFORE "-i" combined with "-c:v copy"
(stream copy) for speed. With stream copy, ffmpeg can only actually
start a segment at a keyframe, so every cut silently snapped to the
nearest keyframe AT OR BEFORE the requested start time -- it can never
snap forward, only backward. For a single one-off clip (like the "🎬
Generate clip" button in the Ask tab) that's usually a harmless 1-2s
of extra footage at the front.

But summarizer.py and silence_remover.py can produce 50-100+ cuts from
one video. If the source video's keyframe interval is a few seconds
(common for lecture-hall/projector recordings that aren't encoded for
fast-seeking), that backward-snap error compounds across every single
cut -- which is exactly what caused the reported bug: a summary
reported as 12.1 minutes of kept segments actually rendering as a
17:05 video, with un-trimmed silence audibly still present at several
cut points.

Fix: two-stage seeking with the video re-encoded (not stream-copied).
  1. A coarse "-ss" BEFORE "-i" still does a fast keyframe-level seek
     to a few seconds before the real start -- this is what keeps
     re-encoding fast, since ffmpeg doesn't have to decode from the
     very beginning of the file for every single cut.
  2. A second, fine-grained "-ss" AFTER "-i" then seeks frame-accurately
     from that point to the exact requested start. This only works
     because we're re-encoding (decoding is required either way for a
     frame-accurate cut) -- stream copy fundamentally cannot do this,
     which is why the old approach couldn't be "fixed" while keeping
     -c:v copy.

Trade-off: this is slower than pure stream copy, since every segment's
video is now actually re-encoded (libx264, "veryfast" preset to keep
this reasonable) instead of copied. For a single clip this difference
is unnoticeable. For 50+ short summarizer/silence-remover cuts it adds
real time, but correctness -- the output actually matching the
timestamps and stats you're shown -- matters more than shaving seconds
off an already-background job.
"""

import shutil
import subprocess
import tempfile
from pathlib import Path

# Ranges whose gap is smaller than this get merged into one cut,
# instead of producing a visible jump-cut for a half-second gap.
MERGE_GAP_SECONDS = 1.5

# How far before the real start point to do the fast, keyframe-level
# coarse seek. Large enough to land before essentially any keyframe
# interval seen in practice, small enough to keep decoding short.
COARSE_SEEK_BUFFER_SECONDS = 5.0

# Re-encode settings for cut segments. "veryfast" trades some
# compression efficiency for speed, which is the right trade for a
# clip that's about to be re-concatenated anyway.
VIDEO_CODEC = "libx264"
VIDEO_PRESET = "veryfast"
VIDEO_CRF = "23"


def _merge_ranges(timestamps: list[dict]) -> list[dict]:
    if not timestamps:
        return []

    ranges = sorted(timestamps, key=lambda t: t["start"])
    merged = [dict(ranges[0])]

    for current in ranges[1:]:
        last = merged[-1]
        if current["start"] <= last["end"] + MERGE_GAP_SECONDS:
            last["end"] = max(last["end"], current["end"])
        else:
            merged.append(dict(current))

    return merged


def cut_and_stitch(video_path: Path, timestamps: list[dict], output_path: Path) -> Path | None:
    """
    Cuts and concatenates the given timestamp ranges out of video_path,
    seeking frame-accurately so the output actually matches the
    requested ranges (see module docstring for why this re-encodes
    instead of stream-copying).
    Returns the output path on success, None if there was nothing valid to cut.
    """
    if not timestamps:
        print("No timestamps provided. Cannot cut video.")
        return None

    video_path = Path(video_path)
    output_path = Path(output_path)
    merged_ranges = _merge_ranges(timestamps)

    tmp_dir = Path(tempfile.mkdtemp(prefix="lecturelens_clip_"))
    segment_paths = []

    try:
        for i, ts in enumerate(merged_ranges):
            start = max(0, ts["start"])
            end = ts["end"]
            duration = end - start

            if duration <= 0:
                continue

            seg_path = tmp_dir / f"seg_{i:03d}.mp4"

            # Coarse seek (before -i): fast, keyframe-level, lands a
            # few seconds before the real start so decoding from there
            # to the exact start point stays cheap.
            coarse_seek = max(0.0, start - COARSE_SEEK_BUFFER_SECONDS)
            # Fine seek (after -i): frame-accurate, covers the small
            # remaining gap between the coarse seek and the real start.
            fine_offset = start - coarse_seek

            print(f"Cutting clip {i + 1}: {start:.1f}s to {end:.1f}s")

            subprocess.run(
                [
                    "ffmpeg", "-y",
                    "-ss", str(coarse_seek),
                    "-i", str(video_path),
                    "-ss", str(fine_offset),
                    "-t", str(duration),
                    "-c:v", VIDEO_CODEC, "-preset", VIDEO_PRESET, "-crf", VIDEO_CRF,
                    "-c:a", "aac", "-b:a", "128k",
                    "-avoid_negative_ts", "make_zero",
                    str(seg_path),
                ],
                check=True,
                capture_output=True,
            )
            segment_paths.append(seg_path)

        if not segment_paths:
            print("No valid clips found after clamping to video bounds.")
            return None

        # Stitch all segments together. Every segment was just encoded
        # with identical codec/preset settings above, so the concat
        # demuxer can safely stream-copy at this stage -- no need to
        # re-encode a second time just to join files that already match.
        concat_list_path = tmp_dir / "concat_list.txt"
        concat_list_path.write_text(
            "\n".join(f"file '{p.resolve().as_posix()}'" for p in segment_paths)
        )

        print("Stitching clips together...")
        subprocess.run(
            [
                "ffmpeg", "-y",
                "-f", "concat", "-safe", "0",
                "-i", str(concat_list_path),
                "-c", "copy",
                str(output_path),
            ],
            check=True,
            capture_output=True,
        )

    except subprocess.CalledProcessError as e:
        print(f"ffmpeg failed: {e.stderr.decode(errors='ignore')}")
        return None
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)

    print(f"Done. Summary video saved at: {output_path}")
    return output_path