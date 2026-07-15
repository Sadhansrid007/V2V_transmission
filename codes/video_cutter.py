"""
video_cutter.py
-----------------
Job: given a video file and a list of {"start", "end"} timestamp
ranges, cut those ranges out and stitch them into one summary clip.

Fix vs the earlier prototypes: timestamps coming back from the LLM can
overlap or sit right next to each other (e.g. [10,20] and [18,30]) --
cutting each separately produces a stutter/repeat in the final video.
We merge overlapping or near-adjacent ranges into single continuous
cuts first.
"""

from pathlib import Path

from moviepy import VideoFileClip, concatenate_videoclips

# Ranges whose gap is smaller than this get merged into one cut,
# instead of producing a visible jump-cut for a half-second gap.
MERGE_GAP_SECONDS = 1.5


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
    Cuts and concatenates the given timestamp ranges out of video_path.
    Returns the output path on success, None if there was nothing valid to cut.
    """
    if not timestamps:
        print("No timestamps provided. Cannot cut video.")
        return None

    video_path = Path(video_path)
    output_path = Path(output_path)
    merged_ranges = _merge_ranges(timestamps)

    video = VideoFileClip(str(video_path))
    clips = []

    try:
        for i, ts in enumerate(merged_ranges):
            start = max(0, ts["start"])
            end = min(ts["end"], video.duration)

            if start >= end:
                continue

            print(f"Cutting clip {i + 1}: {start:.1f}s to {end:.1f}s")
            clips.append(video.subclipped(start, end))

        if not clips:
            print("No valid clips found after clamping to video bounds.")
            return None

        print("Stitching clips together...")
        final_video = concatenate_videoclips(clips)

        print(f"Saving output to: {output_path}")
        final_video.write_videofile(
            str(output_path),
            codec="libx264",
            audio_codec="aac",
            logger=None,
        )
        final_video.close()

    finally:
        video.close()
        for clip in clips:
            clip.close()

    print(f"Done. Summary video saved at: {output_path}")
    return output_path