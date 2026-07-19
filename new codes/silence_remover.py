"""
silence_remover.py
--------------------
Job: strip dead-air silence out of a lecture video, keeping only the
parts where someone is actually speaking.

Two interchangeable backends, so you can compare them directly:

  "pydub"  -- volume-threshold detection (uses MIN_SILENCE_LEN_MS and
              SILENCE_THRESH_DB from config.py). Zero new installs.

  "funasr" -- FunASR's FSMN-VAD model, a small trained model that
              recognizes actual speech patterns rather than just
              volume. Needs: pip install funasr

Either backend produces a list of "keep" segments (where speech is
happening), which we pad slightly (so words don't get clipped at the
edges) and merge (so we don't produce dozens of tiny jump-cuts for
every micro-pause). Those segments are then handed straight to
video_cutter.cut_and_stitch(), which does the actual cutting fast via
ffmpeg stream copy.

CHANGE vs the previous version: the FunASR model is now cached at
module level (_get_funasr_model), matching the pattern every other
client in this codebase already uses (_get_groq_client,
_get_embedding_model, _get_chroma_client). Previously it was
reconstructed from scratch -- reloading weights from disk -- on every
single call to remove_silence(..., method="funasr"), which is wasted
work if it's ever called more than once in the same process (e.g.
across Streamlit reruns).
"""

from pathlib import Path

from pydub import AudioSegment
from pydub.silence import detect_nonsilent

from config import MIN_SILENCE_LEN_MS, SILENCE_THRESH_DB
from video_cutter import cut_and_stitch

# Padding added to both sides of every kept speech segment, so a word
# doesn't get its first/last syllable clipped right at the cut.
PAD_SECONDS = 0.15

# Kept segments separated by a gap shorter than this get merged into
# one continuous segment, instead of producing a jump-cut for a
# barely-there pause (which sounds more jarring than just leaving it in).
MERGE_GAP_SECONDS = 0.3

_funasr_model = None


def _get_funasr_model():
    """Loads the FSMN-VAD model once per process and reuses it on every
    subsequent call, instead of reloading weights from disk every time."""
    global _funasr_model
    if _funasr_model is None:
        # Imported lazily so this file doesn't require `funasr` to be
        # installed unless you actually use this backend.
        from funasr import AutoModel
        _funasr_model = AutoModel(model="fsmn-vad", model_revision="v2.0.4", disable_update=True)
    return _funasr_model


def _merge_and_pad(raw_segments: list[dict], audio_duration_s: float) -> list[dict]:
    """Pads each segment, clamps to audio bounds, then merges any that
    now overlap or sit very close together."""
    if not raw_segments:
        return []

    padded = []
    for seg in raw_segments:
        padded.append({
            "start": max(0.0, seg["start"] - PAD_SECONDS),
            "end": min(audio_duration_s, seg["end"] + PAD_SECONDS),
        })

    padded.sort(key=lambda s: s["start"])
    merged = [padded[0]]
    for current in padded[1:]:
        last = merged[-1]
        if current["start"] <= last["end"] + MERGE_GAP_SECONDS:
            last["end"] = max(last["end"], current["end"])
        else:
            merged.append(current)

    return merged


def _speech_segments_pydub(audio_path: Path) -> tuple[list[dict], float]:
    audio = AudioSegment.from_wav(str(audio_path))
    duration_s = len(audio) / 1000.0

    nonsilent_ranges = detect_nonsilent(
        audio,
        min_silence_len=MIN_SILENCE_LEN_MS,
        silence_thresh=SILENCE_THRESH_DB,
    )

    segments = [
        {"start": start_ms / 1000.0, "end": end_ms / 1000.0}
        for start_ms, end_ms in nonsilent_ranges
    ]
    return segments, duration_s


def _speech_segments_funasr(audio_path: Path) -> tuple[list[dict], float]:
    audio = AudioSegment.from_wav(str(audio_path))
    duration_s = len(audio) / 1000.0

    model = _get_funasr_model()
    result = model.generate(input=str(audio_path))

    # result[0]["value"] is a list of [start_ms, end_ms] speech segments.
    raw_segments = result[0]["value"]
    segments = [
        {"start": start_ms / 1000.0, "end": end_ms / 1000.0}
        for start_ms, end_ms in raw_segments
    ]
    return segments, duration_s


def remove_silence(video_path: Path, audio_path: Path, output_path: Path, method: str = "pydub") -> Path | None:
    """
    Produces a version of video_path with silence stripped out.
    method: "pydub" or "funasr"
    """
    video_path = Path(video_path)
    audio_path = Path(audio_path)
    output_path = Path(output_path)

    if method == "pydub":
        raw_segments, duration_s = _speech_segments_pydub(audio_path)
    elif method == "funasr":
        raw_segments, duration_s = _speech_segments_funasr(audio_path)
    else:
        raise ValueError(f"Unknown method: {method!r}. Use 'pydub' or 'funasr'.")

    if not raw_segments:
        print(f"No speech segments detected using {method}. Nothing to keep.")
        return None

    keep_segments = _merge_and_pad(raw_segments, duration_s)
    original_speech_time = sum(s["end"] - s["start"] for s in raw_segments)
    kept_time = sum(s["end"] - s["start"] for s in keep_segments)
    print(
        f"[{method}] Kept {len(keep_segments)} segments, "
        f"{kept_time:.1f}s out of {duration_s:.1f}s original "
        f"({100 * kept_time / duration_s:.0f}% retained)"
    )

    return cut_and_stitch(video_path, keep_segments, output_path)


if __name__ == "__main__":
    import sys
    from config import VIDEO_DIR, AUDIO_DIR, DATA_DIR

    video_id = sys.argv[1]
    method = sys.argv[2] if len(sys.argv) > 2 else "pydub"

    video_path = VIDEO_DIR / f"{video_id}.mp4"
    audio_path = AUDIO_DIR / f"{video_id}.wav"
    output_path = DATA_DIR / f"silence_removed_{method}_{video_id}.mp4"

    result = remove_silence(video_path, audio_path, output_path, method=method)
    if result:
        print(f"Done: {result}")