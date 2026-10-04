"""
silence_remover.py
--------------------
Job: strip dead-air silence out of a lecture video, keeping only the
parts where someone is actually speaking.

Four interchangeable backends:

  "whisper_gaps" (DEFAULT) -- derives speech spans from the word-level
              timestamps transcribe.py already produces. A "silence" is
              just a gap between consecutive recognized words wider than
              SILENCE_WORD_GAP_SECONDS. Free -- no new model, no extra
              audio pass, reuses a call we're making anyway. Only as
              good as the transcript, though: if Whisper mis-transcribed
              a stretch, this method has nothing to go on there.

  "silero"  -- Silero VAD, a small neural model that classifies the raw
              waveform directly (speech vs non-speech), independent of
              transcription accuracy. Used as the fallback for videos
              where whisper_gaps looks unreliable. Needs:
              pip install silero-vad

  "pydub"   -- volume-threshold detection (uses MIN_SILENCE_LEN_MS and
              SILENCE_THRESH_DB from config.py). Kept as the simplest
              possible fallback -- zero new installs.

  "funasr"  -- FunASR's FSMN-VAD model. Kept as a known option; no
              longer the default since Silero is lighter to integrate
              with comparable accuracy. Needs: pip install funasr

Either backend produces a list of "keep" segments (where speech is
happening), which we pad slightly (so words don't get clipped at the
edges) and merge (so we don't produce dozens of tiny jump-cuts for
every micro-pause). Those segments are then handed straight to
video_cutter.cut_and_stitch().

CHANGE vs the previous version: added whisper_gaps and silero as new
methods (see module docstring above), and remove_silence() now takes
an optional transcript_path parameter, required only when
method="whisper_gaps". Default method changed from "pydub" to
"whisper_gaps".
"""

import json
from pathlib import Path

from pydub import AudioSegment
from pydub.silence import detect_nonsilent

from config import MIN_SILENCE_LEN_MS, SILENCE_THRESH_DB, SILENCE_WORD_GAP_SECONDS
from video_cutter import cut_and_stitch

PAD_SECONDS = 0.15
MERGE_GAP_SECONDS = 0.3

_funasr_model = None
_silero_model = None
_silero_utils = None


def _get_funasr_model():
    global _funasr_model
    if _funasr_model is None:
        from funasr import AutoModel
        _funasr_model = AutoModel(model="fsmn-vad", model_revision="v2.0.4", disable_update=True)
    return _funasr_model


def _get_silero_model():
    """Loads Silero VAD once per process and reuses it on every
    subsequent call, same caching pattern as _get_funasr_model."""
    global _silero_model, _silero_utils
    if _silero_model is None:
        from silero_vad import load_silero_vad, get_speech_timestamps, read_audio
        _silero_model = load_silero_vad()
        _silero_utils = {"get_speech_timestamps": get_speech_timestamps, "read_audio": read_audio}
    return _silero_model, _silero_utils


def _merge_and_pad(raw_segments: list[dict], audio_duration_s: float) -> list[dict]:
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

    raw_segments = result[0]["value"]
    segments = [
        {"start": start_ms / 1000.0, "end": end_ms / 1000.0}
        for start_ms, end_ms in raw_segments
    ]
    return segments, duration_s


def _speech_segments_silero(audio_path: Path) -> tuple[list[dict], float]:
    """Runs Silero VAD directly on the raw waveform -- unlike
    whisper_gaps, this doesn't depend on transcription accuracy at all."""
    audio = AudioSegment.from_wav(str(audio_path))
    duration_s = len(audio) / 1000.0

    model, utils = _get_silero_model()
    wav = utils["read_audio"](str(audio_path), sampling_rate=16000)
    raw_timestamps = utils["get_speech_timestamps"](
        wav, model, sampling_rate=16000, return_seconds=True
    )

    segments = [{"start": t["start"], "end": t["end"]} for t in raw_timestamps]
    return segments, duration_s


def _speech_segments_whisper_gaps(transcript_path: Path, audio_path: Path) -> tuple[list[dict], float]:
    """Derives speech spans from word-level timestamps instead of
    running any acoustic model at all."""
    if not transcript_path.exists():
        raise FileNotFoundError(
            f"No transcript found at {transcript_path}. whisper_gaps needs "
            "transcribe() to have run first."
        )

    audio = AudioSegment.from_wav(str(audio_path))
    duration_s = len(audio) / 1000.0

    transcript = json.loads(transcript_path.read_text())
    all_words = [w for seg in transcript["segments"] for w in seg["words"]]
    all_words.sort(key=lambda w: w["start"])

    if not all_words:
        return [], duration_s

    segments = []
    current_start = all_words[0]["start"]
    current_end = all_words[0]["end"]

    for word in all_words[1:]:
        gap = word["start"] - current_end
        if gap >= SILENCE_WORD_GAP_SECONDS:
            segments.append({"start": current_start, "end": current_end})
            current_start = word["start"]
        current_end = word["end"]

    segments.append({"start": current_start, "end": current_end})
    return segments, duration_s


def remove_silence(
    video_path: Path,
    audio_path: Path,
    output_path: Path,
    method: str = "whisper_gaps",
    transcript_path: Path | None = None,
) -> Path | None:
    """
    Produces a version of video_path with silence stripped out.
    method: "whisper_gaps" (default), "silero", "pydub", or "funasr"
    transcript_path is required only when method="whisper_gaps".
    """
    video_path = Path(video_path)
    audio_path = Path(audio_path)
    output_path = Path(output_path)

    if method == "pydub":
        raw_segments, duration_s = _speech_segments_pydub(audio_path)
    elif method == "funasr":
        raw_segments, duration_s = _speech_segments_funasr(audio_path)
    elif method == "silero":
        raw_segments, duration_s = _speech_segments_silero(audio_path)
    elif method == "whisper_gaps":
        if transcript_path is None:
            raise ValueError("method='whisper_gaps' requires transcript_path.")
        raw_segments, duration_s = _speech_segments_whisper_gaps(Path(transcript_path), audio_path)
    else:
        raise ValueError(
            f"Unknown method: {method!r}. Use 'whisper_gaps', 'silero', 'pydub', or 'funasr'."
        )

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
    from config import VIDEO_DIR, AUDIO_DIR, DATA_DIR, TRANSCRIPT_DIR

    video_id = sys.argv[1]
    method = sys.argv[2] if len(sys.argv) > 2 else "whisper_gaps"

    video_path = VIDEO_DIR / f"{video_id}.mp4"
    audio_path = AUDIO_DIR / f"{video_id}.wav"
    transcript_path = TRANSCRIPT_DIR / f"{video_id}.json"
    output_path = DATA_DIR / f"silence_removed_{method}_{video_id}.mp4"

    result = remove_silence(video_path, audio_path, output_path, method=method, transcript_path=transcript_path)
    if result:
        print(f"Done: {result}")