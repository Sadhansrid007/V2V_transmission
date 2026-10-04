import json
import math
import os
import subprocess
import wave
from pathlib import Path

import numpy as np
import torch
from dotenv import load_dotenv
from groq import Groq
from pydub import AudioSegment
from silero_vad import load_silero_vad, get_speech_timestamps

load_dotenv()

client = Groq(api_key=os.getenv("GROQ_API_KEY"))

# ------------------------------------
# Config
# ------------------------------------

# Groq's free tier caps uploads at 25MB (100MB on the dev/paid tier).
# We stay comfortably under that so re-encoding overhead doesn't push
# a chunk back over the limit.
MAX_CHUNK_BYTES = 20 * 1024 * 1024  # 20MB safety margin

# If no sentence-ending punctuation shows up within this many words,
# force a break anyway. Guards against run-on speech / missed punctuation
# producing one giant "sentence".
MAX_SENTENCE_WORDS = 50

# Common abbreviations that end in "." but do NOT mark a sentence end.
_ABBREVIATIONS = {
    "mr", "mrs", "ms", "dr", "prof", "sr", "jr", "vs", "etc",
    "e.g", "i.e", "u.s", "u.k", "st", "no", "vol", "fig", "approx",
    "gen", "capt", "col", "sgt", "lt", "rev", "inc", "ltd", "co",
}

# ------------------------------------
# Audio stretch config
# ------------------------------------

# atempo_factor = raw_audio_dur / video_dur. A factor of 1.0 means no
# drift at all. Anything beyond +/-3% drift usually means something is
# actually wrong with the source stream (dropped frames, VFR video,
# corrupt container) rather than ordinary clock drift, so we stop
# instead of silently stretching audio by a large, suspicious amount.
# This is only the DEFAULT ceiling -- pass max_stretch_drift= to
# ingestion() to override it for sources with known, legitimate
# clock drift (e.g. long lecture-capture recordings where video and
# audio are timestamped by separate hardware clocks).
MAX_STRETCH_DRIFT = 0.05  # 5% -- this capture system's videos routinely
                           # drift ~3% due to separate audio/video clocks

# ------------------------------------
# Silero VAD silence detection config
# ------------------------------------

# Only keep silence gaps at least this long.
MIN_SILENCE_SECONDS = 10

# Used twice in _detect_silence_silero, for two different merge passes
# with the same "close enough to be one continuous thing" intent:
#   1. Before computing gaps: merge raw VAD speech segments separated by
#      less than this many seconds, so a short in-speech pause doesn't
#      get mistaken for a stand-alone silence.
#   2. After computing the final silence list: merge two silence regions
#      that ended up separated by an isolated stretch of "speech" shorter
#      than this many seconds (e.g. a single stray word or cough sitting
#      between two long silences). Pass (1) only merges speech segments
#      that are themselves close together -- it does NOT catch this case,
#      since an isolated blip far from any other speech has nothing
#      nearby to merge with during pass (1), yet still splits what should
#      be reported as one continuous silence into two.
MERGE_GAP_SECONDS = 2

# Silero VAD expects 16kHz (or 8kHz) mono audio, which is exactly what
# _extract_and_stretch_audio() already produces.
_VAD_SAMPLE_RATE = 16000

_vad_model = None  # loaded lazily, once


def _get_vad_model():
    global _vad_model
    if _vad_model is None:
        _vad_model = load_silero_vad()
    return _vad_model


def _load_wav_for_vad(path: str, expected_sample_rate: int = _VAD_SAMPLE_RATE) -> torch.Tensor:
    """
    Reads a mono PCM WAV file directly (stdlib `wave` + numpy) and
    returns it as a 1-D float32 torch tensor in [-1.0, 1.0], the exact
    format Silero's get_speech_timestamps() expects.

    We deliberately bypass silero_vad's own read_audio() /
    torchaudio.load() here: newer torchaudio releases (2.9+) require
    the separate `torchcodec` package for any audio decoding, which
    may not be installed. Since _extract_and_stretch_audio() already
    guarantees our file is mono PCM16 at _VAD_SAMPLE_RATE, we can read
    it ourselves with zero extra dependencies and skip that whole
    decoding path.
    """
    with wave.open(path, "rb") as wf:
        n_channels = wf.getnchannels()
        sampwidth = wf.getsampwidth()
        frame_rate = wf.getframerate()
        n_frames = wf.getnframes()
        raw = wf.readframes(n_frames)

    if sampwidth != 2:
        raise RuntimeError(
            f"Expected 16-bit PCM audio for VAD, got {sampwidth * 8}-bit ({path})."
        )
    if frame_rate != expected_sample_rate:
        raise RuntimeError(
            f"Expected {expected_sample_rate}Hz audio for VAD, got {frame_rate}Hz ({path})."
        )

    samples = np.frombuffer(raw, dtype=np.int16)
    if n_channels > 1:
        samples = samples.reshape(-1, n_channels).mean(axis=1)

    samples = samples.astype(np.float32) / 32768.0
    return torch.from_numpy(samples)


# ------------------------------------
# Small helpers (duration probing, formatting)
# ------------------------------------
def _get_duration(path_or_url: str, stream: str | None = None) -> float:
    """
    Returns duration in seconds. If `stream` is given (e.g. "v:0" or
    "a:0"), reads that stream's own duration field for precision;
    falls back to container-level format duration if the stream
    doesn't expose one.
    """
    if stream:
        cmd = [
            "ffprobe", "-v", "error",
            "-select_streams", stream,
            "-show_entries", "stream=duration",
            "-of", "default=noprint_wrappers=1:nokey=1",
            path_or_url,
        ]
        r = subprocess.run(cmd, capture_output=True, text=True)
        value = r.stdout.strip()
        if r.returncode == 0 and value and value != "N/A":
            return float(value)

    cmd = [
        "ffprobe", "-v", "error",
        "-show_entries", "format=duration",
        "-of", "default=noprint_wrappers=1:nokey=1",
        path_or_url,
    ]
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError(f"ffprobe failed:\n{r.stderr}")
    return float(r.stdout.strip())


def _seconds_to_hms(seconds: float) -> str:
    total = int(round(seconds))
    h, rem = divmod(total, 3600)
    m, s = divmod(rem, 60)
    return f"{h}:{m:02d}:{s:02d}" if h else f"{m}:{s:02d}"


def _build_atempo_chain(factor: float) -> str:
    """
    ffmpeg's atempo filter only accepts 0.5-2.0 in a single pass; chain
    multiple atempo filters if the factor ever falls outside that.
    """
    if 0.5 <= factor <= 2.0:
        return f"atempo={factor}"
    filters = []
    remaining = factor
    while remaining > 2.0:
        filters.append("atempo=2.0")
        remaining /= 2.0
    while remaining < 0.5:
        filters.append("atempo=0.5")
        remaining /= 0.5
    filters.append(f"atempo={remaining}")
    return ",".join(filters)


# ------------------------------------
# Audio extraction + time-stretch to match video duration
# ------------------------------------
def _extract_and_stretch_audio(
    video_url: str,
    raw_audio_path: str = "audio_raw.wav",
    stretched_audio_path: str = "audio.wav",
    keep_raw_audio: bool = False,
    max_stretch_drift: float = MAX_STRETCH_DRIFT,
) -> str:
    """
    Extracts audio directly from video_url via ffmpeg, then time-stretches
    it (pitch-preserving) so its duration exactly matches the video's own
    duration. This keeps transcript timestamps mapped 1:1 onto the video,
    which otherwise drifts on long recordings due to encoder clock skew.

    Prints video length, raw audio length, and the computed stretch
    factor. Raises if the drift implied by that factor is larger than
    MAX_STRETCH_DRIFT, since that usually signals a broken source stream
    rather than ordinary clock drift.
    """
    extract_command = [
        "ffmpeg",
        "-y",
        "-reconnect", "1",
        "-reconnect_streamed", "1",
        "-reconnect_delay_max", "5",
        "-i", video_url,
        "-vn",
        "-acodec", "pcm_s16le",
        "-ar", "16000",
        "-ac", "1",
        raw_audio_path,
    ]

    print("Extracting audio from URL...")
    extract_result = subprocess.run(extract_command, capture_output=True, text=True)
    if extract_result.returncode != 0:
        raise RuntimeError(f"ffmpeg extraction failed:\n{extract_result.stderr}")

    print("Probing durations...")
    video_dur = _get_duration(video_url, stream="v:0")
    raw_audio_dur = _get_duration(raw_audio_path, stream="a:0")
    atempo_factor = raw_audio_dur / video_dur
    drift_pct = abs(atempo_factor - 1.0)

    print("\n--- Stretch indicator -----------------------------")
    print(f"Video length:      {video_dur:.2f}s  ({_seconds_to_hms(video_dur)})")
    print(f"Raw audio length:  {raw_audio_dur:.2f}s  ({_seconds_to_hms(raw_audio_dur)})")
    print(f"Drift before stretch: {video_dur - raw_audio_dur:+.2f}s")
    print(f"Computed atempo factor: {atempo_factor:.6f}  ({drift_pct * 100:.2f}% drift)")
    print("-----------------------------------------------------\n")

    if drift_pct > max_stretch_drift:
        raise RuntimeError(
            f"Stretch factor {atempo_factor:.6f} implies {drift_pct * 100:.2f}% drift, "
            f"which exceeds the {max_stretch_drift * 100:.0f}% safety limit. "
            f"This usually means the source stream itself is broken (dropped frames, "
            f"VFR video, corrupt container) rather than ordinary clock drift -- "
            f"stretching audio by this much would badly desync the transcript."
        )

    atempo_chain = _build_atempo_chain(atempo_factor)

    stretch_command = [
        "ffmpeg",
        "-y",
        "-i", raw_audio_path,
        "-filter:a", atempo_chain,
        "-ar", "16000",
        "-ac", "1",
        stretched_audio_path,
    ]

    print("Time-stretching audio to match video duration...")
    stretch_result = subprocess.run(stretch_command, capture_output=True, text=True)
    if stretch_result.returncode != 0:
        raise RuntimeError(f"ffmpeg stretch failed:\n{stretch_result.stderr}")

    final_audio_dur = _get_duration(stretched_audio_path, stream="a:0")
    print(f"Stretched audio length: {final_audio_dur:.2f}s  ({_seconds_to_hms(final_audio_dur)})")
    print(f"Remaining drift: {video_dur - final_audio_dur:+.3f}s\n")

    if not keep_raw_audio and os.path.exists(raw_audio_path):
        os.remove(raw_audio_path)

    return stretched_audio_path


# ------------------------------------
# Chunking helper (for Whisper's upload size limit)
# ------------------------------------
def _split_audio(file_path: str, chunk_dir: str = "_chunks"):
    os.makedirs(chunk_dir, exist_ok=True)

    audio = AudioSegment.from_file(file_path)
    total_ms = len(audio)
    total_bytes = os.path.getsize(file_path)

    bytes_per_ms = total_bytes / total_ms
    chunk_ms = max(1000, int(MAX_CHUNK_BYTES / bytes_per_ms))

    chunks = []
    for i, start_ms in enumerate(range(0, total_ms, chunk_ms)):
        end_ms = min(start_ms + chunk_ms, total_ms)
        piece = audio[start_ms:end_ms]

        chunk_path = os.path.join(chunk_dir, f"chunk_{i}.wav")
        piece.export(chunk_path, format="wav")
        chunks.append((chunk_path, start_ms / 1000.0))

    return chunks, chunk_dir


# ------------------------------------
# Silero VAD silence detection
# ------------------------------------
def _merge_close_silences(silences: list[dict], max_gap: float) -> list[dict]:
    """
    Merges consecutive silence regions whose gap -- the stretch of
    "speech" sitting between them -- is shorter than `max_gap` seconds
    into a single silence region spanning the whole run.

    This exists on top of the speech-merge pass inside
    _detect_silence_silero(): that earlier pass only merges raw VAD
    speech segments that are themselves close together, so it can't
    catch an ISOLATED short speech blip (e.g. a single stray word or
    cough) sitting far from any other speech, in between two otherwise
    long silences -- that blip has nothing nearby to merge with during
    the speech pass, yet still splits what should be reported as one
    continuous silence into two separate entries.

    The small gap itself is swallowed, not preserved: the merged range
    is [first.start, last.end], with `duration` recomputed from that.
    No rounding is applied here, matching the full-precision timestamps
    the rest of this module reports.

    Silences are sorted by start first since callers may pass them in
    any order.
    """
    if not silences:
        return []

    ordered = sorted(silences, key=lambda s: s["start"])
    merged = [dict(ordered[0])]

    for s in ordered[1:]:
        last = merged[-1]
        gap = s["start"] - last["end"]
        if gap < max_gap:
            # Absorb the gap: extend the current run's end forward (use
            # max() rather than assuming sorted-by-start also means
            # sorted-by-end, in case of any overlapping/out-of-order
            # input).
            last["end"] = max(last["end"], s["end"])
            last["duration"] = last["end"] - last["start"]
        else:
            merged.append(dict(s))

    return merged


def _detect_silence_silero(
    audio_path: str,
    total_duration: float,
    min_silence_seconds: float = MIN_SILENCE_SECONDS,
    merge_gap_seconds: float = MERGE_GAP_SECONDS,
) -> list[dict]:
    """
    Runs Silero VAD to find every speech segment in the file, merges
    segments separated by less than merge_gap_seconds, then reports
    every remaining gap (including leading silence before the first
    speech and trailing silence after the last) that is at least
    min_silence_seconds long.

    Before returning, any two of those reported silence regions that
    ended up separated by less than merge_gap_seconds of intervening
    "speech" (an isolated blip too far from other speech to have been
    caught by the earlier speech-merge pass -- see
    _merge_close_silences()) are merged into one, swallowing that short
    gap.

    Silence boundaries are reported at full precision (no rounding) --
    the exact start/end/duration the VAD model produced.
    """
    model = _get_vad_model()

    print("Running Silero VAD...")
    wav = _load_wav_for_vad(audio_path, expected_sample_rate=_VAD_SAMPLE_RATE)
    result = get_speech_timestamps(
        wav,
        model,
        sampling_rate=_VAD_SAMPLE_RATE,
        return_seconds=True,
    )

    speech_segments = [{"start": seg["start"], "end": seg["end"]} for seg in result]
    print(f"Speech segments (raw): {len(speech_segments)}")

    merged = []
    for seg in speech_segments:
        if not merged:
            merged.append(seg)
            continue
        if seg["start"] - merged[-1]["end"] <= merge_gap_seconds:
            merged[-1]["end"] = max(merged[-1]["end"], seg["end"])
        else:
            merged.append(seg)
    print(f"Speech segments (merged): {len(merged)}")

    def _make_silence(start: float, end: float) -> dict | None:
        if end <= start:
            return None
        return {
            "start": start,
            "end": end,
            "duration": end - start,
        }

    silence = []
    prev_end = 0.0

    for seg in merged:
        gap = seg["start"] - prev_end
        if gap >= min_silence_seconds:
            entry = _make_silence(prev_end, seg["start"])
            if entry:
                silence.append(entry)
        prev_end = seg["end"]

    # Trailing silence, always checked against total_duration so it's
    # captured even if merged speech segments run right up near the end.
    tail_gap = total_duration - prev_end
    if tail_gap >= min_silence_seconds:
        entry = _make_silence(prev_end, total_duration)
        if entry:
            silence.append(entry)

    print(f"Silence regions (>= {min_silence_seconds}s): {len(silence)}")

    # Second merge pass, on the final silence list itself -- catches an
    # isolated short speech blip sitting between two long silences (see
    # _merge_close_silences() docstring for why the earlier speech-merge
    # pass can't catch this case on its own).
    before_merge = len(silence)
    silence = _merge_close_silences(silence, merge_gap_seconds)
    if len(silence) != before_merge:
        print(
            f"Merged silence regions separated by < {merge_gap_seconds}s of "
            f"intervening speech: {before_merge} -> {len(silence)}"
        )

    return silence


def _overlaps_silence(word: dict, silence_zones: list[dict]) -> bool:
    """
    True if the word's [start, end] interval touches a silence zone at
    all -- including merely sharing an endpoint -- not just if its
    midpoint falls inside one. This is intentionally aggressive: any
    word whose audio even brushes a silence region gets dropped.
    """
    return any(word["start"] <= z["end"] and word["end"] >= z["start"] for z in silence_zones)


# ------------------------------------
# Sentence-boundary helpers (word-level rebuild)
# ------------------------------------
def _is_sentence_end(word_text: str, next_word_text: str | None) -> bool:
    stripped = word_text.rstrip("\"')]»”’")
    if not stripped or stripped[-1] not in ".!?":
        return False

    if stripped[-1] == ".":
        core = stripped[:-1].lower()
        if core in _ABBREVIATIONS:
            return False
        if len(core) <= 2 and core.isalpha():
            return False

    if next_word_text:
        first_alpha = next((c for c in next_word_text if c.isalpha()), None)
        if first_alpha is not None and first_alpha.islower():
            return False

    return True


def _gap_overlaps_silence(gap_start: float, gap_end: float, silence_zones: list[dict]) -> bool:
    """
    True if any silence zone overlaps the (gap_start, gap_end) interval
    that sits between two consecutive surviving words. Used to force a
    sentence cut even when no sentence-ending punctuation was found,
    so a silence region can never end up sitting inside a sentence.
    """
    return any(z["start"] < gap_end and z["end"] > gap_start for z in silence_zones)


def _words_to_sentences(words: list[dict], silence_zones: list[dict] | None = None) -> list[dict]:
    """
    Rebuilds sentence-level segments from word-level timestamps.

    A sentence is flushed when any of the following happens:
      - natural sentence-ending punctuation is found,
      - the running word count hits MAX_SENTENCE_WORDS, or
      - a Silero VAD silence zone falls in the gap between the current
        word and the next one. This guarantees a silence region is
        never left sitting inside a sentence's [start, end] span, even
        if the sentence hadn't grammatically ended yet.
    """
    silence_zones = silence_zones or []
    sentences = []
    current_words: list[dict] = []
    seg_id = 0

    def _flush():
        nonlocal seg_id
        if not current_words:
            return
        start = current_words[0]["start"]
        end = current_words[-1]["end"]
        text = " ".join(w["text"] for w in current_words).strip()
        sentences.append(
            {
                "id": seg_id,
                "start": round(start, 2),
                "end": round(end, 2),
                "duration": round(end - start, 2),
                "text": text,
            }
        )
        seg_id += 1

    for i, w in enumerate(words):
        current_words.append(w)
        next_word = words[i + 1] if i + 1 < len(words) else None
        next_text = next_word["text"] if next_word else None

        sentence_ends_naturally = _is_sentence_end(w["text"], next_text)
        hit_word_cap = len(current_words) >= MAX_SENTENCE_WORDS
        # Force a cut here (even mid-sentence) if a silence zone sits in
        # the gap before the next word -- a silence region must never
        # end up inside a sentence.
        silence_forces_cut = next_word is not None and _gap_overlaps_silence(
            w["end"], next_word["start"], silence_zones
        )

        if sentence_ends_naturally or hit_word_cap or silence_forces_cut:
            _flush()
            current_words = []

    _flush()
    return sentences


# ------------------------------------
# Transcription + Silero-VAD-based cleanup
# ------------------------------------
def ingestion(
    video_url: str,
    raw_audio_path: str = "audio_raw.wav",
    audio_path: str = "audio.wav",
    keep_audio: bool = True,
    min_silence_seconds: float = MIN_SILENCE_SECONDS,
    max_stretch_drift: float = MAX_STRETCH_DRIFT,
):
    """
    1. Extracts audio directly from video_url and time-stretches it to
       exactly match the video's duration (prints video/raw-audio
       length and the stretch factor; raises if drift exceeds
       max_stretch_drift).
    2. Runs Silero VAD on the final stretched audio to find every
       silence region >= min_silence_seconds (including leading and
       trailing silence), merges any two of those regions separated by
       less than MERGE_GAP_SECONDS of intervening speech, and reports
       the result at full precision (no rounding).
    3. Transcribes the same audio with Groq Whisper (word-level
       timestamps, chunked if needed).
    4. Drops every transcribed word that touches a Silero-VAD-detected
       silence region at all -- any overlap, even just sharing an
       endpoint -- not only words whose midpoint falls inside one. This
       removes hallucinated segments sitting in silence and trims the
       boundaries of segments that only partially overlap silence.
    5. Regroups surviving words into sentence-level segments, forcing a
       sentence break wherever a silence zone falls in the gap between
       two consecutive surviving words -- even without sentence-ending
       punctuation -- so a silence region can never end up sitting
       inside a transcript sentence.

    Returns
    -------
    {
        "segments": [...],       # cleaned sentence-level transcript
        "text": "...",           # all sentences joined into one paragraph
        "max_duration": ...,
        "silences": [...],       # Silero-VAD-detected silence zones
    }
    """
    _extract_and_stretch_audio(
        video_url, raw_audio_path, audio_path,
        keep_raw_audio=False, max_stretch_drift=max_stretch_drift,
    )

    total_duration = _get_duration(audio_path, stream="a:0")

    # Silence detection runs on the final (stretched) audio, so its
    # timestamps line up exactly with the transcript we're about to
    # produce from the same file.
    silence_zones = _detect_silence_silero(audio_path, total_duration, min_silence_seconds)

    file_size = os.path.getsize(audio_path)

    chunk_dir = None
    if file_size <= MAX_CHUNK_BYTES:
        chunk_list = [(audio_path, 0.0)]
    else:
        chunk_list, chunk_dir = _split_audio(audio_path)

    all_words = []

    try:
        for chunk_path, offset in chunk_list:
            with open(chunk_path, "rb") as audio_file:
                transcription = client.audio.transcriptions.create(
                    file=audio_file,
                    model="whisper-large-v3-turbo",
                    response_format="verbose_json",
                    timestamp_granularities=["word", "segment"],
                )

            chunk_words = getattr(transcription, "words", None)

            if chunk_words:
                for w in chunk_words:
                    all_words.append(
                        {"text": w["word"].strip(), "start": w["start"] + offset, "end": w["end"] + offset}
                    )
            else:
                for seg in transcription.segments:
                    all_words.append(
                        {"text": seg["text"].strip(), "start": seg["start"] + offset, "end": seg["end"] + offset}
                    )
    finally:
        if chunk_dir:
            for chunk_path, _ in chunk_list:
                if os.path.exists(chunk_path):
                    os.remove(chunk_path)
            if os.path.isdir(chunk_dir) and not os.listdir(chunk_dir):
                os.rmdir(chunk_dir)

    # Drop any word that touches a Silero-VAD-confirmed silence zone at
    # all -- full overlap, partial overlap, or just sharing an endpoint.
    # This both removes hallucinated segments sitting in silence and
    # trims the boundaries of segments that only partially overlap
    # silence, since sentences are rebuilt from surviving words.
    before = len(all_words)
    all_words = [w for w in all_words if not _overlaps_silence(w, silence_zones)]
    dropped = before - len(all_words)
    if dropped:
        print(f"Dropped {dropped} word(s) touching Silero-VAD-detected silence.")

    sentence_segments = _words_to_sentences(all_words, silence_zones)

    # Single flat paragraph made of every sentence in order, for callers
    # that want the full transcript as one block of text rather than
    # iterating over "segments".
    full_text = " ".join(s["text"].strip() for s in sentence_segments).strip()

    if not keep_audio and os.path.exists(audio_path):
        os.remove(audio_path)

    return {
        "segments": sentence_segments,
        "text": full_text,
        "max_duration": round(total_duration, 2),
        "silences": silence_zones,
    }


if __name__ == "__main__":
    V = input("Enter URL : ").strip()
    transcript = ingestion(video_url=V)

    project = Path(__file__).parent

    transcript_path = project / "transcript.json"
    with open(transcript_path, "w", encoding="utf-8") as f:
        json.dump(transcript, f, indent=4, ensure_ascii=False)

    silence_path = project / "silence.json"
    silence_data = {
        "video_url": V,
        "max_duration": transcript["max_duration"],
        "min_silence_seconds": MIN_SILENCE_SECONDS,
        "silences": transcript["silences"],
    }
    with open(silence_path, "w", encoding="utf-8") as f:
        json.dump(silence_data, f, indent=4, ensure_ascii=False)

    print(f"Saved transcript ({len(transcript['segments'])} sentences, "
          f"{len(transcript['text'])} chars of full text) to {transcript_path}")
    print(f"Saved {len(transcript['silences'])} silence range(s) to {silence_path}")