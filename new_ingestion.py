import os
import re
import subprocess
import tempfile
from urllib.parse import urlparse

from groq import Groq
from dotenv import load_dotenv
from pydub import AudioSegment
from pathlib import Path
import json

load_dotenv()

client = Groq(api_key=os.getenv("GROQ_API_KEY"))
# ------------------------------------
# Create Groq client (once)
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
# Lowercased, without the trailing period.
_ABBREVIATIONS = {
    "mr", "mrs", "ms", "dr", "prof", "sr", "jr", "vs", "etc",
    "e.g", "i.e", "u.s", "u.k", "st", "no", "vol", "fig", "approx",
    "gen", "capt", "col", "sgt", "lt", "rev", "inc", "ltd", "co",
}

# Whisper only ever returns time ranges where it heard speech -- any gap
# between what it returned (including before the first word or after the
# last word) never shows up in the transcript at all. Any such gap at
# least this long gets recorded as a silence range instead of silently
# disappearing.
SILENCE_GAP_SECONDS = 20.0

# ------------------------------------
# Acoustic silence detection (ground truth from the waveform itself,
# not inferred from transcript timing/gaps)
# ------------------------------------

# How many dB above this recording's own noise floor still counts as
# "basically silent". Measured relative to a per-file calibrated floor
# rather than one fixed number, since mic gain/room noise vary a lot
# video to video.
SILENCE_MARGIN_DB = 6.0

# Fallback only used when the file has no known-silent gaps at all to
# calibrate against (e.g. a lecture that never stops talking).
DEFAULT_SILENCE_DBFS = -45.0


# ------------------------------------
# Audio extraction helper (ffmpeg pulls straight from the URL)
# ------------------------------------
def _extract_audio(video_url: str, audio_path: str = "audio.wav") -> str:
    """
    Uses ffmpeg to fetch only the audio track directly from the remote
    video URL. ffmpeg does its own HTTP(S) streaming and demuxing, so
    the video is never downloaded or written to disk in full -- only
    the decoded audio ends up on disk, at audio_path.

    Requires ffmpeg to be installed and on PATH.
    """
    cmd = [
        "ffmpeg",
        "-y",              # overwrite audio_path if it already exists
        "-i", video_url,   # ffmpeg fetches the URL itself
        "-vn",              # drop the video stream entirely
        "-acodec", "pcm_s16le",
        "-ar", "16000",
        "-ac", "1",
        audio_path,
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"ffmpeg failed to extract audio:\n{result.stderr}")
    return audio_path


def _get_audio_duration(audio_path: str) -> float:
    """
    Returns the total duration of the audio file, in seconds, via
    ffprobe (ships alongside ffmpeg) rather than loading the whole file
    into memory a second time just to check its length.
    """
    cmd = [
        "ffprobe",
        "-v", "error",
        "-show_entries", "format=duration",
        "-of", "default=noprint_wrappers=1:nokey=1",
        audio_path,
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"ffprobe failed to read audio duration:\n{result.stderr}")
    return float(result.stdout.strip())


# ------------------------------------
# Chunking helper
# ------------------------------------
def _split_audio(file_path: str, chunk_dir: str = "_chunks"):
    """
    Splits an audio/video file into WAV chunks, each under MAX_CHUNK_BYTES.
    Returns a list of (chunk_path, start_offset_seconds) tuples.
    """
    os.makedirs(chunk_dir, exist_ok=True)

    audio = AudioSegment.from_file(file_path)
    total_ms = len(audio)
    total_bytes = os.path.getsize(file_path)

    # Estimate how many ms of audio fit in MAX_CHUNK_BYTES based on the
    # source file's average bitrate, then chunk on that duration.
    bytes_per_ms = total_bytes / total_ms
    chunk_ms = max(1000, int(MAX_CHUNK_BYTES / bytes_per_ms))

    chunks = []
    for i, start_ms in enumerate(range(0, total_ms, chunk_ms)):
        end_ms = min(start_ms + chunk_ms, total_ms)
        piece = audio[start_ms:end_ms]

        chunk_path = os.path.join(chunk_dir, f"chunk_{i}.wav")
        # WAV/PCM re-encode so the exported chunk size is predictable
        # regardless of the source format/codec.
        piece.export(chunk_path, format="wav")
        chunks.append((chunk_path, start_ms / 1000.0))

    return chunks, chunk_dir


# ------------------------------------
# Sentence-boundary helpers (word-level rebuild)
# ------------------------------------
def _is_sentence_end(word_text: str, next_word_text: str | None) -> bool:
    """
    Decides whether `word_text` (already stripped of surrounding
    whitespace) closes a sentence, given the word that follows it.

    Heuristics:
    - Must end in ., !, or ? (after stripping trailing quotes/brackets).
    - A trailing "." is ignored if the word (minus the period) is a
      known abbreviation, or is a bare 1-2 letter token (likely an
      initial, e.g. "A." in "A. Turing").
    - Even if punctuation looks sentence-final, we don't split if the
      next word starts with a lowercase letter -- real sentence starts
      are capitalized in Whisper's punctuated output, so a lowercase
      follow-on word usually means the "." was an abbreviation/decimal
      Whisper didn't punctuate as such.
    """
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


def _words_to_sentences(words: list[dict]) -> list[dict]:
    """
    Takes a flat, time-ordered list of {"text", "start", "end"} word
    dicts (already offset-corrected across chunks) and groups them into
    sentence-level segments.

    Returns a list of {"id", "start", "end", "duration", "text"} dicts,
    matching the shape the rest of the pipeline already expects.
    """
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
        next_text = words[i + 1]["text"] if i + 1 < len(words) else None

        if _is_sentence_end(w["text"], next_text) or len(current_words) >= MAX_SENTENCE_WORDS:
            _flush()
            current_words = []

    _flush()  # trailing partial sentence with no terminal punctuation
    return sentences


# ------------------------------------
# Silence detection (gaps the transcript never captures)
# ------------------------------------
def _find_silence_ranges(
    segments: list[dict],
    total_duration: float,
    min_gap: float = SILENCE_GAP_SECONDS,
) -> list[dict]:
    """
    Walks the time-ordered sentence segments and finds every gap of at
    least `min_gap` seconds where no speech was transcribed:

    - Leading silence: audio starts before the first word is spoken
      (e.g. intro music, dead air before a lecture starts).
    - Internal gaps: silence between one sentence ending and the next
      one starting.
    - Trailing silence: speech stops before the audio itself ends
      (e.g. the speaker stops talking but the recording keeps rolling).

    Returns a list of {"start", "end", "duration"} dicts, in seconds.
    """
    ranges = []

    if not segments:
        # Nothing was transcribed at all -- the whole file is silence.
        if total_duration >= min_gap:
            ranges.append(
                {
                    "start": 0.0,
                    "end": round(total_duration, 2),
                    "duration": round(total_duration, 2),
                }
            )
        return ranges

    # Leading silence: before the first transcribed word.
    leading_gap = segments[0]["start"]
    if leading_gap >= min_gap:
        ranges.append(
            {
                "start": 0.0,
                "end": round(leading_gap, 2),
                "duration": round(leading_gap, 2),
            }
        )

    # Internal gaps: between consecutive sentences.
    for prev_seg, next_seg in zip(segments, segments[1:]):
        gap = next_seg["start"] - prev_seg["end"]
        if gap >= min_gap:
            ranges.append(
                {
                    "start": round(prev_seg["end"], 2),
                    "end": round(next_seg["start"], 2),
                    "duration": round(gap, 2),
                }
            )

    # Trailing silence: after the last transcribed word.
    trailing_gap = total_duration - segments[-1]["end"]
    if trailing_gap >= min_gap:
        ranges.append(
            {
                "start": round(segments[-1]["end"], 2),
                "end": round(total_duration, 2),
                "duration": round(trailing_gap, 2),
            }
        )

    return ranges


# ------------------------------------
# Acoustic hallucination filtering
# ------------------------------------
def _dbfs(audio: AudioSegment, start: float, end: float) -> float:
    """
    Loudness (dBFS) of the audio between start/end seconds.
    Returns -120.0 (a stand-in for -inf) for a clip that's literally
    all-zero samples, since pydub returns -inf there and -inf breaks
    downstream comparisons/arithmetic.
    """
    start_ms = max(0, int(start * 1000))
    end_ms = min(len(audio), int(end * 1000))
    if end_ms <= start_ms:
        return -120.0
    loudness = audio[start_ms:end_ms].dBFS
    return loudness if loudness != float("-inf") else -120.0


def _estimate_noise_floor(audio: AudioSegment, known_silence: list[dict]) -> float:
    """
    Calibrates "what does silence actually sound like in THIS
    recording" by measuring real audio energy during gaps we already
    know are silent (no transcribed speech, gap >= SILENCE_GAP_SECONDS).
    Falls back to DEFAULT_SILENCE_DBFS when there are no such gaps to
    calibrate against (e.g. a lecture that never stops talking).
    """
    if not known_silence:
        return DEFAULT_SILENCE_DBFS
    readings = [_dbfs(audio, r["start"], r["end"]) for r in known_silence]
    return sum(readings) / len(readings)


def _strip_acoustic_hallucinations(
    segments: list[dict],
    audio: AudioSegment,
    noise_floor_dbfs: float,
    margin_db: float = SILENCE_MARGIN_DB,
) -> list[dict]:
    """
    Drops segments where the raw audio underneath the transcribed text
    is at (or below) this recording's own noise floor -- i.e. the mic
    wasn't picking up anything there, so whatever text Whisper produced
    for that stretch ("Thank you.", "Thanks for watching!", etc.) is a
    hallucination, regardless of how short/long the segment is or what
    sits next to it in the transcript.
    """
    threshold = noise_floor_dbfs + margin_db
    kept, dropped = [], 0

    for seg in segments:
        if _dbfs(audio, seg["start"], seg["end"]) <= threshold:
            dropped += 1
            continue
        kept.append(seg)

    if dropped:
        print(f"Dropped {dropped} segment(s) with no real audio underneath (mic effectively silent).")

    # Re-sequence ids so they stay valid list indices -- chunk_transcript.py
    # uses segment "id" directly as an index into `segments`.
    for new_id, seg in enumerate(kept):
        seg["id"] = new_id
    return kept


# ------------------------------------
# Transcription Function
# ------------------------------------
def ingestion(
    video_url: str,
    audio_path: str = "audio.wav",
    keep_audio: bool = True,
    min_silence_seconds: float = SILENCE_GAP_SECONDS,
):
    """
    Extracts the audio track directly from video_url via ffmpeg (no full
    video download) and saves it as audio_path (default "audio.wav"),
    then transcribes it using Groq Whisper Large V3 Turbo with
    word-level timestamps. Automatically splits audio larger than
    Groq's upload limit into chunks, transcribes each, stitches the
    word-level output back into one continuous timeline, and regroups
    it into sentence-level segments (rather than Whisper's silence-based
    segments) so chunk text no longer gets cut mid-sentence.

    Before silence ranges are computed, every segment is checked against
    the RAW AUDIO underneath its timestamps: if that stretch is at (or
    near) this recording's own noise floor, the segment is dropped as a
    hallucination -- the mic effectively wasn't picking up anything
    there, regardless of what text Whisper produced. This is ground
    truth from the waveform itself, not an inference from transcript
    timing, so it also catches hallucinations that a duration/gap-based
    heuristic would miss.

    Also records the audio's total duration and every gap of at least
    `min_silence_seconds` where no speech was transcribed -- leading
    silence, gaps between sentences, and trailing silence -- since
    Whisper's word/segment timestamps only ever cover speech and would
    otherwise make those gaps invisible.

    Parameters
    ----------
    video_url : str
        URL of the video whose audio should be extracted and transcribed.
    audio_path : str
        Where to save the extracted audio (default "audio.wav"). Kept on
        disk after this function returns unless keep_audio=False.
    keep_audio : bool
        If False, deletes audio_path once transcription (and acoustic
        analysis, which needs the file) is done.
    min_silence_seconds : float
        Minimum gap length (in seconds) to record as a silence range.

    Returns
    -------
    {
        "segments": [
            {
                "id": ...,
                "start": ...,
                "end": ...,
                "duration": ...,
                "text": ...   # one full sentence
            }
        ],
        "max_duration": ...,  # total audio duration, in seconds
        "silences": [
            {
                "start": ...,
                "end": ...,
                "duration": ...,
            }
        ]
    }
    """
    _extract_audio(video_url, audio_path)

    # Grab total duration now, before any cleanup below might delete
    # audio_path (only relevant when keep_audio=False).
    total_duration = _get_audio_duration(audio_path)

    file_size = os.path.getsize(audio_path)

    chunk_dir = None
    if file_size <= MAX_CHUNK_BYTES:
        chunk_list = [(audio_path, 0.0)]
    else:
        chunk_list, chunk_dir = _split_audio(audio_path)

    all_words = []  # flat, offset-corrected, time-ordered word stream

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
                        {
                            "text": w["word"].strip(),
                            "start": w["start"] + offset,
                            "end": w["end"] + offset,
                        }
                    )
            else:
                # Fallback: this chunk returned no word-level timestamps
                # (e.g. silent/near-empty chunk). Treat each segment's
                # text as a single "word" so it still participates in
                # sentence stitching instead of being dropped.
                for seg in transcription.segments:
                    all_words.append(
                        {
                            "text": seg["text"].strip(),
                            "start": seg["start"] + offset,
                            "end": seg["end"] + offset,
                        }
                    )
    finally:
        # Clean up temporary split chunks if we created any (these are
        # always disposable -- separate from audio_path itself).
        if chunk_dir:
            for chunk_path, _ in chunk_list:
                if os.path.exists(chunk_path):
                    os.remove(chunk_path)
            if os.path.isdir(chunk_dir) and not os.listdir(chunk_dir):
                os.rmdir(chunk_dir)

    sentence_segments = _words_to_sentences(all_words)

    # Acoustic pass: calibrate this file's noise floor from gaps already
    # known to be silent, then drop any segment whose underlying audio
    # is indistinguishable from that floor (mic effectively off there).
    # Needs the raw audio file, so this must run before any deletion.
    audio_for_analysis = AudioSegment.from_file(audio_path)
    raw_silences = _find_silence_ranges(sentence_segments, total_duration, min_silence_seconds)
    noise_floor = _estimate_noise_floor(audio_for_analysis, raw_silences)
    sentence_segments = _strip_acoustic_hallucinations(sentence_segments, audio_for_analysis, noise_floor)

    # Recompute now that hallucinated segments are gone -- a bogus
    # segment that used to split one long silent gap into two shorter
    # ones now correctly merges back into a single real range.
    silence_ranges = _find_silence_ranges(sentence_segments, total_duration, min_silence_seconds)

    if not keep_audio and os.path.exists(audio_path):
        os.remove(audio_path)

    return {
        "segments": sentence_segments,
        "max_duration": round(total_duration, 2),
        "silences": silence_ranges,
    }


# if __name__ == "__main__":
#     V = input("Enter URL : ").strip()
#     transcript = ingestion(video_url=V)

#     project = Path(__file__).parent

#     transcript_path = project / "transcript.json"
#     with open(transcript_path, "w", encoding="utf-8") as f:
#         json.dump(transcript, f, indent=4, ensure_ascii=False)

#     silence_path = project / "silence.json"
#     silence_data = {
#         "video_url": V,
#         "max_duration": transcript["max_duration"],
#         "min_silence_seconds": SILENCE_GAP_SECONDS,
#         "silences": transcript["silences"],
#     }
#     with open(silence_path, "w", encoding="utf-8") as f:
#         json.dump(silence_data, f, indent=4, ensure_ascii=False)

#     print(f"Saved transcript ({len(transcript['segments'])} sentences) to {transcript_path}")
#     print(f"Saved {len(transcript['silences'])} silence range(s) to {silence_path}")