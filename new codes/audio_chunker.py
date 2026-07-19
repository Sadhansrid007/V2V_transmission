"""
audio_chunker.py
-----------------
Job: split a long audio file into Groq-upload-sized chunks, cutting at
silence gaps instead of fixed time marks so we never slice through a
word mid-syllable.

Nothing in here knows about transcription -- it just hands back a list
of (chunk_path, offset_seconds) tuples. transcribe.py is responsible
for stitching the transcribed pieces back together.
"""

import tempfile
from pathlib import Path

from pydub import AudioSegment
from pydub.silence import detect_silence

from config import MIN_SILENCE_LEN_MS, SILENCE_THRESH_DB

# Groq's free tier caps uploads at 25MB. 16kHz mono 16-bit WAV runs
# ~1.92MB/minute, so 25MB / 1.92 ≈ 13 minutes is the hard ceiling.
# We target well under that so we have room to slide the cut point
# to the nearest silence without accidentally going over.
TARGET_CHUNK_MINUTES = 10
SEARCH_WINDOW_MINUTES = 2  # how far to look for a silence gap near the target cut


def _find_cut_point(silence_ranges: list[tuple[int, int]], target_ms: int, window_ms: int, audio_len_ms: int) -> int:
    """
    Finds the best place to cut near target_ms. Prefers the midpoint of
    a silence gap inside [target_ms - window_ms, target_ms + window_ms].
    Falls back to a hard cut at target_ms if no silence is nearby.
    """
    if target_ms >= audio_len_ms:
        return audio_len_ms

    best_gap = None
    best_distance = window_ms + 1

    for start, end in silence_ranges:
        midpoint = (start + end) // 2
        distance = abs(midpoint - target_ms)
        if distance <= window_ms and distance < best_distance:
            best_gap = midpoint
            best_distance = distance

    return best_gap if best_gap is not None else target_ms


def split_on_silence(audio_path: Path) -> list[dict]:
    """
    Splits audio_path into chunks small enough for Groq's upload limit,
    cutting at silence gaps near each target boundary.

    Returns a list of dicts:
      [{"path": Path, "offset_seconds": float}, ...]

    Chunk files are written to a temp directory. Caller is responsible
    for cleaning them up (see transcribe.py, which deletes them after
    stitching the final transcript together).
    """
    audio = AudioSegment.from_wav(str(audio_path))
    audio_len_ms = len(audio)
    target_ms = TARGET_CHUNK_MINUTES * 60 * 1000

    # If the whole file already fits in one chunk, skip all of this.
    if audio_len_ms <= target_ms:
        return [{"path": audio_path, "offset_seconds": 0.0}]

    silence_ranges = detect_silence(
        audio,
        min_silence_len=MIN_SILENCE_LEN_MS,
        silence_thresh=SILENCE_THRESH_DB,
    )

    window_ms = SEARCH_WINDOW_MINUTES * 60 * 1000
    tmp_dir = Path(tempfile.mkdtemp(prefix="lecturelens_chunks_"))

    chunks = []
    cursor_ms = 0
    chunk_index = 0

    while cursor_ms < audio_len_ms:
        target_cut = cursor_ms + target_ms
        cut_point = _find_cut_point(silence_ranges, target_cut, window_ms, audio_len_ms)

        # Safety: never produce a zero-length or backwards chunk.
        if cut_point <= cursor_ms:
            cut_point = min(cursor_ms + target_ms, audio_len_ms)

        chunk_audio = audio[cursor_ms:cut_point]
        chunk_path = tmp_dir / f"chunk_{chunk_index:03d}.wav"
        chunk_audio.export(str(chunk_path), format="wav")

        chunks.append({
            "path": chunk_path,
            "offset_seconds": cursor_ms / 1000.0,
        })

        cursor_ms = cut_point
        chunk_index += 1

    return chunks