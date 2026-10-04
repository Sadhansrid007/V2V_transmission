"""
transcribe.py
-------------
Job: audio file -> transcript with WORD-LEVEL timestamps, saved to disk.

Uses Groq's hosted Whisper API (whisper-large-v3) instead of running
Whisper locally -- much faster, and offloads compute so this works fine
even on a laptop with no GPU.

Word-level (not just segment-level) timestamps matter because later,
av_summary.py needs to cut out individual filler words like "um" without
chopping out the whole sentence they sit in.

Lectures run ~1 hour, which is way over Groq's 25MB upload cap. So for
anything longer than ~10 minutes, we hand off to audio_chunker.py to
split the file at silence gaps, transcribe each piece separately, then
stitch the pieces back into one transcript -- offsetting every
timestamp so they're correct against the ORIGINAL full-length audio,
not just the chunk it came from.

We create the Groq client once at import time rather than on every call.
"""

import json
import shutil
from pathlib import Path

from groq import Groq

from audio_chunker import split_on_silence
from config import GROQ_API_KEY, TRANSCRIPT_DIR

_client = None


def _get_client() -> Groq:
    global _client
    if _client is None:
        _client = Groq(api_key=GROQ_API_KEY)
    return _client


def _transcribe_chunk(chunk_path: Path) -> dict:
    """
    Runs Groq's whisper-large-v3 on a single audio file (assumed to
    already be under the upload size limit). Returns the raw cleaned
    dict for JUST this chunk, with timestamps relative to the chunk's
    own start (0.0), not yet offset.
    """
    client = _get_client()
    with open(chunk_path, "rb") as f:
        result = client.audio.transcriptions.create(
            file=(chunk_path.name, f.read()),
            model="whisper-large-v3",
            response_format="verbose_json",
            timestamp_granularities=["word", "segment"],
        )

    raw = result.model_dump() if hasattr(result, "model_dump") else dict(result)

    # Groq returns a flat "words" list (not nested per segment), so we
    # bucket words into their segment by matching on time overlap. This
    # keeps the same output shape the rest of the pipeline expects.
    segments_out = []
    all_words = raw.get("words", [])
    for seg in raw.get("segments", []):
        seg_start, seg_end = seg["start"], seg["end"]
        seg_words = [
            {"word": w["word"].strip(), "start": w["start"], "end": w["end"]}
            for w in all_words
            if seg_start <= w["start"] < seg_end
        ]
        segments_out.append({
            "start": seg_start,
            "end": seg_end,
            "text": seg["text"].strip(),
            "words": seg_words,
        })

    return {
        "text": raw.get("text", ""),
        "segments": segments_out,
    }


def transcribe(audio_path: Path, video_id: str) -> dict:
    """
    Transcribes audio_path (splitting into silence-aware chunks first
    if it's too large for a single Groq upload). Returns and saves a
    dict shaped like:

    {
      "text": "full transcript as one string",
      "segments": [
        {
          "start": 12.4, "end": 18.9, "text": "so structures let you...",
          "words": [
            {"word": "so", "start": 12.4, "end": 12.6},
            {"word": "structures", "start": 12.6, "end": 13.1},
            ...
          ]
        },
        ...
      ]
    }
    """
    transcript_path = TRANSCRIPT_DIR / f"{video_id}.json"
    if transcript_path.exists():
        return json.loads(transcript_path.read_text())

    audio_path = Path(audio_path)
    chunks = split_on_silence(audio_path)
    is_chunked = len(chunks) > 1 or chunks[0]["path"] != audio_path

    all_text_parts = []
    all_segments = []

    for chunk in chunks:
        chunk_result = _transcribe_chunk(chunk["path"])
        offset = chunk["offset_seconds"]

        all_text_parts.append(chunk_result["text"].strip())

        for seg in chunk_result["segments"]:
            all_segments.append({
                "start": seg["start"] + offset,
                "end": seg["end"] + offset,
                "text": seg["text"],
                "words": [
                    {"word": w["word"], "start": w["start"] + offset, "end": w["end"] + offset}
                    for w in seg["words"]
                ],
            })

    # Clean up temp chunk files -- only if we actually created a temp
    # dir (i.e. the file was big enough to need splitting).
    if is_chunked:
        temp_dir = chunks[0]["path"].parent
        shutil.rmtree(temp_dir, ignore_errors=True)

    cleaned = {
        "text": " ".join(all_text_parts),
        "segments": all_segments,
    }

    transcript_path.write_text(json.dumps(cleaned, indent=2))
    return cleaned


if __name__ == "__main__":
    import sys
    from config import AUDIO_DIR
    video_id = sys.argv[1]
    audio_path = AUDIO_DIR / f"{video_id}.wav"
    out = transcribe(audio_path, video_id)
    print(f"Transcribed {len(out['segments'])} segments across the full lecture.")