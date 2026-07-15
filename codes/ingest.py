"""
ingest.py
---------
Job: take a URL (YouTube, Google Drive, or direct .mp4 link) OR an
uploaded file's raw bytes, and produce two local files: the video,
and its extracted audio (.wav).

Nothing in here knows or cares about transcription, RAG, or cutting.
That separation is the whole point.
"""

import re
import subprocess
import uuid
from pathlib import Path

import gdown
import yt_dlp

from config import VIDEO_DIR, AUDIO_DIR


def _make_video_id(seed: str) -> str:
    """Short stable-ish id for this video, used to name every downstream file."""
    return uuid.uuid5(uuid.NAMESPACE_URL, seed).hex[:10]


def _is_youtube(url: str) -> bool:
    return "youtube.com" in url or "youtu.be" in url


def _is_gdrive(url: str) -> bool:
    return "drive.google.com" in url


def download_video(url: str) -> tuple[str, Path]:
    """
    Downloads the video from any supported source.
    Returns (video_id, path_to_video_file).
    """
    video_id = _make_video_id(url)
    out_path = VIDEO_DIR / f"{video_id}.mp4"

    if out_path.exists():
        return video_id, out_path

    if _is_youtube(url):
        ydl_opts = {
            "format": "bestvideo+bestaudio/best",
            "outtmpl": str(out_path),
            "merge_output_format": "mp4",
            "quiet": True,
        }
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            ydl.download([url])

    elif _is_gdrive(url):
        file_id_match = re.search(r"/d/([a-zA-Z0-9_-]+)", url)
        if not file_id_match:
            raise ValueError("Couldn't parse a file ID out of that Drive URL.")
        file_id = file_id_match.group(1)
        gdown.download(id=file_id, output=str(out_path), quiet=False)

    else:
        # assume direct link to an mp4 or similar
        subprocess.run(
            ["curl", "-L", "-o", str(out_path), url],
            check=True,
        )

    if not out_path.exists():
        raise RuntimeError(f"Download appears to have failed for {url}")

    return video_id, out_path


def save_uploaded_video(file_bytes: bytes, filename: str) -> tuple[str, Path]:
    """
    Saves a video that was uploaded directly (e.g. via Streamlit's
    file_uploader) rather than downloaded from a URL. Returns
    (video_id, path_to_video_file), same shape as download_video, so
    the rest of the pipeline doesn't need to know which path a video
    came in through.
    """
    # Hash on size + filename so re-uploading the exact same file
    # reuses the cached video/audio/transcript instead of reprocessing.
    seed = f"{filename}:{len(file_bytes)}"
    video_id = _make_video_id(seed)
    out_path = VIDEO_DIR / f"{video_id}.mp4"

    if not out_path.exists():
        out_path.write_bytes(file_bytes)

    return video_id, out_path


def extract_audio(video_path: Path, video_id: str) -> Path:
    """
    Pulls a mono 16kHz WAV out of the video. 16kHz mono is what Whisper
    wants internally anyway, so we do the conversion once here instead
    of making Whisper (or ffmpeg, twice) redo it later.
    """
    audio_path = AUDIO_DIR / f"{video_id}.wav"
    if audio_path.exists():
        return audio_path

    subprocess.run(
        [
            "ffmpeg", "-y",
            "-i", str(video_path),
            "-ac", "1",          # mono
            "-ar", "16000",      # 16kHz
            "-vn",                # no video stream
            str(audio_path),
        ],
        check=True,
        capture_output=True,
    )
    return audio_path


def ingest_url(url: str) -> dict:
    """Entry point for URL-based input. One call, two files, done."""
    video_id, video_path = download_video(url)
    audio_path = extract_audio(video_path, video_id)
    return {"video_id": video_id, "video_path": video_path, "audio_path": audio_path}


def ingest_upload(file_bytes: bytes, filename: str) -> dict:
    """Entry point for direct file-upload input."""
    video_id, video_path = save_uploaded_video(file_bytes, filename)
    audio_path = extract_audio(video_path, video_id)
    return {"video_id": video_id, "video_path": video_path, "audio_path": audio_path}


if __name__ == "__main__":
    import sys
    result = ingest_url(sys.argv[1])
    print(result)