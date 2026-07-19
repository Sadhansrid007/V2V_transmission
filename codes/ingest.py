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
from urllib.parse import urlparse

import gdown
import yt_dlp

from config import VIDEO_DIR, AUDIO_DIR, MAX_STRETCH_DRIFT


def _get_duration(path: Path, stream: str | None = None) -> float:
    """
    Returns duration in seconds via ffprobe. If `stream` is given (e.g.
    "v:0" or "a:0"), reads that stream's own duration for precision;
    falls back to container-level duration if the stream doesn't
    expose one directly.
    """
    if stream:
        result = subprocess.run(
            [
                "ffprobe", "-v", "error",
                "-select_streams", stream,
                "-show_entries", "stream=duration",
                "-of", "default=noprint_wrappers=1:nokey=1",
                str(path),
            ],
            capture_output=True, text=True,
        )
        value = result.stdout.strip()
        if result.returncode == 0 and value and value != "N/A":
            return float(value)

    result = subprocess.run(
        [
            "ffprobe", "-v", "error",
            "-show_entries", "format=duration",
            "-of", "default=noprint_wrappers=1:nokey=1",
            str(path),
        ],
        capture_output=True, text=True,
    )
    if result.returncode != 0:
        raise RuntimeError(f"ffprobe failed:\n{result.stderr}")
    return float(result.stdout.strip())


def _build_atempo_chain(factor: float) -> str:
    """ffmpeg's atempo filter only accepts 0.5-2.0 in a single pass;
    chain multiple atempo filters if the factor falls outside that."""
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
        # Assume direct link to an mp4 or similar. Some portals (like
        # college video servers) block requests that don't look like
        # they're coming from a real browser, so we send a User-Agent
        # and Referer header just like a browser would.
        parsed = urlparse(url)
        referer = f"{parsed.scheme}://{parsed.netloc}/"
        subprocess.run(
            [
                "curl", "-L", "-o", str(out_path),
                "-A", "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0 Safari/537.36",
                "-e", referer,
                url,
            ],
            check=True,
        )

    if not out_path.exists():
        raise RuntimeError(f"Download appears to have failed for {url}")

    # A real lecture video is at least a few MB. Anything smaller is
    # almost certainly an error/login/redirect page saved with a .mp4
    # name, not an actual video -- fail loudly here instead of letting
    # ffmpeg choke on it later with a confusing error.
    size_mb = out_path.stat().st_size / (1024 * 1024)
    if size_mb < 1:
        out_path.unlink()
        raise RuntimeError(
            f"Downloaded file for {url} is only {size_mb * 1024:.0f}KB -- "
            "this is likely an error page, not the actual video. The server "
            "may require authentication or block scripted downloads."
        )

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
    Pulls a mono 16kHz WAV out of the video, then time-stretches it
    (pitch-preserving) so its duration exactly matches the video's own
    duration.

    Why: long recordings can develop tiny clock drift between video and
    audio streams, which otherwise slowly desyncs transcript timestamps
    over an hour-long lecture. If the measured drift is larger than
    MAX_STRETCH_DRIFT, that usually means the source stream itself is
    broken rather than ordinary clock skew -- we skip stretching and
    keep the plain extracted audio rather than risk badly desyncing it
    with an aggressive, likely-wrong correction.
    """
    audio_path = AUDIO_DIR / f"{video_id}.wav"
    if audio_path.exists():
        return audio_path

    raw_path = AUDIO_DIR / f"{video_id}_raw.wav"
    subprocess.run(
        [
            "ffmpeg", "-y",
            "-i", str(video_path),
            "-ac", "1",          # mono
            "-ar", "16000",      # 16kHz
            "-vn",                # no video stream
            str(raw_path),
        ],
        check=True,
        capture_output=True,
    )

    try:
        video_dur = _get_duration(video_path, stream="v:0")
        raw_audio_dur = _get_duration(raw_path, stream="a:0")
        atempo_factor = raw_audio_dur / video_dur
        drift_pct = abs(atempo_factor - 1.0)

        if drift_pct > MAX_STRETCH_DRIFT:
            print(
                f"Audio/video drift is {drift_pct * 100:.2f}%, over the "
                f"{MAX_STRETCH_DRIFT * 100:.0f}% safety limit -- skipping "
                "stretch correction and using the audio as extracted "
                "(this usually means the source file itself has an issue, "
                "not just ordinary clock skew)."
            )
            raw_path.rename(audio_path)
            return audio_path

        atempo_chain = _build_atempo_chain(atempo_factor)
        subprocess.run(
            [
                "ffmpeg", "-y",
                "-i", str(raw_path),
                "-filter:a", atempo_chain,
                "-ar", "16000",
                "-ac", "1",
                str(audio_path),
            ],
            check=True,
            capture_output=True,
        )
        raw_path.unlink()

    except Exception as e:
        # Any probing/stretching failure falls back to the plain
        # extracted audio rather than blocking the whole pipeline.
        print(f"Time-stretch correction skipped due to error: {e}")
        if raw_path.exists() and not audio_path.exists():
            raw_path.rename(audio_path)

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