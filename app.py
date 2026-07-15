"""
Streamlit app: reads chapter.json (produced by chapters.py) and, for each
topic, cuts the matching segments straight out of the ORIGINAL video_url
with ffmpeg (concatenated into one clip per topic) and burns the topic
name onto the video as a label.

This app is intentionally self-contained -- it only needs chapter.json
and ffmpeg; it does not touch Qdrant/Groq/embeddings at all, since every
piece of information it needs (topic, segment start/end) already lives
in chapter.json.

Run with:
    streamlit run app.py
"""

import hashlib
import json
import re
import shutil
import subprocess
from pathlib import Path

import streamlit as st

# ----------------------------------
# Config
# ----------------------------------
DEFAULT_CHAPTERS_PATH = "chapter.json"
DEFAULT_CLIPS_DIR = "clips"

# Common font locations to try, in order, for burning the topic label
# onto the video. If none exist on this system, the app still works --
# it just skips the burned-in text and relies on the label shown in the
# Streamlit UI instead.
_FONT_CANDIDATES = [
    "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
    "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
    "/Library/Fonts/Arial Bold.ttf",
    "/System/Library/Fonts/Supplemental/Arial Bold.ttf",
    "C:/Windows/Fonts/arialbd.ttf",
]


def _find_font_path() -> str | None:
    for path in _FONT_CANDIDATES:
        if Path(path).exists():
            return path
    return None


def _ffmpeg_available() -> bool:
    return shutil.which("ffmpeg") is not None


# ----------------------------------
# chapter.json loading
# ----------------------------------
def load_chapters_db(path: str) -> dict:
    p = Path(path)
    if not p.exists():
        return {}
    with open(p, "r", encoding="utf-8") as f:
        try:
            return json.load(f)
        except json.JSONDecodeError:
            return {}


def topic_total_duration(segments: list[dict]) -> float:
    return round(sum(s["end"] - s["start"] for s in segments), 2)


def _slugify(text: str) -> str:
    slug = re.sub(r"[^a-zA-Z0-9]+", "_", text).strip("_").lower()
    return slug or "topic"


def _clip_filename(video_url: str, topic: str, segments: list[dict]) -> str:
    """
    Deterministic filename that changes automatically if the underlying
    segments change (e.g. chapter.json got regenerated with different
    boundaries), so stale clips never get served silently.
    """
    video_hash = hashlib.md5(video_url.encode("utf-8")).hexdigest()[:10]
    seg_sig = hashlib.md5(
        json.dumps([[s["start"], s["end"]] for s in segments]).encode("utf-8")
    ).hexdigest()[:8]
    return f"{video_hash}__{_slugify(topic)}__{seg_sig}.mp4"


# ----------------------------------
# ffmpeg: cut every segment for a topic out of video_url and concat
# them into one labeled clip, in a single ffmpeg pass (important when
# video_url is a remote/network source -- avoids re-opening the stream
# once per segment).
# ----------------------------------
def _escape_drawtext(text: str) -> str:
    return text.replace("\\", "\\\\").replace("'", "\\'")


def build_ffmpeg_command(
    video_url: str,
    segments: list[dict],
    output_path: str,
    topic_label: str | None,
    font_path: str | None,
) -> list[str]:
    filter_parts = []
    concat_refs = []

    for i, seg in enumerate(segments):
        start, end = seg["start"], seg["end"]
        filter_parts.append(f"[0:v]trim=start={start}:end={end},setpts=PTS-STARTPTS[v{i}]")
        filter_parts.append(f"[0:a]atrim=start={start}:end={end},asetpts=PTS-STARTPTS[a{i}]")
        concat_refs.append(f"[v{i}][a{i}]")

    n = len(segments)
    filter_parts.append(f"{''.join(concat_refs)}concat=n={n}:v=1:a=1[vcat][acat]")

    video_out = "vcat"
    if topic_label and font_path:
        label = _escape_drawtext(topic_label)
        filter_parts.append(
            f"[vcat]drawtext=fontfile='{font_path}':text='{label}':"
            f"x=24:y=24:fontsize=32:fontcolor=white:box=1:boxcolor=black@0.55:boxborderw=10[vout]"
        )
        video_out = "vout"

    filter_complex = ";".join(filter_parts)

    return [
        "ffmpeg", "-y",
        "-reconnect", "1",
        "-reconnect_streamed", "1",
        "-reconnect_delay_max", "5",
        "-i", video_url,
        "-filter_complex", filter_complex,
        "-map", f"[{video_out}]",
        "-map", "[acat]",
        "-c:v", "libx264", "-preset", "veryfast", "-crf", "20",
        "-c:a", "aac", "-b:a", "128k",
        output_path,
    ]


def cut_topic_clip(
    video_url: str,
    topic: str,
    segments: list[dict],
    clips_dir: str,
    burn_label: bool,
    font_path: str | None,
    force: bool = False,
) -> tuple[Path, str | None]:
    """
    Returns (output_path, error_message). error_message is None on
    success. Skips re-running ffmpeg if the deterministic output file
    already exists and force=False.
    """
    Path(clips_dir).mkdir(parents=True, exist_ok=True)
    output_path = Path(clips_dir) / _clip_filename(video_url, topic, segments)

    if output_path.exists() and not force:
        return output_path, None

    cmd = build_ffmpeg_command(
        video_url=video_url,
        segments=segments,
        output_path=str(output_path),
        topic_label=topic if burn_label else None,
        font_path=font_path if burn_label else None,
    )

    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        return output_path, result.stderr[-2000:]  # tail of stderr is usually the useful part

    return output_path, None


# ----------------------------------
# Streamlit UI
# ----------------------------------
st.set_page_config(page_title="Lecture Chapter Clips", layout="wide")
st.title("Lecture Chapter Clips")
st.caption("Cuts labeled clips straight from the source video, one per topic, using chapter.json.")

if not _ffmpeg_available():
    st.error("ffmpeg was not found on this system's PATH. Install it before using this app.")
    st.stop()

with st.sidebar:
    st.header("Settings")
    chapters_path = st.text_input("chapter.json path", value=DEFAULT_CHAPTERS_PATH)
    clips_dir = st.text_input("Output folder for clips", value=DEFAULT_CLIPS_DIR)
    burn_label = st.checkbox("Burn topic label onto the video", value=True)

    detected_font = _find_font_path()
    if burn_label and not detected_font:
        st.warning("No system font found for burning text -- label will only show in this app's UI, not on the video.")
    font_override = st.text_input(
        "Font file path (optional override)",
        value=detected_font or "",
        help="Leave as-is unless the label isn't rendering and you know a valid .ttf path.",
    )
    force_regenerate = st.checkbox("Force regenerate (ignore cached clips)", value=False)

db = load_chapters_db(chapters_path)

if not db:
    st.info(f"No data found at '{chapters_path}'. Run chapters.py first to generate it.")
    st.stop()

video_urls = list(db.keys())
selected_video = st.selectbox("Video", video_urls)

topics = db.get(selected_video, [])
if not topics:
    st.warning("This video has no topics in chapter.json.")
    st.stop()

st.subheader(f"{len(topics)} topic(s) found")

summary_rows = [
    {
        "Topic": t["topic"],
        "Segments": len(t["segments"]),
        "Total duration (s)": topic_total_duration(t["segments"]),
    }
    for t in topics
]
st.dataframe(summary_rows, use_container_width=True, hide_index=True)

font_path = font_override.strip() or None

# ----------------------------------
# Generate all
# ----------------------------------
if st.button("Generate clips for every topic"):
    progress = st.progress(0.0, text="Starting...")
    for i, t in enumerate(topics):
        progress.progress(i / len(topics), text=f"Cutting: {t['topic']}")
        path, err = cut_topic_clip(
            video_url=selected_video,
            topic=t["topic"],
            segments=t["segments"],
            clips_dir=clips_dir,
            burn_label=burn_label,
            font_path=font_path,
            force=force_regenerate,
        )
        if err:
            st.error(f"Failed on '{t['topic']}':\n{err}")
    progress.progress(1.0, text="Done.")
    st.success("Finished generating all topic clips.")

st.divider()

# ----------------------------------
# Per-topic browse + generate + preview
# ----------------------------------
for t in topics:
    topic = t["topic"]
    segments = t["segments"]

    with st.expander(f"{topic}  ·  {len(segments)} segment(s)  ·  {topic_total_duration(segments)}s total"):
        st.write(
            ", ".join(f"[{s['start']}s → {s['end']}s]" for s in segments)
        )

        existing_path = Path(clips_dir) / _clip_filename(selected_video, topic, segments)
        already_exists = existing_path.exists() and not force_regenerate

        if st.button(
            "Play / Generate clip" if not already_exists else "Regenerate & play",
            key=f"gen_{topic}",
        ):
            with st.spinner(f"Cutting '{topic}'..."):
                path, err = cut_topic_clip(
                    video_url=selected_video,
                    topic=topic,
                    segments=segments,
                    clips_dir=clips_dir,
                    burn_label=burn_label,
                    font_path=font_path,
                    force=force_regenerate,
                )
            if err:
                st.error(f"ffmpeg failed:\n{err}")
            else:
                st.session_state[f"clip_{topic}"] = str(path)

        clip_path = st.session_state.get(f"clip_{topic}")
        if clip_path and Path(clip_path).exists():
            st.video(clip_path)
            with open(clip_path, "rb") as f:
                st.download_button(
                    "Download this clip",
                    data=f.read(),
                    file_name=Path(clip_path).name,
                    mime="video/mp4",
                    key=f"dl_{topic}",
                )
