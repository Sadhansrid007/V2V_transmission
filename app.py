"""
app.py
------
LectureLens Streamlit UI.

CHANGE vs the previous version: the Summarize tab used to discard the
stats dict entirely (`result_path, stats = generate_summary_video(...)`
then only `result_path` got used). Stats are now captured and shown in
a collapsible "Debug info" section -- windows_failed, segments_uncovered,
topics_found, topics_scoring_failed are exactly the numbers needed to
tell whether a bad summary is a topic-detection failure, an assignment
failure, or a scoring failure, without digging through the terminal.
"""

import tempfile
from pathlib import Path

import streamlit as st

from ingest import ingest_url, ingest_upload, list_known_videos, get_video_path, get_audio_path
from transcribe import transcribe
from rag_backend import store_transcript, process_query, collection_has_data
from video_cutter import cut_and_stitch
from summarizer import generate_summary_video
from config import TRANSCRIPT_DIR

st.set_page_config(page_title="LectureLens", page_icon="🌐", layout="wide")
st.title("🌐 LectureLens")
st.caption("Ask anything about your lecture, or get a condensed summary.")

# --- session state ---------------------------------------------------
defaults = {
    "video_ready": False,
    "video_id": None,
    "video_path": None,
    "audio_path": None,
    "chat_history": [],
    "summary_video_bytes": None,
    "summary_stats": None,
}
for key, value in defaults.items():
    if key not in st.session_state:
        st.session_state[key] = value


def _generate_and_cleanup(build_fn):
    """
    Runs build_fn(output_path) -> Path|None inside a throwaway temp
    directory, reads the result into memory, then lets the whole temp
    directory get deleted on context exit.
    """
    with tempfile.TemporaryDirectory(prefix="lecturelens_out_") as tmp_dir:
        output_path = Path(tmp_dir) / "output.mp4"
        result_path = build_fn(output_path)
        if result_path is None:
            return None
        return Path(result_path).read_bytes()


# --- sidebar: video input ---------------------------------------------
with st.sidebar:
    st.header("1. Load a lecture")

    def _process(video_id, video_path, audio_path):
        already_indexed = (TRANSCRIPT_DIR / f"{video_id}.json").exists() and collection_has_data(video_id)

        with st.spinner("Loading lecture..." if already_indexed else "Processing lecture... this can take a few minutes."):
            transcribe(audio_path, video_id)
            if not already_indexed:
                store_transcript(video_id)

        st.session_state.video_ready = True
        st.session_state.video_id = video_id
        st.session_state.video_path = video_path
        st.session_state.audio_path = audio_path
        st.session_state.chat_history = []
        st.session_state.summary_video_bytes = None
        st.session_state.summary_stats = None

    known_videos = [
        v for v in list_known_videos()
        if (TRANSCRIPT_DIR / f"{v['video_id']}.json").exists()
    ]
    if known_videos:
        placeholder = "— Select a previous lecture —"
        options = {placeholder: None}
        options.update({f"{v['label']} ({v['video_id']})": v["video_id"] for v in known_videos})
        chosen_label = st.selectbox("Previously processed lectures", list(options.keys()))
        chosen_video_id = options[chosen_label]
        if chosen_video_id and st.button("Load selected lecture"):
            _process(chosen_video_id, get_video_path(chosen_video_id), get_audio_path(chosen_video_id))
        st.divider()

    input_mode = st.radio("Or load a new one", ["Upload file", "Paste link"], label_visibility="visible")

    if input_mode == "Upload file":
        uploaded = st.file_uploader("Video file", type=["mp4", "mov", "avi", "mkv"])
        if uploaded and st.button("Process video", type="primary"):
            result = ingest_upload(uploaded.read(), uploaded.name)
            _process(result["video_id"], result["video_path"], result["audio_path"])

    else:
        url = st.text_input("Video URL", placeholder="YouTube, Google Drive, or direct .mp4 link")
        if url and st.button("Process video", type="primary"):
            try:
                result = ingest_url(url)
                _process(result["video_id"], result["video_path"], result["audio_path"])
            except Exception as e:
                st.error(f"Couldn't process that link: {e}")

    if st.session_state.video_ready:
        st.success(f"Lecture loaded: `{st.session_state.video_id}`")


# --- main ---------------------------------------------------------------
if not st.session_state.video_ready:
    st.info("Load a lecture from the sidebar to get started.")
else:
    tab_ask, tab_summary = st.tabs(["💬 Ask", "📝 Summarize"])

    # ===================================================================
    # TAB 1: Ask
    # ===================================================================
    with tab_ask:
        if prompt := st.chat_input("Ask about the lecture..."):
            with st.spinner("Searching the lecture..."):
                result = process_query(st.session_state.video_id, prompt)
            st.session_state.chat_history.append({
                "query": prompt, "result": result, "clip_bytes": None,
            })

        chat_container = st.container(height=520)
        with chat_container:
            for i, turn in enumerate(st.session_state.chat_history):
                with st.chat_message("user"):
                    st.write(turn["query"])

                with st.chat_message("assistant"):
                    result = turn["result"]
                    st.markdown(f"**{result['short_summary'] or 'No summary available.'}**")
                    if result["detailed_description"]:
                        with st.expander("More detail"):
                            st.write(result["detailed_description"])

                    if result["timestamps"]:
                        if turn["clip_bytes"] is None:
                            if st.button("🎬 Generate clip", key=f"gen_clip_{i}"):
                                with st.spinner("Cutting and stitching the clip..."):
                                    clip_bytes = _generate_and_cleanup(
                                        lambda out_path: cut_and_stitch(
                                            st.session_state.video_path, result["timestamps"], out_path
                                        )
                                    )
                                if clip_bytes:
                                    turn["clip_bytes"] = clip_bytes
                                    st.rerun()
                        else:
                            st.video(turn["clip_bytes"])
                            st.download_button(
                                "⬇️ Download clip", data=turn["clip_bytes"],
                                file_name=f"clip_{i}.mp4", mime="video/mp4",
                                key=f"dl_{i}",
                            )
                    else:
                        st.warning("This topic wasn't found in the lecture. Try rephrasing.")

    # ===================================================================
    # TAB 2: Summarize
    # ===================================================================
    with tab_summary:
        st.write(
            "Condenses the full lecture down to definitions and main topic "
            "explanations only -- examples, tangents, and admin talk are cut."
        )

        if st.button("Generate summary video", type="primary", key="gen_summary"):
            captured_stats = {}

            def _build(out_path):
                result_path, stats = generate_summary_video(
                    st.session_state.video_id, st.session_state.video_path, out_path
                )
                captured_stats.update(stats)
                return result_path

            with st.spinner("Building summary... this can take a few minutes for long lectures."):
                summary_bytes = _generate_and_cleanup(_build)

            st.session_state.summary_stats = captured_stats

            if summary_bytes is None:
                st.error(f"Couldn't generate a summary for this lecture: {captured_stats.get('error', 'unknown error')}")
            else:
                st.session_state.summary_video_bytes = summary_bytes

        if st.session_state.summary_video_bytes:
            st.video(st.session_state.summary_video_bytes)
            st.download_button(
                "⬇️ Download summary", data=st.session_state.summary_video_bytes,
                file_name=f"summary_{st.session_state.video_id}.mp4", mime="video/mp4",
            )

        if st.session_state.summary_stats:
            with st.expander("🔧 Debug info"):
                st.json(st.session_state.summary_stats)