"""
app.py
------
LectureLens Streamlit UI.

Flow: upload/link a lecture -> ingest + transcribe + store in ChromaDB
(once) -> then use either tab as many times as you want:
  - Ask:       chat interface, each turn gets a short summary, detailed
               description, and an optional generated clip.
  - Summarize: condenses the whole lecture into one "core content
               only" video within a target duration budget.

CHANGES vs the previous version:
  1. Removed the "Remove Silence" tab. silence_remover.py itself is
     untouched and still importable/usable directly if needed later --
     it's just not exposed in the UI right now.
  2. Removed all stats/status text from the Summarize tab (segment
     counts, before/after duration metrics, "batches failed" and
     "shorter than usual" messages). The tab now shows only: a button,
     a plain spinner while it runs, then the finished video and a
     download button -- nothing else.
"""

import streamlit as st

from ingest import ingest_url, ingest_upload
from transcribe import transcribe
from rag_backend import store_transcript, process_query
from video_cutter import cut_and_stitch
from summarizer import generate_summary_video
from config import DATA_DIR

st.set_page_config(page_title="LectureLens", page_icon="🎓", layout="wide")
st.title("🎓 LectureLens")
st.caption("Ask anything about your lecture, or get a condensed summary.")

# --- session state ---------------------------------------------------
defaults = {
    "video_ready": False,
    "video_id": None,
    "video_path": None,
    "audio_path": None,
    "chat_history": [],       # list of {"query", "result", "clip_bytes"}
    "summary_video_bytes": None,
}
for key, value in defaults.items():
    if key not in st.session_state:
        st.session_state[key] = value


# --- sidebar: video input ---------------------------------------------
with st.sidebar:
    st.header("1. Load a lecture")
    input_mode = st.radio("Input method", ["Upload file", "Paste link"], label_visibility="collapsed")

    def _process(video_id, video_path, audio_path):
        with st.spinner("Processing lecture... this can take a few minutes."):
            transcribe(audio_path, video_id)
            store_transcript(video_id)

        st.session_state.video_ready = True
        st.session_state.video_id = video_id
        st.session_state.video_path = video_path
        st.session_state.audio_path = audio_path
        st.session_state.chat_history = []
        st.session_state.summary_video_bytes = None

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
        # A new question gets answered and appended to history immediately.
        # We do NOT wait for "Generate clip" here, because clicking any
        # button triggers a full rerun of this script -- if the clip logic
        # lived inside this `if prompt := ...` block, it would vanish on
        # the very next rerun (prompt is empty again after the click).
        if prompt := st.chat_input("Ask about the lecture..."):
            with st.spinner("Searching the lecture..."):
                result = process_query(st.session_state.video_id, prompt)
            st.session_state.chat_history.append({
                "query": prompt, "result": result, "clip_bytes": None,
            })

        # Render every turn from session_state, which survives reruns --
        # this is what makes the clip button work correctly no matter how
        # many times it (or any other button) triggers a rerun.
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
                                output_path = DATA_DIR / f"clip_output_{i}.mp4"
                                clip_path = cut_and_stitch(
                                    st.session_state.video_path, result["timestamps"], output_path
                                )
                            if clip_path:
                                turn["clip_bytes"] = clip_path.read_bytes()
                                st.rerun()  # redraw immediately so the video shows without a second click
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
    # TAB 2: Summarize -- calls summarizer.generate_summary_video, which
    # internally calls video_cutter.cut_and_stitch on the kept segments.
    # No stats or status detail shown -- just the result.
    # ===================================================================
    with tab_summary:
        st.write(
            "Condenses the full lecture down to definitions and main topic "
            "explanations only -- examples, tangents, and admin talk are cut."
        )

        if st.button("Generate summary video", type="primary", key="gen_summary"):
            with st.spinner("Building summary... this can take a few minutes for long lectures."):
                output_path = DATA_DIR / f"summary_{st.session_state.video_id}.mp4"
                result_path, stats = generate_summary_video(
                    st.session_state.video_id, st.session_state.video_path, output_path
                )

            if result_path is None:
                st.error("Couldn't generate a summary for this lecture.")
            else:
                st.session_state.summary_video_bytes = result_path.read_bytes()

        if st.session_state.summary_video_bytes:
            st.video(st.session_state.summary_video_bytes)
            st.download_button(
                "⬇️ Download summary", data=st.session_state.summary_video_bytes,
                file_name=f"summary_{st.session_state.video_id}.mp4", mime="video/mp4",
            )