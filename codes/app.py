"""
app.py
------
LectureLens Streamlit UI.

Flow: upload/link a lecture -> ingest + transcribe + store in ChromaDB
(once) -> ask as many questions as you want in a chat interface, each
getting a short summary, detailed description, and an optional
generated clip.
"""

import streamlit as st

from ingest import ingest_url, ingest_upload
from transcribe import transcribe
from rag_backend import store_transcript, process_query
from video_cutter import cut_and_stitch
from config import DATA_DIR

st.set_page_config(page_title="LectureLens", page_icon="🎓", layout="wide")
st.title("🎓 LectureLens")
st.caption("Ask anything about your lecture. Get an answer, a summary, and a clipped video.")

# --- session state ---------------------------------------------------
if "video_ready" not in st.session_state:
    st.session_state.video_ready = False
if "video_id" not in st.session_state:
    st.session_state.video_id = None
if "video_path" not in st.session_state:
    st.session_state.video_path = None
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []  # list of {"query", "result", "clip_bytes"}

# --- sidebar: video input ---------------------------------------------
with st.sidebar:
    st.header("1. Load a lecture")
    input_mode = st.radio("Input method", ["Upload file", "Paste link"], label_visibility="collapsed")

    def _process(video_id, video_path, audio_path):
        with st.status("Processing lecture...", expanded=True) as status:
            status.write("Transcribing audio (Groq Whisper)...")
            transcript = transcribe(audio_path, video_id)
            status.write(f"Got {len(transcript['segments'])} transcript segments.")

            status.write("Building searchable index...")
            n_chunks = store_transcript(video_id)
            status.write(f"Indexed {n_chunks} chunks.")

            status.update(label="Ready! Ask a question below.", state="complete")

        st.session_state.video_ready = True
        st.session_state.video_id = video_id
        st.session_state.video_path = video_path
        st.session_state.chat_history = []

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

# --- main: chat interface ----------------------------------------------
if not st.session_state.video_ready:
    st.info("Load a lecture from the sidebar to start asking questions.")
else:
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