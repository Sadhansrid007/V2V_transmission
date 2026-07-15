import sys
import os
import streamlit as st
import tempfile
import shutil
import atexit

sys.path.append(os.path.dirname(__file__))

from rag_backend import store_transcript, process_query
from video_cutter import cut_and_stitch
from video_ingest import process_video, clear_vid_process

def cleanup_on_exit():
    vid_process_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "vid_process"
    )
    if os.path.exists(vid_process_path):
        shutil.rmtree(vid_process_path)
        print("Auto cleaned vid_process on exit")

atexit.register(cleanup_on_exit)

st.set_page_config(
    page_title="CLIPBOT",
    page_icon="🎬",
    layout="wide"
)

st.title("🎬 CLIPBOT")
st.subheader("Ask anything about your lecture video")

if "transcript_stored" not in st.session_state:
    st.session_state.transcript_stored = False
if "video_path" not in st.session_state:
    st.session_state.video_path = None
if "input_mode" not in st.session_state:
    st.session_state.input_mode = "upload"
if "search_result" not in st.session_state:
    st.session_state.search_result = None
if "final_video_bytes" not in st.session_state:
    st.session_state.final_video_bytes = None

with st.sidebar:
    st.header("Input Video")
    
    input_mode = st.radio(
        "Choose input method",
        ["Upload Video File", "Paste Video Link"]
    )
    
    if input_mode == "Upload Video File":
        video_file = st.file_uploader(
            "Upload your video",
            type=["mp4", "avi", "mov"]
        )
        
        if video_file:
            if st.button("Process Video"):
                clear_vid_process()
                st.session_state.search_result = None
                st.session_state.final_video_bytes = None
                st.session_state.transcript_stored = False
                
                progress = st.empty()
                
                progress.info("Step 1 🟢 — Saving video...")
                video_temp = tempfile.NamedTemporaryFile(
                    delete=False,
                    suffix=".mp4"
                )
                video_temp.write(video_file.read())
                video_temp.close()
                st.session_state.video_path = video_temp.name
                
                progress.info("Step 2 🟢 — Extracting audio and transcribing...")
                transcript_path = process_video(
                    video_path=st.session_state.video_path
                )
                
                progress.info("Step 3 🟢 — Building search database...")
                store_transcript(transcript_path)
                st.session_state.transcript_stored = True
                
                progress.empty()
                st.success("Video processed. Ask your question below.")
    
    else:
        video_url = st.text_input(
            "Paste video link here",
            placeholder="https://example.com/video.mp4"
        )
        
        if video_url:
            if st.button("Process Video"):
                clear_vid_process()
                st.session_state.search_result = None
                st.session_state.final_video_bytes = None
                st.session_state.transcript_stored = False
                
                progress = st.empty()
                
                try:
                    progress.info("Step 1 🟢 — Downloading video...")
                    
                    vid_process_dir = os.path.join(
                        os.path.dirname(os.path.abspath(__file__)),
                        "vid_process"
                    )
                    os.makedirs(vid_process_dir, exist_ok=True)
                    
                    transcript_path = process_video(
                        video_url=video_url
                    )
                    
                    st.session_state.video_path = os.path.join(
                        vid_process_dir, "downloaded_video.mp4"
                    )
                    
                    progress.info("Step 2 🟢 — Transcribing audio...")
                    progress.info("Step 3 🟢 — Building search database...")
                    store_transcript(transcript_path)
                    st.session_state.transcript_stored = True
                    
                    progress.empty()
                    st.success("Video processed. Ask your question below.")
                
                except Exception as e:
                    st.error(f"Failed to process video link: {str(e)}")
                    progress.empty()

if not st.session_state.transcript_stored:
    st.info("Upload a video or paste a link on the left to get started.")

else:
    st.success("Video loaded and ready.")
    
    student_prompt = st.text_input(
        "What do you want to find in the video?",
        placeholder="e.g. show me the part where X topic is discussed"
    )
    
    if st.button("Search") and student_prompt:
        with st.spinner("Searching through video content..."):
            result = process_query(
                student_prompt=student_prompt,
                transcript_path=None
            )
        st.session_state.search_result = result
        st.session_state.final_video_bytes = None
    
    if st.session_state.search_result:
        result = st.session_state.search_result
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### Quick Summary")
            st.write(
                result["short_summary"] if result["short_summary"]
                else result["answer"]
            )
        
        with col2:
            st.markdown("### Detailed Description")
            st.write(
                result["detailed_description"] if result["detailed_description"]
                else ""
            )
        
        if result["timestamps"]:
            
            if st.session_state.final_video_bytes is None:
                if st.button("Generate Clip"):
                    with st.spinner("Cutting and stitching video clips..."):
                        output_path = tempfile.mktemp(suffix=".mp4")
                        final_video = cut_and_stitch(
                            video_path=st.session_state.video_path,
                            timestamps=result["timestamps"],
                            output_path=output_path
                        )
                    if final_video:
                        with open(final_video, "rb") as f:
                            st.session_state.final_video_bytes = f.read()
            
            if st.session_state.final_video_bytes:
                st.markdown("### Your Summary Clip")
                st.video(st.session_state.final_video_bytes)
                st.download_button(
                    label="⬇️ Download Summary Video",
                    data=st.session_state.final_video_bytes,
                    file_name="summary.mp4",
                    mime="video/mp4"
                )
        
        else:
            st.warning(
                "This topic was not found in the video. "
                "Please try rephrasing your question."
            )