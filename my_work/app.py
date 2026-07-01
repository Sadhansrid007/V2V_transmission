import streamlit as st
from ai_bridge import get_ai_timestamps
from video_cutter import cut_video_segments

st.set_page_config(page_title="AI Video Summarizer", page_icon="🎥", layout="centered")

st.title("🎥 Smart Video Segment Search")
st.write("Type what you're looking for, and the local AI will cut the video to matching segments.")

# Chat input box
user_query = st.text_input("Ask the AI about the video:", placeholder="e.g., give me the summary of the video")

if st.button("🎬 Generate Video Clip") and user_query:
    with st.spinner("🧠 Searching vector database and asking local Llama..."):
        # 1. Get timestamps dynamically from your bridge script
        timestamps = get_ai_timestamps(user_query)
        
    if timestamps:
        with st.spinner("✂️ MoviePy is cutting and stitching your video segments..."):
            # 2. Pass those timestamps dynamically to your cutter script
            output_clip = cut_video_segments(timestamps)
            
            if output_clip:
                st.success("🎉 Your custom video summary clip is ready!")
                # 3. Render the video player directly in the UI
                st.video(output_clip)
            else:
                st.error("❌ The video cutter encountered an error processing the clips.")
    else:
        st.warning("🔍 The AI couldn't isolate any specific timestamp brackets for that query.")