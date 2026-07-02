import os
# In MoviePy v2, we import directly from 'moviepy' instead of 'moviepy.editor'
from moviepy import VideoFileClip, concatenate_videoclips

def cut_video_segments(target_segments, input_video="source_video.mp4", output_video="output_summary.mp4"):
    """
    Dynamically slices and concatenates video segments based on incoming timestamps.
    
    :param target_segments: List of time pairs from the AI, e.g., [[16, 18], [326, 330]]
    :param input_video: Path to the original full-length video file
    :param output_video: Target path for the generated summary clip
    :return: Path to the output video string if successful, None otherwise
    """
    # Safety net: if the AI found nothing, don't waste time running MoviePy
    if not target_segments:
        print("⚠️ No timestamps provided to the video cutter. Skipping process.")
        return None
        
    if not os.path.exists(input_video):
        print(f"❌ Error: Could not find original video file at '{input_video}'.")
        return None

    print(f"🎬 Opening {input_video}...")
    source = VideoFileClip(input_video)
    
    clips = []
    try:
        for i, (start, end) in enumerate(target_segments):
            print(f"✂️ Slicing segment {i+1}: from {start}s to {end}s...")
            # MoviePy v2 renamed '.subclip()' to '.subclipped()'
            sub_clip = source.subclipped(start, end)
            clips.append(sub_clip)
        
        print("🧵 Stitching all dynamic clips together...")
        final_summary = concatenate_videoclips(clips)
        
        print("💾 Exporting final summary video file...")
        final_summary.write_videofile(output_video, fps=24, codec="libx264", audio_codec="aac")
        print(f"🎉 SUCCESS! Created: '{output_video}'")
        return output_video
        
    except Exception as e:
        print(f"❌ Something went wrong while cutting the video: {e}")
        return None
        
    finally:
        print("🧹 Cleaning up system memory...")
        source.close()
        if 'final_summary' in locals():
            final_summary.close()
        for clip in clips:
            clip.close()

# This allows you to still test it independently if you run it directly!
if __name__ == "__main__":
    test_run_timestamps = [[15, 25], [35, 50]]
    cut_video_segments(test_run_timestamps)