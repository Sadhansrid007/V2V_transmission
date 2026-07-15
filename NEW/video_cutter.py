import sys
import os
import imageio.plugins.ffmpeg
from moviepy import VideoFileClip, concatenate_videoclips

def cut_and_stitch(video_path, timestamps, output_path="output_summary.mp4"):
    
    if not timestamps:
        print("No timestamps provided. Cannot cut video.")
        return None
    
    print(f"Loading video from: {video_path}")
    video = VideoFileClip(video_path)
    
    clips = []
    for i, ts in enumerate(timestamps):
        start = ts["start"]
        end = ts["end"]
        
        start = max(0, start)
        end = min(end, video.duration)
        
        if start >= end:
            continue
            
        print(f"Cutting clip {i+1}: {start}s to {end}s")
        clip = video.subclipped(start, end)
        clips.append(clip)
    
    if not clips:
        print("No valid clips found.")
        return None
    
    print("Stitching clips together...")
    final_video = concatenate_videoclips(clips)
    
    print(f"Saving output to: {output_path}")
    final_video.write_videofile(
        output_path,
        codec="libx264",
        audio_codec="aac",
        logger=None
    )
    
    video.close()
    final_video.close()
    for clip in clips:
        clip.close()
    
    print(f"Done. Summary video saved at: {output_path}")
    return output_path


if __name__ == "__main__":
    print("video_cutter.py loaded. Import cut_and_stitch to use.")