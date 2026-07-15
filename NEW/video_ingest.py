import os
import subprocess
import whisper
import requests
import shutil

def clear_vid_process():
    vid_process_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "vid_process"
    )
    if os.path.exists(vid_process_path):
        shutil.rmtree(vid_process_path)
        print("Cleared vid_process folder")

def get_vid_process_dir():
    vid_process_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "vid_process"
    )
    os.makedirs(vid_process_path, exist_ok=True)
    return vid_process_path

def download_video(url, output_path):
    print(f"Downloading video from: {url}")
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
    }
    response = requests.get(url, headers=headers, stream=True)
    response.raise_for_status()
    
    with open(output_path, "wb") as f:
        for chunk in response.iter_content(chunk_size=8192):
            if chunk:
                f.write(chunk)
    
    print(f"Video downloaded to: {output_path}")
    return output_path

def extract_audio(video_path, audio_output_path):
    print("Extracting audio from video...")
    command = [
        "ffmpeg", "-i", video_path,
        "-vn", "-ac", "1", "-ar", "16000",
        audio_output_path, "-y"
    ]
    subprocess.run(command, capture_output=True)
    print(f"Audio saved to: {audio_output_path}")
    return audio_output_path

def transcribe_audio(audio_path, transcript_output_path):
    print("Loading Whisper model...")
    model = whisper.load_model("tiny")
    
    print("Transcribing audio...")
    result = model.transcribe(audio_path, fp16=False)
    
    print("Saving transcript...")
    with open(transcript_output_path, "w", encoding="utf-8") as f:
        for segment in result["segments"]:
            start = round(segment["start"], 2)
            end = round(segment["end"], 2)
            text = segment["text"].strip()
            f.write(f"{start}\t{end}\t{text}\n")
    
    print(f"Transcript saved to: {transcript_output_path}")
    return transcript_output_path

def process_video(video_path=None, video_url=None):
    vid_process_dir = get_vid_process_dir()
    
    if video_url:
        video_path = os.path.join(vid_process_dir, "downloaded_video.mp4")
        download_video(video_url, video_path)
    
    if not video_path:
        raise ValueError("Either video_path or video_url must be provided")
    
    audio_path = os.path.join(vid_process_dir, "extracted_audio.wav")
    transcript_path = os.path.join(vid_process_dir, "audio_transcript.txt")
    
    extract_audio(video_path, audio_path)
    transcribe_audio(audio_path, transcript_path)
    
    print(f"All files saved in: {vid_process_dir}")
    return transcript_path


if __name__ == "__main__":
    VIDEO_PATH = r"C:\Users\Madhu\OneDrive\Desktop\CLG\techNOVA\video_ingestion\sample vid.mp4"
    transcript = process_video(video_path=VIDEO_PATH)
    print(f"Transcript: {transcript}")