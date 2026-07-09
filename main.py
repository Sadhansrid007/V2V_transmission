from new_ingestion import ingestion
from new_chunking import chunk_transcript
from chapters import generate_chapters


import json
from pathlib import Path



BASE_DIR = Path(__file__).parent
CHAPTER_DB = BASE_DIR / "chapters.json"


def chapter_exists(video_url: str):
    """
    Returns existing chapter data if already generated.
    """

    if not CHAPTER_DB.exists():
        return None

    try:
        with open(CHAPTER_DB, "r", encoding="utf-8") as f:
            db = json.load(f)
    except json.JSONDecodeError:
        return None

    return db.get(video_url)


def process_video(video_url: str):
    """
    Complete processing pipeline.
    """

    # -----------------------------------------
    # Step 0 : Skip if already processed
    # -----------------------------------------

    existing = chapter_exists(video_url)

    if existing:
        print("\nVideo already processed.")
        print("Returning stored chapters...\n")
        return existing

    # -----------------------------------------
    # Step 1 : Transcription
    # -----------------------------------------

    print("\n[1/3] Transcribing...")

    transcript = ingestion(
        video_url=video_url,
        keep_audio=False
    )

    print("Done.")

    # -----------------------------------------
    # Step 2 : Chunking + Embedding
    # -----------------------------------------

    print("\n[2/3] Creating embeddings...")

    chunk_transcript(
        transcript=transcript,
        video_url=video_url
    )

    print("Done.")

    # -----------------------------------------
    # Step 3 : Chapter Generation
    # -----------------------------------------

    print("\n[3/3] Generating chapters...")

    chapters = generate_chapters(
        video_url=video_url,
        silences=transcript["silences"]
    )

    print("Done.")

    return chapters


def display_chapters(chapters):
    """
    Nicely print generated chapters.
    """

    print("\n" + "=" * 70)
    print("Generated Chapters")
    print("=" * 70)

    for chapter in chapters["chapters"]:

        print(
            f"[{chapter['segment_id']:02}] "
            f"{chapter['topic']:<30}"
            f"{chapter['start_time']:>8.2f}s"
            f"  ->  "
            f"{chapter['end_time']:.2f}s"
        )


if __name__ == "__main__":

    video_url = input("Enter Video URL : ").strip()

    result = process_video(video_url)

    display_chapters(result)

    print("\nPipeline Completed Successfully.")