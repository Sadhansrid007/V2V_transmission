"""
Connects the whole pipeline:

    ingestion.py   video_url -> transcript (segments, text, silences)
    chunking.py    transcript -> stored + embedded in Qdrant (the only
                   place anything gets persisted -- ingestion.py itself
                   saves nothing)
    chapters.py    Qdrant chunks + subject_code's topic list -> chapter.json

topics.py is intentionally NOT called from here -- it's a standalone,
manually-run seeding script (see topics.py's __main__ block) since it
isn't part of the per-video flow and needs a subject's topic list to
already exist before generate_chapters() can run for that subject.

Note on Qdrant (local/embedded mode): only one open client can hold a
given on-disk path at a time. Each function below (chunk_transcript,
generate_chapters) opens and releases its own QdrantClient scoped to
that single call, so running these sequentially -- as this script does
-- is safe. Don't call them concurrently from multiple
threads/processes against the same qdrant_path.
"""

from chapters import generate_chapters
from chunking import chunk_transcript
from ingestion import ingestion


def run_pipeline(video_url: str, subject_code: str, force: bool = False):
    """
    Runs the full pipeline for one video and returns its final
    chapter.json entry: [{"topic": ..., "segments": [...]}, ...].

    Parameters
    ----------
    video_url : str
        The video to process. Used as the key throughout Qdrant and
        chapter.json / topics_review.json.
    subject_code : str
        Selects which allowed topic list (seeded via topics.py) chapters.py
        is allowed to choose from for this video. Must already exist --
        run topics.py first if it doesn't.
    force : bool
        If True, re-runs chunking (re-embed/overwrite) and chapter
        generation even if this video_url was already processed before.
        Note: ingestion() itself always re-transcribes -- it has no
        notion of "already done" since it doesn't persist anything on
        its own.
    """
    print(f"\n=== 1/3 ingestion: {video_url} ===")
    transcript = ingestion(video_url=video_url)
    print(
        f"Ingestion done: {len(transcript['segments'])} segment(s), "
        f"{len(transcript['silences'])} silence range(s), "
        f"{transcript['max_duration']}s total."
    )

    print(f"\n=== 2/3 chunking + Qdrant storage: {video_url} ===")
    stored_segments = chunk_transcript(transcript, video_url=video_url, force=force)
    if stored_segments is None:
        print("Chunking skipped (video already stored) -- reusing existing Qdrant data.")
    else:
        print(f"Stored {len(stored_segments)} chunk(s) in Qdrant.")

    print(f"\n=== 3/3 chapters: {video_url} (subject_code={subject_code}) ===")
    chapters = generate_chapters(video_url=video_url, subject_code=subject_code, force=force)
    print(f"Done. {len(chapters)} topic(s) written to chapter.json for {video_url}.")

    return chapters


if __name__ == "__main__":
    video_url = input("Enter video URL: ").strip()
    subject_code = input("Enter subject code: ").strip()
    run_pipeline(video_url, subject_code)
