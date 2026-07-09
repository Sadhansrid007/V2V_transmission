import hashlib
import chromadb
from sentence_transformers import SentenceTransformer

# ----------------------------------
# Load embedding model once
# ----------------------------------
model = SentenceTransformer("BAAI/bge-m3")


def chunk_transcript(
    transcript: dict,
    video_url: str,
    segments_per_window: int = 6,   # 6 sentence-segments = 1 window
    overlap_segments: int = 2,      # consecutive windows share 2 segments
    chroma_path: str = "./chroma_db",
    collection_name: str = "lecture_transcript",
):
    """
    Creates dynamic overlapping chunks from a transcript based on
    SEGMENT COUNT (not word count), generates embeddings, and stores
    them in ChromaDB.

    Windowing rule
    --------------
    - Every window is made up of `segments_per_window` consecutive
      sentence-segments (default 6).
    - Consecutive windows overlap by `overlap_segments` segments
      (default 2), so the window slides forward by
      (segments_per_window - overlap_segments) segments each step.
    - Word/character size of a window is not constrained -- some
      windows will naturally be longer or shorter depending on how
      wordy each sentence-segment is.
    - The final window is handled specially: if it ends up shorter
      than a full window (i.e. the transcript ran out of segments
      before it could be filled), it's merged into the previous
      window instead of being kept as a tiny/overlap-heavy tail.

    Parameters
    ----------
    transcript : dict
        Output from ingestion(). Must contain "segments", a list of
        dicts with sequential integer "id" fields starting at 0.

    video_url : str
        Original video URL. Used as the retrieval key.

    Returns
    -------
    windows : list
        List of chunk dictionaries.
    """
    # ----------------------------------
    # Connect to ChromaDB
    # ----------------------------------
    client = chromadb.PersistentClient(path=chroma_path)

    collection = client.get_or_create_collection(
        name=collection_name
    )

    # ----------------------------------
    # Check if this video already exists
    # ----------------------------------
    existing = collection.get(
        where={"video_url": video_url},
        limit=1
    )

    if len(existing["ids"]) > 0:
        print(f"\nVideo already exists in ChromaDB.")
        print(f"URL: {video_url}")
        print("Skipping chunking and embedding.\n")
        return

    segments = transcript["segments"]

    windows = []
    i = 0
    step = segments_per_window - overlap_segments

    # ----------------------------------
    # Dynamic Segment-Based Chunking
    # ----------------------------------
    while i < len(segments):

        current_segments = segments[i : i + segments_per_window]

        if not current_segments:
            break

        windows.append(
            {
                "chunk_index": len(windows),
                "start_segment": current_segments[0]["id"],
                "end_segment": current_segments[-1]["id"],
                "start_time": current_segments[0]["start"],
                "end_time": current_segments[-1]["end"],
                "segment_count": len(current_segments),
                "word_count": sum(
                    len(s["text"].split()) for s in current_segments
                ),
                "text": " ".join(
                    s["text"].strip() for s in current_segments
                ),
            }
        )

        # End of transcript -- this window already consumed the tail
        if i + segments_per_window >= len(segments):
            break

        i += step

    # ----------------------------------
    # Manage last window
    # ----------------------------------
    # If the final window ended up short (transcript ran out of
    # segments before a full window could be built), it's mostly/
    # entirely overlap with the previous window and adds little new
    # content. Merge it into the previous window instead of keeping
    # it as a separate tiny chunk.
    if len(windows) >= 2:
        last = windows[-1]
        last_segment_count = last["end_segment"] - last["start_segment"] + 1

        if last_segment_count < segments_per_window:
            merged = windows.pop()
            prev = windows[-1]

            start_id = prev["start_segment"]
            end_id = merged["end_segment"]
            # segment "id"s are sequential starting at 0, so they can
            # be used directly as list indices into `segments`.
            merged_segments = segments[start_id : end_id + 1]

            prev["end_segment"] = end_id
            prev["end_time"] = merged_segments[-1]["end"]
            prev["segment_count"] = len(merged_segments)
            prev["word_count"] = sum(
                len(s["text"].split()) for s in merged_segments
            )
            prev["text"] = " ".join(
                s["text"].strip() for s in merged_segments
            )

            print(
                f"Merged final window ({last_segment_count} segment(s)) "
                "into previous window."
            )

    print(f"\nCreated {len(windows)} chunks")

    # ----------------------------------
    # Generate embeddings
    # ----------------------------------
    texts = [w["text"] for w in windows]

    embeddings = model.encode(
        texts,
        batch_size=32,
        normalize_embeddings=True,
        convert_to_numpy=True,
        show_progress_bar=True,
    )

    for idx, emb in enumerate(embeddings):
        windows[idx]["embedding"] = emb.tolist()

    print("Embeddings generated.")

    # ----------------------------------
    # Unique ID for every chunk
    # ----------------------------------
    def make_id(video_url, text, index):
        return hashlib.md5(
            f"{video_url}_{index}_{text}".encode("utf-8")
        ).hexdigest()

    ids = [
        make_id(video_url, w["text"], i)
        for i, w in enumerate(windows)
    ]

    # ----------------------------------
    # Store in ChromaDB
    # ----------------------------------
    collection.add(
        ids=ids,
        documents=[w["text"] for w in windows],
        embeddings=[w["embedding"] for w in windows],
        metadatas=[
            {
                "video_url": video_url,
                "chunk_index": w["chunk_index"],
                "start_time": w["start_time"],
                "end_time": w["end_time"],
                "start_segment": w["start_segment"],
                "end_segment": w["end_segment"],
                "segment_count": w["segment_count"],
                "word_count": w["word_count"],
            }
            for w in windows
        ],
    )

    print(f"Stored {len(windows)} chunks in ChromaDB.")
    print(f"Video URL: {video_url}")

    return windows