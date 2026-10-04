"""
Step 3+4 of the pipeline: load & validate transcript.json, generate
segment embeddings, and store/reuse them in a persistent ChromaDB
collection.

Reused by run_segmentation.py. Does not modify transcript.json.
"""
import json

import numpy as np

import config


def load_transcript(video_url: str, path: str = config.TRANSCRIPT_PATH) -> list[dict]:
    """
    Loads this video's entry from transcript.json (a shared file keyed by
    config.url_key(video_url), see config.py) and returns a validated,
    chronologically sorted list of segment dicts, each with exactly the
    keys: id, start, end, duration, text.

    Malformed segments (missing id, empty text, invalid/inverted
    timestamps) are skipped with a warning rather than raising, so one
    bad segment doesn't take down the whole pipeline.
    """
    data = config.load_url_keyed_entry(path, video_url)
    if data is None:
        raise FileNotFoundError(
            f"No transcript found for this video_url in '{path}' "
            f"(looked up under key '{config.url_key(video_url)}'). Run the "
            f"ingestion stage for this video_url first."
        )

    raw_segments = data.get("segments") if isinstance(data, dict) else None
    if raw_segments is None:
        raise ValueError(
            f"This video's entry in {path} does not have the expected "
            f"{{'segments': [...]}} structure."
        )

    segments = []
    skipped = 0

    for raw in raw_segments:
        seg_id = raw.get("id")
        text = (raw.get("text") or "").strip()
        start = raw.get("start")
        end = raw.get("end")
        duration = raw.get("duration")

        if seg_id is None:
            skipped += 1
            continue
        if not text:
            skipped += 1
            continue
        if not isinstance(start, (int, float)) or not isinstance(end, (int, float)):
            skipped += 1
            continue
        if end < start:
            skipped += 1
            continue
        if duration is None:
            duration = round(end - start, 2)

        segments.append(
            {
                "id": int(seg_id),
                "start": float(start),
                "end": float(end),
                "duration": float(duration),
                "text": text,
            }
        )

    if skipped:
        print(f"Skipped {skipped} malformed/empty segment(s) while loading transcript.")

    if not segments:
        raise ValueError("No valid segments found in transcript.json after validation.")

    segments.sort(key=lambda s: s["start"])
    return segments


_model = None  # loaded lazily, once


def get_embedding_model():
    """
    Lazily loads and caches the SentenceTransformer model named by
    config.EMBEDDING_MODEL, so it's only loaded once per process
    regardless of how many times this is called.
    """
    global _model
    if _model is None:
        from sentence_transformers import SentenceTransformer
        print(f"Loading embedding model: {config.EMBEDDING_MODEL} ...")
        _model = SentenceTransformer(config.EMBEDDING_MODEL)
    return _model


def _get_chroma_collection(video_url: str):
    """One ChromaDB collection per video (config.collection_name_for_url),
    so different videos' segment embeddings are fully isolated -- ids like
    'segment_0' from two different lectures can never collide."""
    import chromadb
    client = chromadb.PersistentClient(path=config.CHROMA_PATH)
    return client.get_or_create_collection(
        name=config.collection_name_for_url(video_url),
        metadata={"hnsw:space": "cosine"},
    )


def _chroma_id(segment_id: int) -> str:
    return f"segment_{segment_id}"


def get_or_compute_embeddings(segments: list[dict], video_url: str) -> np.ndarray:
    """
    Returns a (N, dim) float32 numpy array of L2-normalized embeddings,
    one row per segment in `segments`, in the same order.

    Idempotency / reuse strategy: if this video's ChromaDB collection
    already contains an embedding for every segment id AND the stored
    text for each one still matches the current transcript text exactly,
    those stored embeddings are reused as-is and the model is never
    loaded. Otherwise the model is loaded once and ALL segments are
    (re-)embedded in a single batched call, then upserted -- upsert
    (not add) is used specifically so re-running this on an existing
    collection overwrites rather than duplicates entries.

    This guarantees the embedding model is called at most once per
    run, and only when something has actually changed. `video_url`
    determines which per-video collection is read/written (see
    config.collection_name_for_url), so this never mixes embeddings
    across videos.
    """
    collection = _get_chroma_collection(video_url)
    ids = [_chroma_id(s["id"]) for s in segments]

    existing = None
    try:
        existing = collection.get(ids=ids, include=["embeddings", "documents"])
    except Exception:
        existing = None

    can_reuse = (
        existing is not None
        and existing.get("ids")
        and len(existing["ids"]) == len(ids)
        and all(
            existing["documents"][existing["ids"].index(cid)] == seg["text"]
            for cid, seg in zip(ids, segments)
            if cid in existing["ids"]
        )
        and all(cid in existing["ids"] for cid in ids)
    )

    if can_reuse:
        print(f"Reusing {len(ids)} existing embeddings from ChromaDB collection "
              f"'{config.collection_name_for_url(video_url)}' (text unchanged, skipping model call).")
        id_to_row = {cid: existing["embeddings"][existing["ids"].index(cid)] for cid in ids}
        embeddings = np.array([id_to_row[cid] for cid in ids], dtype=np.float32)
        return embeddings

    model = get_embedding_model()
    texts = [s["text"] for s in segments]

    print(f"Encoding {len(texts)} segments with {config.EMBEDDING_MODEL} "
          f"(batch_size={config.EMBEDDING_BATCH_SIZE}) ...")
    embeddings = model.encode(
        texts,
        batch_size=config.EMBEDDING_BATCH_SIZE,
        normalize_embeddings=True,
        show_progress_bar=True,
        convert_to_numpy=True,
    ).astype(np.float32)

    metadatas = [
        {
            "segment_id": s["id"],
            "start": s["start"],
            "end": s["end"],
            "duration": s["duration"],
        }
        for s in segments
    ]

    collection.upsert(
        ids=ids,
        embeddings=embeddings.tolist(),
        documents=texts,
        metadatas=metadatas,
    )
    print(f"Stored {len(ids)} embeddings in ChromaDB "
          f"(path='{config.CHROMA_PATH}', collection='{config.collection_name_for_url(video_url)}').")

    return embeddings
