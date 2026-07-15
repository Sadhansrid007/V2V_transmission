import uuid

from sentence_transformers import SentenceTransformer

from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    FieldCondition,
    Filter,
    MatchValue,
    PointStruct,
    VectorParams,
)

# ----------------------------------
# Config
# ----------------------------------

# Local, on-disk Qdrant (embedded mode -- no server needed). Only ONE
# process/client can hold this path open at a time; open the client,
# do the work, and let it go out of scope before another part of the
# pipeline (e.g. chapters.py) opens its own client on the same path.
QDRANT_PATH = "./qdrant_db"

# One point per SEGMENT (== one chunk; chunking.py no longer builds
# sliding windows -- ingestion.py already produced final, duration-based
# segments, so each segment is stored as exactly one chunk here).
CHUNKS_COLLECTION = "lecture_chunks"

# One point per VIDEO, holding everything that would otherwise be
# duplicated across every chunk: full transcript text, silences, and
# max_duration. Chunk points only carry their own start/end/text plus
# the video_url they belong to.
VIDEOS_COLLECTION = "video_metadata"

# Video-metadata points don't need to be found by vector similarity --
# only by video_url -- so they get a trivial 1-dim placeholder vector
# instead of a real embedding.
_METADATA_VECTOR_SIZE = 1

# ----------------------------------
# Load embedding model once
# ----------------------------------
_model = None


def _get_model() -> SentenceTransformer:
    global _model
    if _model is None:
        _model = SentenceTransformer("BAAI/bge-m3")
    return _model


# ----------------------------------
# Deterministic IDs (so re-running the same video/segment overwrites
# instead of duplicating, and so video metadata can be fetched by ID
# directly instead of always needing a payload filter/scroll)
# ----------------------------------
_ID_NAMESPACE = uuid.NAMESPACE_URL


def _video_point_id(video_url: str) -> str:
    return str(uuid.uuid5(_ID_NAMESPACE, video_url))


def _chunk_point_id(video_url: str, segment_id: int) -> str:
    return str(uuid.uuid5(_ID_NAMESPACE, f"{video_url}::segment::{segment_id}"))


def _video_already_stored(client: QdrantClient, video_url: str) -> bool:
    existing = client.retrieve(
        collection_name=VIDEOS_COLLECTION,
        ids=[_video_point_id(video_url)],
    )
    return len(existing) > 0


def _ensure_collections(client: QdrantClient, embedding_size: int) -> None:
    if not client.collection_exists(CHUNKS_COLLECTION):
        client.create_collection(
            collection_name=CHUNKS_COLLECTION,
            vectors_config=VectorParams(size=embedding_size, distance=Distance.COSINE),
        )
    if not client.collection_exists(VIDEOS_COLLECTION):
        client.create_collection(
            collection_name=VIDEOS_COLLECTION,
            vectors_config=VectorParams(size=_METADATA_VECTOR_SIZE, distance=Distance.COSINE),
        )


# ----------------------------------
# Main entry point
# ----------------------------------
def chunk_transcript(
    transcript: dict,
    video_url: str,
    qdrant_path: str = QDRANT_PATH,
    force: bool = False,
):
    """
    Takes the output of ingestion() and stores EVERYTHING in Qdrant --
    ingestion.py itself doesn't persist anything, so this is where the
    transcript first gets saved.

    Storage layout
    --------------
    - `video_metadata` collection: exactly ONE point per video_url,
      payload = {video_url, text, silences, max_duration, segment_count}.
      This is where the full transcript text and the silence list live,
      so they are never duplicated across chunks.
    - `lecture_chunks` collection: one point per segment (1 segment ==
      1 chunk -- no windowing/overlap anymore, ingestion.py already
      produced final duration-based segments). Each point's vector is
      the embedding of that segment's own text; payload =
      {video_url, segment_id, start, end, duration, text}.

    Parameters
    ----------
    transcript : dict
        Output of ingestion(). Must contain "segments" (list of dicts
        with "id"/"start"/"end"/"duration"/"text"), "text", "silences",
        and "max_duration".
    video_url : str
        Used as the retrieval key everywhere in this pipeline.
    force : bool
        Re-embed and overwrite even if this video_url is already stored.

    Returns
    -------
    list[dict]
        The segments that were embedded and stored (same shape as
        transcript["segments"], each with an added "embedding" key).
        Empty list if there was nothing to store, None if skipped
        because the video already existed and force=False.
    """
    client = QdrantClient(path=qdrant_path)

    segments = transcript.get("segments", [])
    if not segments:
        print("No segments in transcript -- nothing to embed or store.")
        return []

    model = _get_model()
    embedding_size = model.get_sentence_embedding_dimension()
    _ensure_collections(client, embedding_size)

    if not force and _video_already_stored(client, video_url):
        print(f"'{video_url}' already stored in Qdrant -- skipping.")
        print("Pass force=True to re-embed and overwrite.")
        return None

    # ----------------------------------
    # Embed every segment's text (1 segment = 1 chunk)
    # ----------------------------------
    texts = [s["text"] for s in segments]
    embeddings = model.encode(
        texts,
        batch_size=32,
        normalize_embeddings=True,
        convert_to_numpy=True,
        show_progress_bar=True,
    )

    chunk_points = []
    for seg, emb in zip(segments, embeddings):
        seg["embedding"] = emb.tolist()
        chunk_points.append(
            PointStruct(
                id=_chunk_point_id(video_url, seg["id"]),
                vector=seg["embedding"],
                payload={
                    "video_url": video_url,
                    "segment_id": seg["id"],
                    "start": seg["start"],
                    "end": seg["end"],
                    "duration": seg["duration"],
                    "text": seg["text"],
                },
            )
        )

    client.upsert(collection_name=CHUNKS_COLLECTION, points=chunk_points)
    print(f"Stored {len(chunk_points)} chunk(s) in '{CHUNKS_COLLECTION}'.")

    # ----------------------------------
    # Video-level metadata: stored exactly once, not duplicated per chunk
    # ----------------------------------
    video_point = PointStruct(
        id=_video_point_id(video_url),
        vector=[0.0] * _METADATA_VECTOR_SIZE,
        payload={
            "video_url": video_url,
            "text": transcript.get("text", ""),
            "silences": transcript.get("silences", []),
            "max_duration": transcript.get("max_duration"),
            "segment_count": len(segments),
        },
    )
    client.upsert(collection_name=VIDEOS_COLLECTION, points=[video_point])
    print(f"Stored video metadata in '{VIDEOS_COLLECTION}' for {video_url}")

    return segments


# ----------------------------------
# Read-side helpers used by chapters.py
# ----------------------------------
def load_chunks_for_video(video_url: str, qdrant_path: str = QDRANT_PATH) -> list[dict]:
    """
    Pulls every chunk for this video_url, sorted chronologically, with
    each chunk's embedding attached as a plain list of floats.
    """
    client = QdrantClient(path=qdrant_path)

    if not client.collection_exists(CHUNKS_COLLECTION):
        return []

    points, _ = client.scroll(
        collection_name=CHUNKS_COLLECTION,
        scroll_filter=Filter(
            must=[FieldCondition(key="video_url", match=MatchValue(value=video_url))]
        ),
        with_vectors=True,
        limit=100_000,
    )

    chunks = [
        {
            "id": p.payload["segment_id"],
            "start": p.payload["start"],
            "end": p.payload["end"],
            "duration": p.payload["duration"],
            "text": p.payload["text"],
            "embedding": p.vector,
        }
        for p in points
    ]
    chunks.sort(key=lambda c: c["start"])
    return chunks


def load_video_metadata(video_url: str, qdrant_path: str = QDRANT_PATH) -> dict | None:
    """Returns {video_url, text, silences, max_duration, segment_count} or None."""
    client = QdrantClient(path=qdrant_path)

    if not client.collection_exists(VIDEOS_COLLECTION):
        return None

    points = client.retrieve(
        collection_name=VIDEOS_COLLECTION,
        ids=[_video_point_id(video_url)],
    )
    if not points:
        return None
    return points[0].payload
