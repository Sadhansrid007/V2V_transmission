import uuid

from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams

# ----------------------------------
# Config
# ----------------------------------

# Deliberately a SEPARATE Qdrant DB from the main pipeline's
# (chunking.py's QDRANT_PATH="./qdrant_db"). This isn't part of the
# per-video ingestion -> chunking -> chapters flow: it's a one-time (or
# occasional) manual seed of the allowed topic list per subject, so it
# gets its own storage instead of living alongside video chunks.
TOPICS_QDRANT_PATH = "./qdrant_topics_db"
TOPICS_COLLECTION = "subject_topics"

# No embedding needed here (topics are matched by exact subject_code,
# not by similarity search -- see chapters.py, which pulls the full
# topic list for a subject_code and has the LLM pick directly from it).
# A trivial 1-dim placeholder vector is used so the collection still
# satisfies Qdrant's "every point needs a vector" requirement.
_VECTOR_SIZE = 1

_ID_NAMESPACE = uuid.NAMESPACE_URL


def _subject_point_id(subject_code: str) -> str:
    return str(uuid.uuid5(_ID_NAMESPACE, f"subject::{subject_code}"))


def _ensure_collection(client: QdrantClient) -> None:
    if not client.collection_exists(TOPICS_COLLECTION):
        client.create_collection(
            collection_name=TOPICS_COLLECTION,
            vectors_config=VectorParams(size=_VECTOR_SIZE, distance=Distance.COSINE),
        )


# ----------------------------------
# Write side
# ----------------------------------
def upload_topics(
    topics_by_subject: dict[str, list[str]],
    qdrant_path: str = TOPICS_QDRANT_PATH,
) -> None:
    """
    Stores the given topic lists, one Qdrant point per subject_code,
    overwriting any existing entry for that subject_code.

    Parameters
    ----------
    topics_by_subject : dict[str, list[str]]
        Exactly what you hand this function yourself, e.g.:
            {
                "CS101": [
                    "Binary Search Trees",
                    "Recursive Base Cases",
                    "Preprocessor Directives",
                ],
                "CS201": [
                    "Dijkstra's Shortest Path",
                    "Graph Traversal",
                ],
            }
        There is no file-reading step -- you pass the dict in directly.
    """
    client = QdrantClient(path=qdrant_path)
    _ensure_collection(client)

    points = [
        PointStruct(
            id=_subject_point_id(subject_code),
            vector=[0.0] * _VECTOR_SIZE,
            payload={
                "subject_code": subject_code,
                "topics": sorted(set(topics)),
            },
        )
        for subject_code, topics in topics_by_subject.items()
    ]

    client.upsert(collection_name=TOPICS_COLLECTION, points=points)

    for subject_code, topics in topics_by_subject.items():
        print(f"Stored {len(set(topics))} topic(s) for subject '{subject_code}'.")


# ----------------------------------
# Read side (used by chapters.py)
# ----------------------------------
def get_topics_for_subject(
    subject_code: str,
    qdrant_path: str = TOPICS_QDRANT_PATH,
) -> list[str]:
    """Returns the allowed topic list for a subject_code, or [] if none stored."""
    client = QdrantClient(path=qdrant_path)

    if not client.collection_exists(TOPICS_COLLECTION):
        return []

    points = client.retrieve(
        collection_name=TOPICS_COLLECTION,
        ids=[_subject_point_id(subject_code)],
    )
    if not points:
        return []
    return points[0].payload.get("topics", [])


# ----------------------------------
# Edit this dict with your own subject -> topics list, then run this
# file directly (`python topics.py`) to seed/update Qdrant with it.
# ----------------------------------
TOPICS_BY_SUBJECT: dict[str, list[str]] = {
    # "UE25CS151": [
    #     "Binary Search Trees",
    #     "Recursive Base Cases",
    #     "Preprocessor Directives",
    # ],
}


if __name__ == "__main__":
    print("=" * 50)
    print("Upload Subject Topics to Qdrant")
    print("=" * 50)

    subject_code = input("Enter Subject Code: ").strip()

    topics_input = input(
        "\nEnter topics separated by commas:\n"
    ).strip()

    topics = [t.strip() for t in topics_input.split(",") if t.strip()]

    if not subject_code:
        print("\nSubject code cannot be empty.")
    elif not topics:
        print("\nNo topics entered.")
    else:
        upload_topics({subject_code: topics})
        print("\nTopics uploaded successfully.")