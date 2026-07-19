"""
rag_backend.py
---------------
Job: store a transcribed lecture's segments in ChromaDB, and answer
student queries by retrieving relevant segments + asking Groq's LLM
to summarize them and pick exact timestamps for clipping.

Key fixes vs the two earlier prototypes:
- Uses config.CHROMA_DIR (not a hardcoded relative path) -- this is
  exactly what was causing the "ChromaDB path mismatch between
  modules" bug before.
- Uses ONE COLLECTION PER VIDEO (named after video_id) instead of a
  single shared collection that gets wiped every time a new video is
  processed. This means old videos stay searchable and you don't lose
  data by uploading a second lecture.
- Reads directly from transcribe.py's JSON output (segments with
  word-level timestamps) instead of a separate .txt transcript format.
"""

import json

import chromadb
from groq import Groq
from sentence_transformers import SentenceTransformer

from config import GROQ_API_KEY, GROQ_MODEL, EMBEDDING_MODEL, CHROMA_DIR, TRANSCRIPT_DIR, SEGMENT_MIN_SECONDS, SEGMENT_MAX_GAP_SECONDS
from query_expander import expand_query

_embedding_model = None
_groq_client = None
_chroma_client = None


def _get_embedding_model() -> SentenceTransformer:
    global _embedding_model
    if _embedding_model is None:
        _embedding_model = SentenceTransformer(EMBEDDING_MODEL)
    return _embedding_model


def _get_groq_client() -> Groq:
    global _groq_client
    if _groq_client is None:
        _groq_client = Groq(api_key=GROQ_API_KEY)
    return _groq_client


def _get_chroma_client():
    global _chroma_client
    if _chroma_client is None:
        _chroma_client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    return _chroma_client


def _collection_name(video_id: str) -> str:
    return f"lecture_{video_id}"


def _merge_segments(
    segments: list[dict],
    min_seconds: float = SEGMENT_MIN_SECONDS,
    max_gap_seconds: float = SEGMENT_MAX_GAP_SECONDS,
) -> list[dict]:
    """
    Greedily merges consecutive transcript segments into retrieval
    chunks based on duration and pause length, rather than a fixed
    number of segments per chunk.

    A chunk keeps growing by appending segments until its running
    duration (last.end - first.start) reaches min_seconds -- at that
    point it's closed, including the segment that pushed it over.

    Early-close override: while still under the duration target, if
    the gap to the next segment is >= max_gap_seconds, the chunk closes
    early rather than reaching across a long pause (e.g. a break, or a
    switch in topic) just to hit the duration target.

    This is more natural than a fixed window/stride: a long uninterrupted
    explanation becomes one coherent chunk, while a short segment right
    before a long pause doesn't get artificially padded with unrelated
    content from after the pause.
    """
    if not segments:
        return []

    def _duration(group: list[dict]) -> float:
        return group[-1]["end"] - group[0]["start"]

    def _flush(group: list[dict]) -> dict:
        return {
            "text": " ".join(s["text"].strip() for s in group).strip(),
            "start": group[0]["start"],
            "end": group[-1]["end"],
        }

    merged = []
    current = [segments[0]]

    for seg in segments[1:]:
        if _duration(current) >= min_seconds:
            merged.append(_flush(current))
            current = [seg]
            continue

        gap = seg["start"] - current[-1]["end"]
        if gap >= max_gap_seconds:
            merged.append(_flush(current))
            current = [seg]
            continue

        current.append(seg)

    if current:
        merged.append(_flush(current))

    return merged


def store_transcript(video_id: str) -> int:
    """
    Loads the transcript JSON for video_id (produced by transcribe.py),
    merges segments into retrieval-friendly chunks, embeds them, and
    stores them in a collection dedicated to this video.

    Returns the number of chunks stored.
    """
    transcript_path = TRANSCRIPT_DIR / f"{video_id}.json"
    if not transcript_path.exists():
        raise FileNotFoundError(
            f"No transcript found for video_id={video_id}. Run transcribe() first."
        )

    transcript = json.loads(transcript_path.read_text())
    merged_chunks = _merge_segments(transcript["segments"])

    if not merged_chunks:
        raise ValueError(f"Transcript for video_id={video_id} has no segments to store.")

    client = _get_chroma_client()
    collection = client.get_or_create_collection(name=_collection_name(video_id))

    # Re-processing the same video should replace old chunks, not
    # duplicate them -- so wipe this video's own collection only
    # (never touches other videos' collections).
    existing = collection.get()
    if existing["ids"]:
        collection.delete(ids=existing["ids"])

    embedder = _get_embedding_model()
    vectors = embedder.encode([c["text"] for c in merged_chunks]).tolist()

    collection.add(
        ids=[str(i) for i in range(len(merged_chunks))],
        embeddings=vectors,
        documents=[c["text"] for c in merged_chunks],
        metadatas=[{"start": c["start"], "end": c["end"]} for c in merged_chunks],
    )

    return len(merged_chunks)


def _search_with_expansion(video_id: str, student_prompt: str, top_k: int = 8) -> list[dict]:
    client = _get_chroma_client()
    try:
        collection = client.get_collection(name=_collection_name(video_id))
    except Exception:
        raise ValueError(
            f"No stored transcript for video_id={video_id}. Call store_transcript() first."
        )

    embedder = _get_embedding_model()
    expanded_queries = expand_query(student_prompt)

    all_results = {}
    for query in expanded_queries:
        query_vector = embedder.encode(query).tolist()
        results = collection.query(query_embeddings=[query_vector], n_results=top_k)

        for i in range(len(results["documents"][0])):
            doc_id = results["ids"][0][i]
            if doc_id not in all_results:
                all_results[doc_id] = {
                    "text": results["documents"][0][i],
                    "start": results["metadatas"][0][i]["start"],
                    "end": results["metadatas"][0][i]["end"],
                }

    # Sort chronologically -- makes the context block read like a
    # coherent excerpt of the lecture instead of a shuffled bag of
    # matches, which helps the LLM write a better description.
    return sorted(all_results.values(), key=lambda c: c["start"])


def _get_answer_and_timestamps(video_id: str, student_prompt: str) -> str:
    matched_chunks = _search_with_expansion(video_id, student_prompt)

    context = "\n".join(
        f"[{c['start']:.1f}s - {c['end']:.1f}s] {c['text']}" for c in matched_chunks
    )

    prompt = f"""You are helping a student find and clip specific parts of a lecture video.

Student request: "{student_prompt}"

Relevant transcript segments (in chronological order):
{context}

Do exactly three things:

1. Write a SHORT SUMMARY (1-2 sentences) of what was found.
   Label it exactly: SHORT: your summary here

2. Write a DETAILED DESCRIPTION (4-6 sentences) explaining what happens
   in the relevant segments, including context and key points.
   Label it exactly: DETAILED: your detailed description here

3. Return the timestamps needed for complete coverage of what the
   student asked, merging adjacent/overlapping ranges into single
   ranges rather than many tiny fragments.
   Label it exactly:
   TIMESTAMPS: [{{"start": 10.5, "end": 25.3}}, {{"start": 40.0, "end": 60.2}}]

Only use information from the transcript. If the topic isn't covered,
return TIMESTAMPS: []
"""

    client = _get_groq_client()
    response = client.chat.completions.create(
        model=GROQ_MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.2,
    )
    return response.choices[0].message.content


def _parse_response(llm_response: str) -> dict:
    short_summary, detailed_description, timestamps = "", "", []

    if "SHORT:" in llm_response:
        part = llm_response.split("SHORT:")[1]
        short_summary = (part.split("DETAILED:")[0] if "DETAILED:" in part else part).strip()

    if "DETAILED:" in llm_response:
        part = llm_response.split("DETAILED:")[1]
        detailed_description = (part.split("TIMESTAMPS:")[0] if "TIMESTAMPS:" in part else part).strip()

    if "TIMESTAMPS:" in llm_response:
        ts_string = llm_response.split("TIMESTAMPS:")[1].strip()
        try:
            timestamps = json.loads(ts_string)
        except json.JSONDecodeError:
            # Safety net: fish out any {"start": x, "end": y}-shaped
            # pairs even if the model wrapped them in extra text.
            import re
            pairs = re.findall(r'"start":\s*([\d.]+),\s*"end":\s*([\d.]+)', ts_string)
            timestamps = [{"start": float(s), "end": float(e)} for s, e in pairs]

    return {
        "short_summary": short_summary,
        "detailed_description": detailed_description,
        "timestamps": timestamps,
        "answer": short_summary,
    }


def process_query(video_id: str, student_prompt: str) -> dict:
    """Main entry point: ask a question about a stored video, get back
    a summary, description, and exact clip timestamps."""
    raw_response = _get_answer_and_timestamps(video_id, student_prompt)
    return _parse_response(raw_response)