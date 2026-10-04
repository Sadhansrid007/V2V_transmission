"""
retrieve_topics.py

Three-stage, url-keyed retrieval over the pipeline's output:

    Stage 0 (expansion):  rewrite conversational query phrasing -- e.g.
                          "when did the mam explain the necessary
                          conditions for deadlock" -- into a clean topical
                          phrase ("necessary conditions for deadlock")
                          before it's used for embedding recall. Falls
                          back to your original wording if this fails.

    Stage A (recall):    embed the (expanded) query, rank this video's topics in
                          topic_analysis.json by cosine similarity, and
                          take the top N (config.TOP_K_EMBEDDING_CANDIDATES)
                          as candidates. Cheap, but similarity alone can
                          surface topics that only superficially match the
                          query's wording, or miss ones that use different
                          terminology.

    Stage B (judgment):  pull each candidate's ACTUAL transcript text (from
                          topic_segmentation.json, not just its generated
                          description) and hand the whole shortlist to an
                          LLM in a single call, asking it to decide -- the
                          way a careful human research assistant would --
                          which of those candidates are genuinely relevant
                          to the query. The model is free to pick as many,
                          or as few, as it judges relevant, including none
                          or all of them.

Only topics judged relevant make it into the final results, and every one
of them is printed with its ready-to-click direct-timestamp link. There is
no interactive picker: the LLM judge already decided what's relevant, and
among those, the single highest-relevance_score result is automatically
opened in your browser -- no manual selection step.

Every input/cache here is scoped to a single video_url (see config.py's
url_key/url_scoped_path helpers), since topic_analysis.json,
topic_segmentation.json, silence.json, and topic_time_summary.json are all
shared files covering every video you've ever run the pipeline on. Because
a project directory can hold multiple videos, this script always asks you
which video to search -- it will never silently guess.

Run:
    python retrieve_topics.py "what are the necessary conditions for deadlock"
    python retrieve_topics.py                      # prompts for a query
    python retrieve_topics.py "process memory" --top-k 3
    python retrieve_topics.py "..." --url "https://..."   # select which video's topics to search
    python retrieve_topics.py "..." --candidates 12 # widen the embedding pool sent to the LLM judge
    python retrieve_topics.py "..." --no-open       # just print results (with links), don't open a browser

Requires: sentence-transformers, numpy, groq, pydantic, python-dotenv (all
already needed elsewhere in this pipeline, so nothing new to install).

Environment:
    GROQ_API_KEY must be set (used for the Stage 0 query expander and the
    Stage B relevance judge).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import sys
import time
import webbrowser
from typing import Dict, List, Optional
from urllib.parse import urlparse, parse_qs, urlencode, urlunparse

import numpy as np
from pydantic import BaseModel, Field, ValidationError

import config

try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:
    pass

try:
    import groq
    from groq import Groq
except ImportError:
    print("The 'groq' package is not installed. Run: pip install groq pydantic python-dotenv")
    sys.exit(1)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

ANALYSIS_FILE = "topic_analysis.json"          # url-keyed: {url_key: [topic analysis entries]}
SEGMENTATION_FILE = config.OUTPUT_TOPIC_JSON   # url-keyed: {url_key: [topics incl. full segments/text]}
SILENCE_FILE = "silence.json"                  # url-keyed; each entry records that video's video_url
TIME_SUMMARY_FILE = "topic_time_summary.json"  # url-keyed; each entry records that video's video_url

# Reuse the exact same embedding model the segmentation stage used, so the
# model is already downloaded/cached locally and query <-> topic embeddings
# live in the same space.
EMBEDDING_MODEL = config.EMBEDDING_MODEL

# Types to leave out of retrieval by default. "filler" has effectively no
# retrieval value per the classification spec; include it with --include-filler
# if you ever want to search it too.
DEFAULT_EXCLUDE_TYPES = {"filler"}

TOP_K_DEFAULT = 5  # how many FINAL (LLM-judged-relevant) results to show

# Cache of topic embeddings, one pair of files per video (see
# config.url_scoped_path), invalidated automatically whenever that video's
# topic_analysis.json content or the embedding model changes.
CACHE_FILE = "topic_embeddings_cache.npz"
CACHE_META_FILE = "topic_embeddings_cache_meta.json"

# Groq relevance-judge retry budget.
MAX_RETRIES = 4


# ---------------------------------------------------------------------------
# Stage B schema: the LLM judge's structured verdict per candidate topic
# ---------------------------------------------------------------------------

class TopicJudgment(BaseModel):
    topic_id: int
    relevant: bool
    relevance_score: float = Field(ge=0.0, le=1.0)
    reason: str


class RetrievalJudgment(BaseModel):
    judgments: List[TopicJudgment]


JUDGE_SYSTEM_PROMPT = """You are an expert retrieval assistant helping someone search a lecture's \
transcript.

You are given a user's search query and a shortlist of candidate topics that were already \
pre-selected by embedding similarity search. Embedding similarity is a rough recall filter: it \
sometimes surfaces topics that only superficially resemble the query's wording but aren't \
actually useful to someone who asked it, and it can also under-rank a genuinely relevant topic \
that just happens to use different terminology.

For EACH candidate topic below, read its actual transcript text (not just its short generated \
summary) and decide, the way a careful, honest human research assistant would, whether that \
topic would actually help answer -- or is meaningfully about -- the user's query.

You have complete freedom here:
- Select as many, or as few, of the candidates as are genuinely relevant. There is no cap --
  if the query is broad and five, eight, or every single candidate genuinely addresses it,
  mark all of them relevant=true. Do not artificially limit yourself to picking just one or two.
- It is correct to select zero if truly none of them address the query.
- It is correct to select several, or all, if the query is broad and multiple topics apply.
- Do not select a topic just because it was in the shortlist, and do not pad your selections to \
seem more helpful than the transcript actually supports.

Respond with ONLY a single JSON object of this shape, covering every candidate topic_id you were \
given (in any order):

{"judgments": [{"topic_id": <int>, "relevant": <true|false>, "relevance_score": <float 0.0-1.0>, \
"reason": <short one-sentence string>}, ...]}

"relevance_score" should reflect your actual confidence that the topic is relevant, not just \
mirror "relevant" as 0/1. No markdown code fences, no preamble, no commentary outside the JSON \
object."""


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------

def _hash_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _load_json(path: str):
    if not os.path.exists(path):
        return None
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _topic_corpus_text(topic: dict) -> str:
    name = topic.get("name", "")
    description = topic.get("description", "")
    keywords = ", ".join(topic.get("keywords", []))
    return f"{name}. {description} Keywords: {keywords}".strip()


_model = None  # loaded lazily, once


def _get_model():
    global _model
    if _model is None:
        from sentence_transformers import SentenceTransformer
        print(f"Loading embedding model: {EMBEDDING_MODEL} ...")
        _model = SentenceTransformer(EMBEDDING_MODEL)
    return _model


# ---------------------------------------------------------------------------
# Stage 0: query expansion (runs first, before Stage A embedding)
# ---------------------------------------------------------------------------
#
# People type queries the way they'd ask a classmate, not the way a topic
# summary is written -- e.g. "when did the mam explain the necessary
# conditions for deadlock" instead of "necessary conditions for deadlock".
# Raw phrases like that embed poorly against topic_analysis.json entries
# (which are written as clean topical summaries), so before Stage A we ask
# an LLM to strip the conversational wrapper -- references to "mam"/"sir"/
# "the professor", "when did ... explain/discuss/mention/cover", "where
# does she talk about", etc. -- and rewrite the query as a short, clean
# topical phrase, while preserving every substantive qualifier (e.g.
# "necessary conditions for X" must stay distinct from bare "X"). The
# ORIGINAL query is still shown to the Stage B judge alongside the
# expansion, so the judge sees the full intent even if the expansion is
# imperfect. Falls back to the original query, unmodified, if expansion
# fails for any reason -- this stage is a recall aid, never a hard gate.

QUERY_EXPANSION_SYSTEM_PROMPT = """You are a query-rewriting assistant for a lecture-video \
transcript search tool.

People phrase their searches conversationally, referring to the lecturer as "mam" / "ma'am" / \
"sir" / "the professor" / "the teacher", and wrapping the topic they actually want inside phrases \
like "when did the mam explain X", "where does sir talk about X", "what did she say about X", \
"did the professor cover X", "find where he mentioned X", "at what point does he discuss X".

Your job: strip that conversational wrapper and rewrite the query as a short, clean, topical \
search phrase -- the kind of phrase that would appear in a topic name or summary -- suitable for \
semantic embedding search. Concretely:
- Remove references to the lecturer ("mam", "sir", "professor", "she", "he", "the teacher", etc.) \
and meta-verbs about the act of teaching ("explain", "talk about", "discuss", "cover", "mention", \
"go over", "when did", "where does") UNLESS those words are themselves the actual subject matter \
(e.g. a query genuinely about "explanation techniques" should keep "explanation").
- Preserve every substantive qualifier in the original query exactly as scoped -- "necessary \
conditions for deadlock" is NOT the same search as "deadlock"; "difference between X and Y" is \
NOT the same as "X". Never broaden or narrow the topic beyond what was asked.
- Never invent subject matter that wasn't in the original query.
- If the query is already a clean topical phrase with no conversational wrapper, return it \
essentially unchanged.

Examples:
- "when did the mam explain deadlock" -> "deadlock"
- "when did the mam explain the necessary conditions for deadlock" -> "necessary conditions for deadlock"
- "where does sir talk about process memory layout" -> "process memory layout"
- "what did the professor say about virtual memory vs physical memory" -> "virtual memory vs physical memory"
- "did she cover page replacement algorithms" -> "page replacement algorithms"
- "attendance" -> "attendance"

Respond with ONLY a single JSON object of this exact shape, no markdown fences, no commentary \
outside the JSON:

{"expanded_query": <string>}"""


class QueryExpansion(BaseModel):
    expanded_query: str


QUERY_EXPANSION_MAX_RETRIES = 2


def expand_query(client: Groq, raw_query: str) -> str:
    """Runs Stage 0. Returns the LLM's rewritten, embedding-friendly query
    on success, or raw_query unchanged if the call fails after retries --
    expansion is a recall aid, so a failure here should never block the
    search."""
    last_error: Optional[Exception] = None

    for attempt in range(1, QUERY_EXPANSION_MAX_RETRIES + 1):
        try:
            response = client.chat.completions.create(
                model=config.RETRIEVAL_LLM_MODEL,
                temperature=0.0,
                max_tokens=200,
                response_format={"type": "json_object"},
                messages=[
                    {"role": "system", "content": QUERY_EXPANSION_SYSTEM_PROMPT},
                    {"role": "user", "content": raw_query},
                ],
            )
            content = response.choices[0].message.content
            if not content or not content.strip():
                raise ValueError("Empty response from model")

            data = json.loads(content)
            expansion = QueryExpansion(**data)
            expanded = expansion.expanded_query.strip()
            return expanded or raw_query

        except (json.JSONDecodeError, ValidationError, ValueError) as e:
            last_error = e
        except groq.RateLimitError as e:
            last_error = e
            time.sleep(min(3 * attempt, 10))
        except (groq.APIConnectionError, groq.APITimeoutError, groq.APIStatusError) as e:
            last_error = e

    print(f"  [query expansion] falling back to your original wording ({last_error})")
    return raw_query


# ---------------------------------------------------------------------------
# Loading this video's topics + their real transcript text
# ---------------------------------------------------------------------------

def load_topics(video_url: str, exclude_types: set) -> List[dict]:
    """Loads THIS video's topic_analysis.json entry (see config.url_key)."""
    topics = config.load_url_keyed_entry(ANALYSIS_FILE, video_url)
    if not topics:
        print(f"ERROR: no topic analysis found for this video_url in {ANALYSIS_FILE} "
              f"(key {config.url_key(video_url)}). Run the analysis stage for this video first.")
        sys.exit(1)
    filtered = [t for t in topics if t.get("type") not in exclude_types]
    if not filtered:
        print(f"No topics remain after excluding types {exclude_types}.")
        sys.exit(1)
    return filtered


def load_transcript_by_topic_id(video_url: str) -> Dict[int, str]:
    """Loads THIS video's topic_segmentation.json entry and returns a
    {topic_id: joined_transcript_text} map, so Stage B can hand the LLM
    judge each candidate's real transcript rather than just its generated
    description. Returns {} (not an error) if segmentation output isn't
    available for this video -- callers fall back to description-only text
    and print a warning."""
    segmentation_topics = config.load_url_keyed_entry(SEGMENTATION_FILE, video_url)
    if not segmentation_topics:
        return {}

    by_id = {}
    for topic in segmentation_topics:
        lines = [
            seg.get("text", "").strip()
            for seg in topic.get("segments", [])
            if seg.get("text", "").strip()
        ]
        text = "\n".join(lines)
        if len(text) > config.RETRIEVAL_MAX_TRANSCRIPT_CHARS_PER_TOPIC:
            text = text[: config.RETRIEVAL_MAX_TRANSCRIPT_CHARS_PER_TOPIC] + "\n[...truncated for length...]"
        by_id[topic["topic_id"]] = text
    return by_id


def get_or_compute_topic_embeddings(topics: List[dict], video_url: str) -> np.ndarray:
    """Embeds each topic's name/description/keywords, reusing a cached
    result if this video's topic_analysis.json content and the embedding
    model are both unchanged since the cache was written. Cache files are
    namespaced per-video via config.url_scoped_path."""
    cache_file = config.url_scoped_path(CACHE_FILE, video_url)
    cache_meta_file = config.url_scoped_path(CACHE_META_FILE, video_url)

    corpus_hash = _hash_text(
        EMBEDDING_MODEL + "||" + "||".join(_topic_corpus_text(t) for t in topics)
    )
    meta = _load_json(cache_meta_file)

    if (
        meta
        and meta.get("corpus_hash") == corpus_hash
        and os.path.exists(cache_file)
    ):
        print("Reusing cached topic embeddings (topic_analysis.json unchanged for this video).")
        return np.load(cache_file)["embeddings"]

    model = _get_model()
    texts = [_topic_corpus_text(t) for t in topics]
    print(f"Embedding {len(texts)} topic(s)...")
    embeddings = model.encode(
        texts,
        normalize_embeddings=True,
        convert_to_numpy=True,
        show_progress_bar=False,
    ).astype(np.float32)

    np.savez(cache_file, embeddings=embeddings)
    with open(cache_meta_file, "w", encoding="utf-8") as f:
        json.dump({"corpus_hash": corpus_hash, "embedding_model": EMBEDDING_MODEL}, f)

    return embeddings


def embed_query(query: str) -> np.ndarray:
    model = _get_model()
    return model.encode([query], normalize_embeddings=True, convert_to_numpy=True)[0].astype(np.float32)


def rank_topics_by_embedding(topics: List[dict], topic_embeddings: np.ndarray, query_embedding: np.ndarray, top_k: int):
    scores = topic_embeddings @ query_embedding  # cosine similarity, both sides L2-normalized
    order = np.argsort(-scores)[:top_k]
    return [(topics[i], float(scores[i])) for i in order]


# ---------------------------------------------------------------------------
# Stage B: LLM relevance judge
# ---------------------------------------------------------------------------

def build_judge_user_prompt(
    original_query: str,
    expanded_query: str,
    candidates: List[tuple],
    transcripts_by_id: Dict[int, str],
) -> str:
    """candidates is a list of (topic_dict, embedding_score) tuples, already
    ranked by Stage A. Shows the judge BOTH the user's original wording
    (full intent, including conversational phrasing like "when did the mam
    explain X") and the Stage 0 expansion used to drive embedding search
    (a clean topical phrase) -- so the judge can lean on the original if
    the expansion dropped or misread something."""
    parts = [f'User query (original wording): "{original_query}"']
    if expanded_query.strip().lower() != original_query.strip().lower():
        parts.append(f'Interpreted search topic (used for candidate recall): "{expanded_query}"')
    parts += ["", f"Candidate topics ({len(candidates)}):"]

    for topic, embedding_score in candidates:
        topic_id = topic["topic_id"]
        transcript_text = transcripts_by_id.get(topic_id)
        if not transcript_text:
            # No segmentation data available for this topic (e.g. the
            # segmentation output was pruned/regenerated) -- fall back to
            # the generated description so the judge still has something
            # concrete to reason over.
            transcript_text = f"[transcript unavailable; description only] {topic.get('description', '')}"

        parts.append(
            f"\n--- Candidate topic_id={topic_id} "
            f"(embedding_score={embedding_score:.3f}, "
            f"{topic.get('start_time', 0):.1f}s-{topic.get('end_time', 0):.1f}s) ---"
        )
        parts.append(f"Name: {topic.get('name', '')}")
        parts.append(f"Generated description: {topic.get('description', '')}")
        parts.append(f'Actual transcript:\n"""\n{transcript_text}\n"""')

    parts.append("\nReturn the JSON object now, covering every candidate topic_id above.")
    return "\n".join(parts)


def _is_truncation_error(e: Exception) -> bool:
    text = str(e)
    return "json_validate_failed" in text or "max completion tokens reached" in text


def call_judge_with_retries(client: Groq, user_prompt: str) -> RetrievalJudgment:
    last_error: Optional[Exception] = None
    max_tokens = 2000

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            response = client.chat.completions.create(
                model=config.RETRIEVAL_LLM_MODEL,
                temperature=config.RETRIEVAL_LLM_TEMPERATURE,
                max_tokens=max_tokens,
                reasoning_effort=config.RETRIEVAL_LLM_REASONING_EFFORT,
                response_format={"type": "json_object"},
                messages=[
                    {"role": "system", "content": JUDGE_SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt},
                ],
            )
            content = response.choices[0].message.content
            if not content or not content.strip():
                raise ValueError("Empty response from model")

            data = json.loads(content)
            judgment = RetrievalJudgment(**data)
            for j in judgment.judgments:
                j.relevance_score = max(0.0, min(1.0, j.relevance_score))
            return judgment

        except (json.JSONDecodeError, ValidationError, ValueError) as e:
            last_error = e
            wait = min(2 ** attempt, 15) + random.uniform(0, 1)
            print(f"  [retry {attempt}/{MAX_RETRIES}] malformed judge response ({e}); retrying in {wait:.1f}s")
            time.sleep(wait)

        except groq.RateLimitError as e:
            last_error = e
            wait = min(5 * attempt, 45) + random.uniform(0, 2)
            print(f"  [retry {attempt}/{MAX_RETRIES}] rate limited; retrying in {wait:.1f}s")
            time.sleep(wait)

        except (groq.APIConnectionError, groq.APITimeoutError) as e:
            last_error = e
            wait = min(2 ** attempt, 20) + random.uniform(0, 1)
            print(f"  [retry {attempt}/{MAX_RETRIES}] network/timeout error ({e}); retrying in {wait:.1f}s")
            time.sleep(wait)

        except groq.APIStatusError as e:
            last_error = e
            if _is_truncation_error(e) and max_tokens < 6000:
                max_tokens = min(int(max_tokens * 1.6), 6000)
                print(f"  [retry {attempt}/{MAX_RETRIES}] output truncated; raising max_tokens to {max_tokens}")
            else:
                wait = min(2 ** attempt, 20) + random.uniform(0, 1)
                print(f"  [retry {attempt}/{MAX_RETRIES}] API error ({e}); retrying in {wait:.1f}s")
                time.sleep(wait)

    raise RuntimeError(f"Judge failed to return a valid response after {MAX_RETRIES} attempts: {last_error}")


def judge_candidates(
    client: Groq,
    original_query: str,
    expanded_query: str,
    candidates: List[tuple],
    transcripts_by_id: Dict[int, str],
) -> List[dict]:
    """Runs Stage B: sends the whole shortlist to the LLM in one call and
    returns a list of dicts (one per candidate) merging the Stage A
    embedding score with the LLM's relevant/relevance_score/reason
    verdict, sorted with relevant=True first (by relevance_score desc),
    then relevant=False (by relevance_score desc). The judge is explicitly
    free to mark ANY number of candidates relevant=True -- there is no
    artificial cap here; capping only happens later, for display, via
    --top-k."""
    user_prompt = build_judge_user_prompt(original_query, expanded_query, candidates, transcripts_by_id)

    print(f"Asking the LLM judge ({config.RETRIEVAL_LLM_MODEL}) to review "
          f"{len(candidates)} candidate topic(s)...")
    judgment = call_judge_with_retries(client, user_prompt)

    verdict_by_id = {j.topic_id: j for j in judgment.judgments}
    topic_by_id = {t["topic_id"]: (t, score) for t, score in candidates}

    merged = []
    for topic_id, (topic, embedding_score) in topic_by_id.items():
        verdict = verdict_by_id.get(topic_id)
        if verdict is None:
            # Model dropped this topic_id from its response; treat as "not
            # relevant" rather than silently keeping it.
            merged.append({
                "topic": topic,
                "embedding_score": embedding_score,
                "relevant": False,
                "relevance_score": 0.0,
                "reason": "(no verdict returned by the LLM judge for this candidate)",
            })
        else:
            merged.append({
                "topic": topic,
                "embedding_score": embedding_score,
                "relevant": verdict.relevant,
                "relevance_score": verdict.relevance_score,
                "reason": verdict.reason,
            })

    merged.sort(key=lambda m: (not m["relevant"], -m["relevance_score"]))
    return merged


# ---------------------------------------------------------------------------
# Video URL resolution (also determines WHICH video's topics get searched)
# ---------------------------------------------------------------------------

def _known_video_urls() -> Dict[str, str]:
    """Scans silence.json and topic_time_summary.json for {url_key:
    video_url} pairs, so we can list every video this project directory
    has ever processed and make you pick one explicitly."""
    known = {}
    for path in (SILENCE_FILE, TIME_SUMMARY_FILE):
        data = config.load_url_keyed_json(path)
        for key, entry in data.items():
            if isinstance(entry, dict) and entry.get("video_url"):
                known[key] = entry["video_url"]
    return known


def resolve_video_url(url_override: Optional[str]) -> str:
    """Resolves which video's topics to search AND which video to open in
    the browser later -- both are the same video_url.

    This ALWAYS asks (unless --url was passed): with multiple videos
    known, it prints a numbered menu and requires you to pick one -- no
    silent auto-selection. With exactly one video known, it still shows
    it explicitly and requires you to confirm (Enter) or override with a
    different URL, so it's never ambiguous which video you're searching."""
    if url_override:
        return url_override

    known = _known_video_urls()
    urls = list(known.values())

    if len(urls) == 0:
        entered = input("No previously-processed videos were detected in this project "
                         "directory. Video URL to search: ").strip()
        if not entered:
            print("ERROR: no video URL provided and none could be detected.")
            sys.exit(1)
        return entered

    print("Videos found in this project directory:")
    for i, url in enumerate(urls, start=1):
        print(f"  [{i}] {url}")

    if len(urls) == 1:
        prompt = f"Search video [1] above? Press Enter to confirm, or paste a different URL: "
    else:
        prompt = f"Which video do you want to search? Enter a number [1-{len(urls)}] or paste a URL: "

    while True:
        entered = input(prompt).strip()

        if not entered and len(urls) == 1:
            return urls[0]

        if entered.isdigit():
            idx = int(entered) - 1
            if 0 <= idx < len(urls):
                return urls[idx]
            print(f"'{entered}' isn't one of the listed numbers -- try again.")
            continue

        if entered:
            return entered

        print("Please enter a number from the list above or paste a video URL.")


def build_timestamped_url(url: str, start_seconds: float, end_seconds: Optional[float] = None) -> str:
    """Best-effort timestamped link that jumps straight to the topic when
    opened:
      - YouTube: ?t=123s query param
      - Vimeo: #t=123s fragment
      - Anything else (direct video file URLs, e.g. mp4/webm): #t=start,end
        Media Fragments URI syntax -- Chrome/Firefox seek to that range when
        you open the link directly or in a <video> tag.

    Note: the #t= fragment form only seeks automatically when the URL points
    straight at a media file (or a page using a plain HTML5 <video> tag) --
    it does nothing on a generic webpage whose player is driven by
    JavaScript, since fragments aren't sent to the server or read by
    arbitrary page scripts. YouTube/Vimeo are special-cased above precisely
    because they honor query/fragment params for seeking.
    """
    seconds = int(round(start_seconds))
    parsed = urlparse(url)
    host = parsed.netloc.lower()

    if "youtube.com" in host or "youtu.be" in host:
        query = parse_qs(parsed.query)
        query["t"] = [f"{seconds}s"]
        new_query = urlencode(query, doseq=True)
        return urlunparse(parsed._replace(query=new_query))

    base = url.split("#")[0]
    if "vimeo.com" in host:
        return f"{base}#t={seconds}s"

    if end_seconds is not None:
        return f"{base}#t={seconds},{int(round(end_seconds))}"
    return f"{base}#t={seconds}"


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _looks_like_remote_environment() -> Optional[str]:
    """Best-effort detection of environments where webbrowser.open() launches
    a browser on the machine running the script, not the machine you're
    looking at -- which is why it can silently do nothing."""
    if os.environ.get("SSH_CONNECTION") or os.environ.get("SSH_CLIENT"):
        return "an SSH session"
    if os.environ.get("CODESPACES"):
        return "a GitHub Codespace"
    if os.environ.get("REMOTE_CONTAINERS") or os.environ.get("REMOTE_CONTAINERS_IPC"):
        return "a VS Code Dev Container"
    if os.environ.get("WSL_DISTRO_NAME"):
        return "WSL"
    if os.environ.get("VSCODE_GIT_ASKPASS_NODE") and not os.environ.get("DISPLAY") and sys.platform.startswith("linux"):
        return "a VS Code remote/headless session"
    return None


def print_results(results: List[dict], video_url: str) -> None:
    """Prints every result with its ready-to-click direct-timestamp link
    up front, so you have the jump-to-time URL immediately without having
    to go through the interactive picker below."""
    print()
    for rank, r in enumerate(results, start=1):
        topic = r["topic"]
        link = build_timestamped_url(video_url, topic["start_time"], topic["end_time"])
        print(f"[{rank}] relevance={r['relevance_score']:.2f}  embedding_sim={r['embedding_score']:.3f}  "
              f"type={topic['type']}  {topic['start_time']:.1f}s-{topic['end_time']:.1f}s")
        print(f"    {topic['name']}")
        print(f"    judge: {r['reason']}")
        print(f"    >>> {link}")
        print()


def main() -> None:
    parser = argparse.ArgumentParser(description="Two-stage (embedding + LLM judge) retrieval over topic_analysis.json")
    parser.add_argument("query", nargs="?", help="Search query. Omit to be prompted.")
    parser.add_argument("--top-k", type=int, default=TOP_K_DEFAULT,
                         help="Max number of final (LLM-judged-relevant) results to show.")
    parser.add_argument("--candidates", type=int, default=config.TOP_K_EMBEDDING_CANDIDATES,
                         help="How many embedding-ranked candidates to send to the LLM judge.")
    parser.add_argument("--include-filler", action="store_true",
                         help="Also search topics classified as filler.")
    parser.add_argument("--no-open", action="store_true",
                         help="Only print results (with links); don't auto-open the best match in a browser.")
    parser.add_argument("--url", type=str, default=None,
                         help="Video URL to search (and later open). Skips the video picker prompt.")
    args = parser.parse_args()

    query = args.query or input("Enter your query: ").strip()
    if not query:
        print("ERROR: empty query.")
        sys.exit(1)

    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        print("ERROR: GROQ_API_KEY environment variable is not set (needed for the LLM relevance judge).")
        sys.exit(1)
    client = Groq(api_key=api_key)

    # Stage 0: rewrite conversational phrasing ("when did the mam explain
    # X") into a clean topical phrase before it's used for embedding
    # recall. The judge in Stage B still sees your original wording too.
    expanded_query = expand_query(client, query)
    if expanded_query.strip().lower() != query.strip().lower():
        print(f'Interpreted your query as: "{expanded_query}"')

    video_url = resolve_video_url(args.url)
    print(f"\nSearching video: {video_url}  (key {config.url_key(video_url)})")

    exclude_types = set() if args.include_filler else set(DEFAULT_EXCLUDE_TYPES)
    topics = load_topics(video_url, exclude_types)
    transcripts_by_id = load_transcript_by_topic_id(video_url)
    if not transcripts_by_id:
        print(f"WARNING: no {SEGMENTATION_FILE} entry found for this video -- the LLM judge will "
              f"fall back to generated descriptions instead of actual transcript text.")

    topic_embeddings = get_or_compute_topic_embeddings(topics, video_url)
    query_embedding = embed_query(expanded_query)

    candidate_pool = min(args.candidates, len(topics))
    embedding_candidates = rank_topics_by_embedding(topics, topic_embeddings, query_embedding, candidate_pool)

    judged = judge_candidates(client, query, expanded_query, embedding_candidates, transcripts_by_id)

    all_relevant = [j for j in judged if j["relevant"]]
    relevant_results = all_relevant[: args.top_k]
    if len(all_relevant) > len(relevant_results):
        print(f"\nNote: the LLM judge found {len(all_relevant)} relevant topic(s) total; "
              f"showing the top {len(relevant_results)} (raise --top-k to see more).")

    if not relevant_results:
        print("\nThe LLM judge did not find any of the embedding-shortlisted candidates clearly "
              "relevant to this query. Showing the single best embedding match instead, "
              "flagged as unconfirmed:\n")
        fallback = judged[0] if judged else None
        if fallback is None:
            print("No candidates at all -- nothing to show.")
            return
        fallback = dict(fallback)
        fallback["reason"] = "(fallback: best embedding match; the LLM judge did not confirm this as relevant) " + fallback["reason"]
        relevant_results = [fallback]

    print_results(relevant_results, video_url)

    # No human picker: the judge already decided what's relevant, and among
    # those, relevant_results is sorted by relevance_score descending (see
    # judge_candidates' merged.sort), so index 0 IS the judge's best pick.
    best = relevant_results[0]
    best_topic = best["topic"]
    timestamped_url = build_timestamped_url(
        video_url, best_topic["start_time"], best_topic["end_time"]
    )
    print(f"Auto-selected (highest relevance_score={best['relevance_score']:.2f}): "
          f"{best_topic['name']} ({best_topic['start_time']:.1f}s-{best_topic['end_time']:.1f}s)")
    print(f">>> {timestamped_url}")

    if args.no_open:
        return

    remote_hint = _looks_like_remote_environment()
    if remote_hint:
        print(f"\nDetected you're likely running this from {remote_hint} via VS Code -- "
              f"opening a browser here would open one on the remote machine, not yours, "
              f"which looks like nothing happening.")
        print("Ctrl/Cmd+click the URL printed above in the VS Code terminal to open it "
              "on your own machine instead.")
        return

    opened = False
    try:
        opened = webbrowser.open(timestamped_url)
    except Exception as e:
        print(f"webbrowser.open() raised an error: {e}")

    if not opened:
        print("Could not launch a browser automatically. Ctrl/Cmd+click the URL "
              "above to open it manually.")


if __name__ == "__main__":
    main()