"""
analyze_topics.py

Second-stage LLM-based topic analysis pipeline.

Reads topic_segmentation.json (output of an earlier transcript segmentation
stage), sends each topic's transcript text to a Groq-hosted LLM, and asks it
to:
    - classify the topic as important / filler / uncertain
    - generate a short, precise, retrieval-friendly name
    - generate a semantically dense, retrieval-oriented description
    - extract retrieval keywords
    - give a short classification rationale
    - give a confidence score

Results are written to topic_analysis.json. The original
topic_segmentation.json is never modified.

Run:
    python analyze_topics.py

Requires:
    pip install groq pydantic python-dotenv

Environment:
    GROQ_API_KEY must be set (see README notes / chat explanation for setup).
"""

from __future__ import annotations

import json
import os
import random
import sys
import time
from typing import List, Literal, Optional

from pydantic import BaseModel, Field, ValidationError

import config

try:
    # Optional: lets you keep GROQ_API_KEY in a local .env file instead of
    # exporting it in every shell session. Safe to skip if you don't use it.
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

INPUT_FILE = "topic_segmentation.json"
OUTPUT_FILE = "topic_analysis.json"

# Leave blank to be prompted, or pass the URL as a command-line argument.
VIDEO_URL = ""

# Groq model used for classification + generation. Change this freely if
# Groq's recommended model lineup changes; nothing else in this script
# depends on a specific model name.
GROQ_MODEL = "openai/gpt-oss-120b"

GROQ_TEMPERATURE = 0.1     # low temperature: this is classification/extraction, not creative writing
MAX_RETRIES = 5            # per-topic retry budget for API/parsing failures
MAX_OUTPUT_TOKENS = 1600   # headroom for reasoning + description + keywords + rationale
MAX_OUTPUT_TOKENS_CAP = 4000  # ceiling when auto-increasing after a truncation error

# openai/gpt-oss-120b (and gpt-oss-20b) support a reasoning_effort setting.
# Their hidden reasoning tokens count against max_tokens, so on longer topics
# "medium" reasoning can eat the whole budget before the JSON body is even
# written, causing "max completion tokens reached before generating a valid
# document" errors. "low" leaves more of the budget for the actual answer;
# raise this back to "medium" if you find classification quality suffers.
REASONING_EFFORT = "low"

# If True, skip topics that already have a valid entry in OUTPUT_FILE
# (matched on topic_id + boundary metadata), so reruns don't burn API calls
# on already-analyzed topics.
RESUME = True

# If True, copy the original `segments` array from topic_segmentation.json
# into each entry of topic_analysis.json. Default False: topic_id /
# start_segment / end_segment / start_time / end_time are already enough to
# trace back to the source file, and segments can contain a lot of text.
INCLUDE_SEGMENTS = False

# If True, give the model a little context from neighboring topics to help
# resolve ambiguity (e.g. a short topic that only makes sense next to the
# one before/after it). This does NOT send full transcripts repeatedly:
#   - previous topic: just its already-generated name + type (cheap, it was
#     already computed earlier in this same run)
#   - next topic: a short raw preview (first 2 segments) since it hasn't
#     been classified yet
USE_NEIGHBOR_CONTEXT = True

# Safety cap on how much transcript text we send per topic, to keep token
# usage predictable even if a topic has an unusually large number of
# segments. Very unlikely to trigger for typical lecture topics.
MAX_TRANSCRIPT_CHARS = 8000


# ---------------------------------------------------------------------------
# Output schema
# ---------------------------------------------------------------------------

class TopicClassification(BaseModel):
    type: Literal["important", "filler", "uncertain"]
    confidence: float = Field(ge=0.0, le=1.0)
    name: str
    description: str
    keywords: List[str] = Field(default_factory=list)
    rationale: str


# ---------------------------------------------------------------------------
# Prompting
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """You are an expert assistant that analyzes segments (topics) of a lecture \
transcript for a downstream retrieval and summarization system.

For the given topic, you must reason carefully about its actual semantic content (not just \
surface keywords) and produce a single JSON object with these fields:

- "type": one of "important", "filler", "uncertain"
- "confidence": a float between 0.0 and 1.0
- "name": a short, precise, descriptive topic name (about 2-8 words), specific enough to be \
useful for search. Do not use generic names like "Discussion", "Lecture", "Topic", "Important \
Topic".
- "description": a semantically dense, retrieval-oriented description (roughly 50-120 words). \
This is NOT a generic human-readable summary - it must preserve the specific concepts, \
terminology, definitions, examples, procedures, relationships, and facts actually present in \
the transcript, phrased so that a future search query about those specifics would match this \
topic. Do not add information that is not present in the transcript.
- "keywords": a list of 3-15 specific technical terms, concepts, entities, names, or \
domain-specific phrases actually present in the topic. Do not include generic words like \
"lecture", "teacher", "discussion", "class", "topic" unless they are genuinely relevant.
- "rationale": one or two short sentences explaining the classification decision (not hidden \
chain-of-thought, just the key factor(s)).

Classification guidance:

IMPORTANT - use this when the topic contains information useful for answering questions, \
understanding a concept/example/procedure/explanation/definition, understanding relationships \
between concepts, or remembering a significant classroom announcement or instruction. \
Attendance-related content (reminders to mark attendance, attendance deadlines, instructions \
about attendance) is ALWAYS important, never filler, even though it is administrative.

FILLER - use this for greetings, generic introductions with no real content, repetition that \
adds no new information, purely conversational classroom interaction ("are you understanding?", \
"can you hear me?", "okay?", "right?", "yes"), meaningless acknowledgements, off-topic chatter, \
and administrative material that is NOT attendance-related and has no retrieval value.

UNCERTAIN - use this only when you genuinely cannot confidently decide between important and \
filler: e.g. the transcript is too fragmented, the context is unclear, or it's a genuine mix of \
meaningful and meaningless material. Do not force every borderline case into important or \
filler - use uncertain when that is the honest answer.

Be careful with repetition: repeating an already-explained fact with no new information is \
filler, but restating something while adding new context, nuance, or elaboration is important.

Be careful with classroom questions: a question that carries real subject-matter content (e.g. \
"why does the OS maintain a separate stack per thread?") is potentially important; a purely \
conversational question ("are you following?") is filler.

Never hallucinate facts that are not present in the transcript text you are given, even if they \
would normally be true about the subject matter.

Respond with ONLY a single valid JSON object matching the schema above. No markdown code \
fences, no preamble, no commentary outside the JSON object."""


def build_transcript_block(topic: dict) -> str:
    """Concatenate a topic's segment texts into a clean block, truncating if needed."""
    lines = [seg.get("text", "").strip() for seg in topic.get("segments", []) if seg.get("text", "").strip()]
    text = "\n".join(lines)
    if len(text) > MAX_TRANSCRIPT_CHARS:
        text = text[:MAX_TRANSCRIPT_CHARS] + "\n[...truncated for length...]"
    return text


def build_user_prompt(
    topic: dict,
    prev_result: Optional[dict],
    next_topic: Optional[dict],
) -> str:
    topic_id = topic["topic_id"]
    start_time = topic.get("start_time")
    end_time = topic.get("end_time")

    parts = [f"Topic ID: {topic_id}"]
    if start_time is not None and end_time is not None:
        parts.append(f"Time range: {start_time:.2f}s - {end_time:.2f}s")

    if USE_NEIGHBOR_CONTEXT and prev_result is not None:
        parts.append(
            "Previous topic (for context only, already classified as "
            f"\"{prev_result.get('type')}\"): {prev_result.get('name')}"
        )

    if USE_NEIGHBOR_CONTEXT and next_topic is not None:
        preview_lines = [
            seg.get("text", "").strip()
            for seg in next_topic.get("segments", [])[:2]
            if seg.get("text", "").strip()
        ]
        if preview_lines:
            parts.append(
                "Upcoming topic preview (unclassified, for context only): "
                + " ".join(preview_lines)
            )

    transcript_block = build_transcript_block(topic)
    parts.append(f'\nTopic {topic_id} transcript:\n"""\n{transcript_block}\n"""')
    parts.append("\nReturn the JSON object now.")

    return "\n".join(parts)


# ---------------------------------------------------------------------------
# Groq call with retries / backoff
# ---------------------------------------------------------------------------

def _is_truncation_error(e: Exception) -> bool:
    """Detect Groq's 'ran out of tokens before finishing the JSON' error so
    we can respond by raising the token budget instead of just waiting."""
    text = str(e)
    return "json_validate_failed" in text or "max completion tokens reached" in text


def call_groq_with_retries(client: Groq, user_prompt: str) -> TopicClassification:
    last_error: Optional[Exception] = None
    max_tokens = MAX_OUTPUT_TOKENS

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            response = client.chat.completions.create(
                model=GROQ_MODEL,
                temperature=GROQ_TEMPERATURE,
                max_tokens=max_tokens,
                reasoning_effort=REASONING_EFFORT,
                response_format={"type": "json_object"},
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt},
                ],
            )
            content = response.choices[0].message.content

            if not content or not content.strip():
                raise ValueError("Empty response from model")

            data = json.loads(content)
            classification = TopicClassification(**data)
            # Keep confidence strictly within bounds even if the model
            # returns something like 1.0000001 due to formatting quirks.
            classification.confidence = max(0.0, min(1.0, classification.confidence))
            return classification

        except (json.JSONDecodeError, ValidationError, ValueError) as e:
            last_error = e
            wait = min(2 ** attempt, 20) + random.uniform(0, 1)
            print(f"      [retry {attempt}/{MAX_RETRIES}] malformed response ({e}); retrying in {wait:.1f}s")
            time.sleep(wait)

        except groq.RateLimitError as e:
            last_error = e
            wait = min(5 * attempt, 60) + random.uniform(0, 2)
            print(f"      [retry {attempt}/{MAX_RETRIES}] rate limited; retrying in {wait:.1f}s")
            time.sleep(wait)

        except (groq.APIConnectionError, groq.APITimeoutError) as e:
            last_error = e
            wait = min(2 ** attempt, 30) + random.uniform(0, 1)
            print(f"      [retry {attempt}/{MAX_RETRIES}] network/timeout error ({e}); retrying in {wait:.1f}s")
            time.sleep(wait)

        except groq.APIStatusError as e:
            last_error = e
            if _is_truncation_error(e) and max_tokens < MAX_OUTPUT_TOKENS_CAP:
                max_tokens = min(int(max_tokens * 1.6), MAX_OUTPUT_TOKENS_CAP)
                wait = 1.0
                print(f"      [retry {attempt}/{MAX_RETRIES}] output truncated; raising max_tokens to {max_tokens} and retrying in {wait:.1f}s")
            else:
                wait = min(2 ** attempt, 30) + random.uniform(0, 1)
                print(f"      [retry {attempt}/{MAX_RETRIES}] API error ({e}); retrying in {wait:.1f}s")
            time.sleep(wait)

    raise RuntimeError(f"Failed to get a valid classification after {MAX_RETRIES} attempts: {last_error}")


# ---------------------------------------------------------------------------
# Checkpointing helpers
# ---------------------------------------------------------------------------

def load_existing_results(path: str, video_url: str) -> dict:
    """Load this video's previously saved results (from the shared,
    url-keyed OUTPUT_FILE -- see config.py), keyed by topic_id, for resume
    support."""
    entry = config.load_url_keyed_entry(path, video_url)
    if not entry:
        return {}

    by_id = {}
    for item in entry:
        if isinstance(item, dict) and "topic_id" in item:
            by_id[item["topic_id"]] = item
    return by_id


def is_valid_existing_entry(entry: dict, topic: dict) -> bool:
    """Check that a saved entry matches the current source topic's boundaries
    and has all required analysis fields, so we know it's safe to reuse."""
    required_meta = ["start_segment", "end_segment", "start_time", "end_time"]
    for key in required_meta:
        if entry.get(key) != topic.get(key):
            return False
    required_analysis = ["type", "confidence", "name", "description", "keywords", "rationale"]
    return all(k in entry for k in required_analysis)


def save_results(path: str, video_url: str, results_by_id: dict, ordered_topic_ids: List[int]) -> None:
    ordered = [results_by_id[tid] for tid in ordered_topic_ids if tid in results_by_id]
    config.save_url_keyed_entry(path, video_url, ordered)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def _resolve_video_url(video_url: Optional[str]) -> str:
    if video_url:
        return video_url
    video_url = (sys.argv[1] if len(sys.argv) > 1 else "").strip() or VIDEO_URL.strip()
    if not video_url:
        video_url = input("Enter video URL (selects which video's topics to analyze): ").strip()
    if not video_url:
        print("ERROR: no video URL provided.")
        sys.exit(1)
    return video_url


def main(video_url: Optional[str] = None) -> None:
    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        print("ERROR: GROQ_API_KEY environment variable is not set.")
        print("See the setup instructions for your shell (PowerShell / bash / zsh).")
        sys.exit(1)

    video_url = _resolve_video_url(video_url)
    print(f"video_url = {video_url}")
    print(f"Loading {INPUT_FILE} (key {config.url_key(video_url)})...")

    topics = config.load_url_keyed_entry(INPUT_FILE, video_url)
    if not isinstance(topics, list) or not topics:
        print(f"ERROR: No topics found for this video_url in {INPUT_FILE}. "
              f"Run the segmentation stage for this video_url first.")
        sys.exit(1)

    print(f"Found {len(topics)} topics.")
    print(f"Using Groq model: {GROQ_MODEL}\n")

    client = Groq(api_key=api_key)

    ordered_topic_ids = [t["topic_id"] for t in topics]
    results_by_id = load_existing_results(OUTPUT_FILE, video_url) if RESUME else {}

    counts = {"important": 0, "filler": 0, "uncertain": 0}

    for i, topic in enumerate(topics, start=1):
        topic_id = topic["topic_id"]

        if RESUME and topic_id in results_by_id and is_valid_existing_entry(results_by_id[topic_id], topic):
            existing = results_by_id[topic_id]
            counts[existing["type"]] = counts.get(existing["type"], 0) + 1
            print(f"[{i}/{len(topics)}] Topic {topic_id} (skipped, already analyzed)")
            print(f"      Type: {existing['type']}")
            print(f"      Confidence: {existing['confidence']}")
            print(f"      Name: {existing['name']}\n")
            continue

        prev_result = results_by_id.get(ordered_topic_ids[i - 2]) if i >= 2 else None
        next_topic = topics[i] if i < len(topics) else None

        user_prompt = build_user_prompt(topic, prev_result, next_topic)

        print(f"[{i}/{len(topics)}] Topic {topic_id}")
        try:
            classification = call_groq_with_retries(client, user_prompt)
        except RuntimeError as e:
            # Do not lose already-processed topics: save what we have and stop.
            print(f"      FAILED: {e}")
            print("\nStopping so no work is lost. Already-analyzed topics are saved.")
            print(f"Rerun the script (RESUME={RESUME}) to continue from this topic.")
            save_results(OUTPUT_FILE, video_url, results_by_id, ordered_topic_ids)
            sys.exit(1)

        entry = {
            "topic_id": topic["topic_id"],
            "start_segment": topic.get("start_segment"),
            "end_segment": topic.get("end_segment"),
            "start_time": topic.get("start_time"),
            "end_time": topic.get("end_time"),
            "boundary_score": topic.get("boundary_score"),
            "type": classification.type,
            "confidence": round(classification.confidence, 3),
            "name": classification.name,
            "description": classification.description,
            "keywords": classification.keywords,
            "rationale": classification.rationale,
        }
        if INCLUDE_SEGMENTS:
            entry["segments"] = topic.get("segments", [])

        results_by_id[topic_id] = entry
        counts[classification.type] = counts.get(classification.type, 0) + 1

        # Save after every topic so a crash never loses completed work.
        save_results(OUTPUT_FILE, video_url, results_by_id, ordered_topic_ids)

        print(f"      Type: {entry['type']}")
        print(f"      Confidence: {entry['confidence']}")
        print(f"      Name: {entry['name']}\n")

    print("Finished.\n")
    print(f"Important: {counts.get('important', 0)}")
    print(f"Filler: {counts.get('filler', 0)}")
    print(f"Uncertain: {counts.get('uncertain', 0)}")
    print(f"\nSaved:\n    {OUTPUT_FILE}  (entry for key {config.url_key(video_url)})")


if __name__ == "__main__":
    main()