"""
config.py
---------
Single source of truth for paths and settings. Every other file imports
from here instead of hardcoding paths -- this is exactly what fixes the
"ChromaDB path mismatch between modules" problem you hit before.
"""

import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()  # reads .env and puts variables into os.environ

# --- API keys ---
GROQ_API_KEY = os.environ.get("GROQ_API_KEY")
if not GROQ_API_KEY:
    raise RuntimeError(
        "GROQ_API_KEY not found. Copy .env.example to .env and add your key."
    )

# --- Paths (all relative to this file, so it works on any machine) ---
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
VIDEO_DIR = DATA_DIR / "videos"
AUDIO_DIR = DATA_DIR / "audio"
TRANSCRIPT_DIR = DATA_DIR / "transcripts"
CHROMA_DIR = DATA_DIR / "chroma_db"

for d in (VIDEO_DIR, AUDIO_DIR, TRANSCRIPT_DIR, CHROMA_DIR):
    d.mkdir(parents=True, exist_ok=True)

# --- Model settings ---
GROQ_MODEL = "openai/gpt-oss-120b"          # main LLM: answers + timestamps
EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"      # sentence-transformers model for RAG

# --- Audio/video sync correction ---
MAX_STRETCH_DRIFT = 0.03  # 3%

# --- RAG chunk merging (duration + gap based, not fixed window/stride) ---
SEGMENT_MIN_SECONDS = 20.0
SEGMENT_MAX_GAP_SECONDS = 10.0

# --- AV summary settings ---
TARGET_SUMMARY_MIN_MINUTES = 20
TARGET_SUMMARY_MAX_MINUTES = 25
FILLER_WORDS = {"um", "uh", "umm", "uhh", "erm", "hmm"}
MIN_SILENCE_LEN_MS = 700        # pause longer than this = candidate for trimming/splitting (pydub backend)
SILENCE_THRESH_DB = -40         # quieter than this = "silence" (pydub backend)

# NEW: threshold for the whisper_gaps silence-removal backend. This
# operates on word-level timestamps we already have from transcribe.py,
# not raw audio -- a "silence" here is just a gap between the end of
# one recognized word and the start of the next that's at least this
# long. Kept separate from MIN_SILENCE_LEN_MS since the two backends
# measure fundamentally different things (transcript gaps vs raw
# amplitude) and shouldn't be forced to share one threshold.
SILENCE_WORD_GAP_SECONDS = 1.0