"""
config.py — Central configuration for SHL Assessment Agent
"""
import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# ── Paths ──────────────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).parent
CATALOG_PATH = BASE_DIR / "shl_product_catalog.json"
EMBEDDINGS_CACHE_PATH = BASE_DIR / ".embeddings_cache.npy"
INDEX_CACHE_PATH = BASE_DIR / ".faiss_index.bin"

# ── LLM / Embedding ────────────────────────────────────────────────────────
GROQ_API_KEY: str = os.getenv("GROQ_API_KEY", "")
GROQ_MODEL: str = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")

# ── Retrieval ──────────────────────────────────────────────────────────────
SIMILARITY_TOP_K: int = 25          # Fetch more from FAISS, then filter/rank
MAX_RECOMMENDATIONS: int = 10       # Hard cap per API spec
MIN_RECOMMENDATIONS: int = 1

# ── Conversation ───────────────────────────────────────────────────────────
MAX_TURNS: int = 8                  # Hard cap from assignment
# We must recommend by this turn at the latest (leaving turn 8 as fallback)
RECOMMEND_BY_TURN: int = 7

# ── Server ─────────────────────────────────────────────────────────────────
HOST: str = os.getenv("HOST", "0.0.0.0")
PORT: int = int(os.getenv("PORT", "8000"))
LLM_TIMEOUT_SECONDS: float = 29.5   # Leave 0.5s buffer for grader's hard network cutoff

# ── Test Type Code Mapping (keys → single letter codes) ───────────────────
# Derived from sample conversations (C1-C10)
KEY_TO_CODE: dict[str, str] = {
    "Ability & Aptitude": "A",
    "Assessment Exercises": "E",
    "Biodata & Situational Judgment": "B",
    "Competencies": "C",
    "Development & 360": "D",
    "Knowledge & Skills": "K",
    "Personality & Behavior": "P",
    "Simulations": "S",
}

CODE_TO_KEY: dict[str, str] = {v: k for k, v in KEY_TO_CODE.items()}

# ── Job Level Normalization ────────────────────────────────────────────────
# Maps common user terms → catalog job level names
JOB_LEVEL_ALIASES: dict[str, list[str]] = {
    "entry": ["Entry-Level", "Graduate"],
    "junior": ["Entry-Level", "Graduate"],
    "graduate": ["Graduate"],
    "mid": ["Mid-Professional", "Professional Individual Contributor"],
    "senior": ["Professional Individual Contributor", "Mid-Professional"],
    "manager": ["Manager", "Front Line Manager"],
    "frontline": ["Front Line Manager", "Supervisor"],
    "supervisor": ["Supervisor"],
    "director": ["Director"],
    "executive": ["Executive", "Director"],
    "cxo": ["Executive", "Director"],
    "leadership": ["Director", "Executive", "Manager"],
    "general": ["General Population"],
}

# ── Language Normalization ─────────────────────────────────────────────────
LANGUAGE_ALIASES: dict[str, str] = {
    "english": "English (USA)",
    "english us": "English (USA)",
    "english usa": "English (USA)",
    "english uk": "English International",
    "english international": "English International",
    "spanish": "Spanish",
    "latin american spanish": "Latin American Spanish",
    "french": "French",
    "german": "German",
    "chinese": "Chinese Simplified",
    "chinese simplified": "Chinese Simplified",
    "chinese traditional": "Chinese Traditional",
    "japanese": "Japanese",
    "portuguese": "Portuguese",
    "portuguese brazil": "Portuguese (Brazil)",
    "dutch": "Dutch",
    "arabic": "Arabic",
    "korean": "Korean",
    "italian": "Italian",
    "russian": "Russian",
    "turkish": "Turkish",
}
