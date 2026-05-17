"""
generate_embeddings.py — Pre-generate and cache FAISS embeddings locally.

Run this ONCE before starting the server or building Docker:
    .venv/bin/python generate_embeddings.py

Takes ~4 minutes (377 items at ~92/min on free tier).
Generates .embeddings_cache.npy which is loaded instantly at server startup.
Include this file in your Docker image / Render deployment so the server
never needs to call the embedding API during cold start.
"""
import sys
import time
import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
)
logger = logging.getLogger(__name__)


def main():
    logger.info("=" * 60)
    logger.info("SHL Embedding Pre-Generator")
    logger.info("=" * 60)

    # Check for API key
    from config import GEMINI_API_KEY, EMBEDDINGS_CACHE_PATH
    if not GEMINI_API_KEY or GEMINI_API_KEY == "your_gemini_api_key_here":
        logger.error("GEMINI_API_KEY not set in .env — aborting.")
        sys.exit(1)

    # Check if cache already exists
    if EMBEDDINGS_CACHE_PATH.exists():
        logger.info(f"Cache already exists at {EMBEDDINGS_CACHE_PATH}")
        logger.info("Delete it and re-run to regenerate. Exiting.")
        sys.exit(0)

    logger.info("Loading catalog...")
    from catalog import catalog_index
    import json
    import numpy as np

    with open("shl_product_catalog.json") as f:
        raw_data = json.load(f)

    from catalog import _normalize_record, _build_searchable_text, _embed_texts, _normalize_vectors, _build_faiss_index
    records = [_normalize_record(r) for r in raw_data]
    texts = [r.searchable_text for r in records]

    logger.info(f"Generating embeddings for {len(texts)} assessments...")
    logger.info("Free tier: ~92 items/min → estimated ~4 minutes. Please wait...")
    logger.info("-" * 60)

    t0 = time.time()
    raw_embeddings = _embed_texts(texts)
    normalized = _normalize_vectors(raw_embeddings)

    np.save(str(EMBEDDINGS_CACHE_PATH), normalized)
    elapsed = time.time() - t0

    logger.info("-" * 60)
    logger.info(f"Done in {elapsed:.0f}s ({elapsed/60:.1f} min)")
    logger.info(f"Saved {len(texts)} embeddings → {EMBEDDINGS_CACHE_PATH}")
    logger.info(f"Shape: {normalized.shape}")
    logger.info("")
    logger.info("Next step: start the server — it will load cache instantly.")
    logger.info("  .venv/bin/uvicorn app:app --host 0.0.0.0 --port 8000")


if __name__ == "__main__":
    main()
