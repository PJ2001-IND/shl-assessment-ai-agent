"""
catalog.py — Catalog loading, normalization, embedding, and FAISS search

Responsibilities:
  1. Load shl_product_catalog.json → list[AssessmentRecord]
  2. Generate/cache offline SentenceTransformers embeddings for all 377 assessments
  3. Build FAISS index for semantic search
  4. Provide search + metadata-filter functions
"""
from __future__ import annotations

import json
import logging
import pickle
import re
import time
from pathlib import Path
from typing import Optional

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

from config import (
    CATALOG_PATH,
    EMBEDDINGS_CACHE_PATH,
    KEY_TO_CODE,
    SIMILARITY_TOP_K,
)
from models import AssessmentRecord

logger = logging.getLogger(__name__)


# ── Duration Parsing ───────────────────────────────────────────────────────

def _parse_duration_minutes(duration_raw: str) -> Optional[int]:
    """Extract integer minutes from raw duration string."""
    if not duration_raw:
        return None
    match = re.search(r"(\d+)", duration_raw)
    if match:
        return int(match.group(1))
    return None


# ── Test Type Code Derivation ──────────────────────────────────────────────

def _derive_test_type_codes(keys: list[str]) -> str:
    """Convert list of key strings to comma-separated codes like 'K,S' or 'A'."""
    codes = []
    for key in keys:
        code = KEY_TO_CODE.get(key)
        if code and code not in codes:
            codes.append(code)
    return ",".join(codes) if codes else "K"  # default to K if unknown


# ── Searchable Text Builder ────────────────────────────────────────────────

def _build_searchable_text(record: dict) -> str:
    """
    Build a rich text string for embedding. Combines all fields so
    semantic search can match on role, skill, domain, language, etc.
    """
    name = record.get("name", "")
    desc = record.get("description", "")
    levels = ", ".join(record.get("job_levels", []))
    langs = ", ".join(record.get("languages", []))
    keys = ", ".join(record.get("keys", []))
    duration = record.get("duration", "")
    adaptive = "adaptive" if record.get("adaptive") == "yes" else "non-adaptive"

    return (
        f"{name}. "
        f"{desc} "
        f"Job levels: {levels}. "
        f"Languages: {langs}. "
        f"Test types: {keys}. "
        f"Duration: {duration}. "
        f"{adaptive}."
    ).strip()


# ── Record Normalization ───────────────────────────────────────────────────

def _normalize_record(raw: dict) -> AssessmentRecord:
    """Convert raw JSON dict → AssessmentRecord."""
    keys = raw.get("keys", [])
    duration_raw = raw.get("duration_raw", "")
    duration = raw.get("duration", "")

    return AssessmentRecord(
        entity_id=str(raw.get("entity_id", "")),
        name=raw.get("name", ""),
        url=raw.get("link", ""),
        description=raw.get("description", ""),
        job_levels=raw.get("job_levels", []),
        languages=raw.get("languages", []),
        duration=duration,
        duration_minutes=_parse_duration_minutes(duration_raw or duration),
        remote=raw.get("remote", "yes") == "yes",
        adaptive=raw.get("adaptive", "no") == "yes",
        keys=keys,
        test_type_codes=_derive_test_type_codes(keys),
        searchable_text=_build_searchable_text(raw),
    )


# ── Embedding Helpers ──────────────────────────────────────────────────────

_model = None

def _get_model() -> SentenceTransformer:
    global _model
    if _model is None:
        logger.info("Initializing local SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')...")
        _model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
    return _model


def _embed_texts(texts: list[str]) -> np.ndarray:
    """
    Generate embeddings for a list of texts using local SentenceTransformer.
    Runs 100% offline and extremely fast.
    """
    model = _get_model()
    embeddings = model.encode(texts, show_progress_bar=False, convert_to_numpy=True)
    return embeddings.astype("float32")


def _embed_query(query: str) -> np.ndarray:
    """Generate embedding for a single search query locally."""
    model = _get_model()
    vec = model.encode([query], show_progress_bar=False, convert_to_numpy=True)
    return vec.astype("float32").reshape(1, -1)


def _normalize_vectors(vecs: np.ndarray) -> np.ndarray:
    """L2-normalize vectors for cosine similarity via inner product."""
    norms = np.linalg.norm(vecs, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1e-10, norms)  # prevent division by zero
    return vecs / norms


# ── FAISS Index Builder ────────────────────────────────────────────────────

def _build_faiss_index(embeddings: np.ndarray) -> faiss.IndexFlatIP:
    """Build a FAISS inner-product index from normalized embeddings."""
    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings)
    logger.info(f"Built FAISS index with {index.ntotal} vectors (dim={dim}).")
    return index


# ── Catalog Singleton ──────────────────────────────────────────────────────

class CatalogIndex:
    """
    Singleton that holds the loaded catalog + FAISS index in memory.
    Built once at startup, reused for all requests.
    """

    def __init__(self):
        self.records: list[AssessmentRecord] = []
        self.index: faiss.IndexFlatIP | None = None
        self.embeddings: np.ndarray | None = None
        self._name_map: dict[str, AssessmentRecord] = {}  # name (lower) → record
        self._url_map: dict[str, AssessmentRecord] = {}   # url → record

    def load(self) -> None:
        """
        Load catalog from JSON, build/load embeddings, build FAISS index.
        Uses disk cache for embeddings to avoid re-generating on each restart.
        """
        # 1. Load and normalize catalog
        logger.info(f"Loading catalog from {CATALOG_PATH}...")
        with open(CATALOG_PATH, "r", encoding="utf-8") as f:
            raw_data = json.load(f)

        self.records = [_normalize_record(r) for r in raw_data]
        logger.info(f"Loaded {len(self.records)} assessments.")

        # Build lookup maps
        self._name_map = {r.name.lower(): r for r in self.records}
        self._url_map = {r.url: r for r in self.records}

        # 2. Load or generate embeddings
        cache_path = Path(EMBEDDINGS_CACHE_PATH)
        if cache_path.exists():
            logger.info("Loading embeddings from cache...")
            self.embeddings = np.load(str(cache_path))
        else:
            logger.info("Generating embeddings (first-time setup)...")
            texts = [r.searchable_text for r in self.records]
            raw_embeddings = _embed_texts(texts)
            self.embeddings = _normalize_vectors(raw_embeddings)
            np.save(str(cache_path), self.embeddings)
            logger.info(f"Embeddings cached to {cache_path}.")

        # 3. Build FAISS index
        self.index = _build_faiss_index(self.embeddings)

    # ── Lookup helpers ─────────────────────────────────────────────────────

    def find_by_name(self, name: str) -> AssessmentRecord | None:
        """Case-insensitive exact name lookup."""
        return self._name_map.get(name.lower())

    def find_by_url(self, url: str) -> AssessmentRecord | None:
        """Exact URL lookup."""
        return self._url_map.get(url)

    def find_by_name_fuzzy(self, name: str, threshold: float = 0.75) -> AssessmentRecord | None:
        """
        Fuzzy name match — returns closest record if similarity >= threshold.
        Used for validating LLM-generated recommendation names.
        """
        name_lower = name.lower().strip()

        # Exact match first
        if name_lower in self._name_map:
            return self._name_map[name_lower]

        # Substring match
        for record_name, record in self._name_map.items():
            if name_lower in record_name or record_name in name_lower:
                return record

        # Token overlap match
        name_tokens = set(name_lower.split())
        best_score = 0.0
        best_record = None
        for record_name, record in self._name_map.items():
            record_tokens = set(record_name.split())
            if not record_tokens:
                continue
            overlap = len(name_tokens & record_tokens) / len(name_tokens | record_tokens)
            if overlap > best_score:
                best_score = overlap
                best_record = record

        if best_score >= threshold:
            return best_record
        return None

    # ── Semantic Search ────────────────────────────────────────────────────

    def semantic_search(self, query: str, top_k: int = SIMILARITY_TOP_K) -> list[tuple[AssessmentRecord, float]]:
        """
        Embed the query and search FAISS.
        Returns list of (AssessmentRecord, similarity_score) tuples, sorted desc.
        """
        if self.index is None:
            raise RuntimeError("Catalog index not loaded. Call load() first.")

        query_vec = _embed_query(query)
        query_vec = _normalize_vectors(query_vec)

        k = min(top_k, len(self.records))
        scores, indices = self.index.search(query_vec, k)

        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx >= 0:
                results.append((self.records[idx], float(score)))

        return results

    # ── Metadata Filtering ─────────────────────────────────────────────────

    def filter_by_metadata(
        self,
        candidates: list[tuple[AssessmentRecord, float]],
        job_levels: list[str] | None = None,
        languages: list[str] | None = None,
        test_type_codes: list[str] | None = None,
        max_duration_minutes: int | None = None,
        adaptive_only: bool = False,
    ) -> list[tuple[AssessmentRecord, float]]:
        """
        Apply hard metadata filters to a list of (record, score) tuples.
        Returns filtered list (score preserved for downstream ranking).
        """
        filtered = []

        for record, score in candidates:
            # Job level filter (soft — at least one level must match)
            if job_levels:
                record_levels_lower = {lvl.lower() for lvl in record.job_levels}
                query_levels_lower = {lvl.lower() for lvl in job_levels}
                if not record_levels_lower & query_levels_lower:
                    continue

            # Language filter (soft — at least one language must match)
            if languages:
                record_langs_lower = {lang.lower() for lang in record.languages}
                query_langs_lower = {lang.lower() for lang in languages}
                if not record_langs_lower & query_langs_lower:
                    continue

            # Test type filter (soft — at least one type must match)
            if test_type_codes:
                record_codes = set(record.test_type_codes.split(","))
                query_codes = set(test_type_codes)
                if not record_codes & query_codes:
                    continue

            # Duration filter (hard — must be within limit if specified)
            if max_duration_minutes and record.duration_minutes:
                if record.duration_minutes > max_duration_minutes:
                    continue

            # Adaptive filter
            if adaptive_only and not record.adaptive:
                continue

            filtered.append((record, score))

        return filtered

    # ── Combined Search ────────────────────────────────────────────────────

    def search(
        self,
        query: str,
        job_levels: list[str] | None = None,
        languages: list[str] | None = None,
        test_type_codes: list[str] | None = None,
        max_duration_minutes: int | None = None,
        top_k: int = SIMILARITY_TOP_K,
    ) -> list[AssessmentRecord]:
        """
        Full pipeline: semantic search → metadata filter → return top_k records.
        """
        # Step 1: Semantic retrieval (retrieve more than needed for filtering headroom)
        candidates = self.semantic_search(query, top_k=top_k * 2)

        # Step 2: Metadata filter (only if constraints specified)
        if any([job_levels, languages, test_type_codes, max_duration_minutes]):
            candidates = self.filter_by_metadata(
                candidates,
                job_levels=job_levels,
                languages=languages,
                test_type_codes=test_type_codes,
                max_duration_minutes=max_duration_minutes,
            )

        # Step 3: Return top_k by similarity score
        return [record for record, _ in candidates[:top_k]]

    # ── Validation ─────────────────────────────────────────────────────────

    def validate_recommendation_name(self, name: str) -> AssessmentRecord | None:
        """
        Validate that a recommendation name maps to a real catalog entry.
        Returns the matched record (with corrected name/URL) or None if invalid.
        """
        return self.find_by_name_fuzzy(name)

    def get_all_records(self) -> list[AssessmentRecord]:
        return self.records


# ── Module-level singleton ─────────────────────────────────────────────────
# Created at module import; loaded at application startup via catalog_index.load()
catalog_index = CatalogIndex()
