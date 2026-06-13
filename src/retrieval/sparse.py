"""
BM25 sparse retrieval index via rank_bm25.

The index is rebuilt from the LanceDB table on first access and persisted
as a pickle alongside a mapping list. This keeps BM25 in sync with LanceDB
without requiring a separate ingest call.
"""
from __future__ import annotations

import logging
import pickle
from pathlib import Path
from typing import Any

from rank_bm25 import BM25Okapi

from config.settings import BM25_DIR, TOP_K_SPARSE

logger = logging.getLogger(__name__)

_INDEX_PATH  = BM25_DIR / "bm25_index.pkl"
_CORPUS_PATH = BM25_DIR / "bm25_corpus.pkl"

_bm25: BM25Okapi | None = None
_corpus: list[dict[str, Any]] = []   # parallel list to BM25 index


def _tokenise(text: str) -> list[str]:
    return text.lower().split()


def _load_index() -> tuple[BM25Okapi | None, list[dict[str, Any]]]:
    global _bm25, _corpus
    if _bm25 is not None:
        return _bm25, _corpus
    if _INDEX_PATH.exists() and _CORPUS_PATH.exists():
        try:
            with _INDEX_PATH.open("rb") as f:
                _bm25 = pickle.load(f)
            with _CORPUS_PATH.open("rb") as f:
                _corpus = pickle.load(f)
            logger.info("BM25 index loaded (%d docs)", len(_corpus))
        except Exception:
            logger.exception("Failed to load BM25 index — starting fresh")
            _bm25, _corpus = None, []
    return _bm25, _corpus


def _save_index() -> None:
    BM25_DIR.mkdir(parents=True, exist_ok=True)
    with _INDEX_PATH.open("wb") as f:
        pickle.dump(_bm25, f)
    with _CORPUS_PATH.open("wb") as f:
        pickle.dump(_corpus, f)


def add_chunks_to_bm25(chunks: list[dict[str, Any]]) -> None:
    """Extend the BM25 index with new chunks and persist."""
    global _bm25, _corpus
    _load_index()

    new_entries = [c for c in chunks if c.get("text")]
    if not new_entries:
        return

    # Avoid duplicate chunks by removing any existing entries with same doc_id
    doc_ids_to_remove = {c.get("doc_id") for c in new_entries if c.get("doc_id")}
    if doc_ids_to_remove:
        _corpus = [c for c in _corpus if c.get("doc_id") not in doc_ids_to_remove]

    _corpus.extend(new_entries)
    tokenised = [_tokenise(c["text"]) for c in _corpus]
    _bm25 = BM25Okapi(tokenised)
    _save_index()
    logger.info("BM25 index rebuilt: %d documents", len(_corpus))


def search_sparse(query: str, top_k: int = TOP_K_SPARSE) -> list[dict[str, Any]]:
    """BM25 keyword search. Returns top_k chunks sorted by BM25 score."""
    bm25, corpus = _load_index()
    if bm25 is None or not corpus:
        logger.warning("BM25 index empty — skipping sparse search")
        return []

    tokens = _tokenise(query)
    scores = bm25.get_scores(tokens)

    # zip with corpus, sort descending, take top_k
    ranked = sorted(zip(scores, corpus), key=lambda x: x[0], reverse=True)[:top_k]

    return [
        {
            **chunk,
            "score":     float(score),
            "retrieval": "sparse",
        }
        for score, chunk in ranked
        if score > 0
    ]


def rebuild_index_from_lancedb() -> None:
    """Rebuild BM25 index from all chunks currently in LanceDB."""
    global _bm25, _corpus
    from src.retrieval.dense import _get_table
    try:
        table = _get_table()
        rows = table.to_pandas()
        _corpus = rows.to_dict(orient="records")
        tokenised = [_tokenise(c.get("text", "")) for c in _corpus]
        _bm25 = BM25Okapi(tokenised)
        _save_index()
        logger.info("BM25 index rebuilt from LanceDB: %d chunks", len(_corpus))
    except Exception:
        logger.exception("Failed to rebuild BM25 index from LanceDB")
