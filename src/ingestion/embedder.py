"""
Batch embedding via nomic-embed-text running locally through Ollama.
No cloud API — every embedding call stays on-device.

nomic-embed-text produces 768-dimensional vectors and outperforms
OpenAI text-embedding-ada-002 on most retrieval benchmarks at zero API cost.
"""

import logging
import time
from typing import Any

import httpx

from config.settings import EMBED_MODEL, OLLAMA_BASE_URL

logger = logging.getLogger(__name__)

_EMBED_URL = f"{OLLAMA_BASE_URL}/api/embed"
_BATCH_SIZE = 16   # Ollama embed processes one text at a time; this limits queue depth


def embed_chunks(chunks: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """
    Add an 'embedding' key to each chunk dict.
    Chunks without embeddings are dropped (logged as warnings).
    Returns only successfully embedded chunks.
    """
    texts = [c["text"] for c in chunks]
    embeddings = embed_texts(texts)

    result = []
    for chunk, emb in zip(chunks, embeddings):
        if emb:
            result.append({**chunk, "embedding": emb})
        else:
            logger.warning("Skipping chunk from %s — embedding failed", chunk.get("source"))
    return result


def embed_texts(texts: list[str]) -> list[list[float] | None]:
    """
    Embed a list of raw strings. Returns parallel list of embeddings.
    Failed embeddings are returned as None.
    """
    results: list[list[float] | None] = []
    for i, text in enumerate(texts):
        if i > 0 and i % _BATCH_SIZE == 0:
            logger.debug("Embedded %d/%d texts", i, len(texts))
        emb = _embed_single(text)
        results.append(emb)
    return results


def embed_query(text: str) -> list[float]:
    """
    Embed a single query string. Raises RuntimeError if embedding fails —
    the query pipeline cannot proceed without a valid query embedding.
    """
    emb = _embed_single(text)
    if emb is None:
        raise RuntimeError(f"Failed to embed query: {text[:80]!r}")
    return emb


def _embed_single(text: str, retries: int = 3) -> list[float] | None:
    text = text.strip()
    if not text:
        return None

    for attempt in range(retries):
        try:
            resp = httpx.post(
                _EMBED_URL,
                json={"model": EMBED_MODEL, "input": text},
                timeout=30,
            )
            resp.raise_for_status()
            data = resp.json()

            # Ollama returns {"embeddings": [[...]]} for /api/embed
            emb = data.get("embeddings")
            if emb and isinstance(emb[0], list):
                return emb[0]

            # Fallback: older Ollama versions return {"embedding": [...]}
            emb = data.get("embedding")
            if emb:
                return emb

            logger.warning("Unexpected embedding response shape: %s", list(data.keys()))
            return None

        except httpx.TimeoutException:
            wait = 2 ** attempt
            logger.warning("Embed timeout (attempt %d/%d) — retrying in %ds", attempt + 1, retries, wait)
            time.sleep(wait)
        except Exception:
            logger.exception("Embed request failed (attempt %d/%d)", attempt + 1, retries)
            if attempt == retries - 1:
                return None
            time.sleep(1)

    return None
