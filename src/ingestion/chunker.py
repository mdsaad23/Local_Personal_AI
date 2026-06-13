"""
Semantic chunker with recursive fallback.

Strategy:
  1. Try semantic chunking — split where embedding similarity drops below threshold.
  2. If a resulting chunk exceeds CHUNK_SIZE, split recursively on paragraph/sentence boundaries.
  3. Discard chunks below MIN_CHUNK_SIZE.

Why semantic over fixed-size:
  Fixed-size splits break mid-concept, degrading retrieval precision by 15-30% on
  structured documents. Semantic splits preserve conceptual units at the cost of
  variable chunk size — which LanceDB handles natively.
"""

import logging
import re
from typing import Any

import numpy as np
import httpx

from config.settings import (
    CHUNK_OVERLAP,
    CHUNK_SIZE,
    EMBED_MODEL,
    MIN_CHUNK_SIZE,
    OLLAMA_BASE_URL,
)

logger = logging.getLogger(__name__)

_SENTENCE_RE = re.compile(r"(?<=[.!?])\s+")
_PARA_RE = re.compile(r"\n{2,}")


def chunk_pages(pages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """
    Chunk a list of parsed page dicts into smaller units.
    Preserves all metadata from the source page.
    """
    chunks: list[dict[str, Any]] = []
    for page in pages:
        page_chunks = _chunk_text(page["text"])
        for i, chunk_text in enumerate(page_chunks):
            chunks.append({
                **page,
                "text": chunk_text,
                "chunk_index": i,
                "chunk_count": len(page_chunks),
            })
    return chunks


def _chunk_text(text: str) -> list[str]:
    """Split text into chunks using semantic similarity with recursive fallback."""
    paragraphs = [p.strip() for p in _PARA_RE.split(text) if p.strip()]

    if len(paragraphs) <= 1:
        return _recursive_split(text)

    try:
        return _semantic_split(paragraphs)
    except Exception:
        logger.warning("Semantic chunking failed — falling back to recursive split")
        return _recursive_split(text)


def _semantic_split(paragraphs: list[str]) -> list[str]:
    """
    Embed each paragraph, compute cosine similarity between consecutive pairs,
    split at similarity valleys (topic shifts).
    """
    embeddings = _embed_batch(paragraphs)
    if not embeddings:
        return _recursive_split("\n\n".join(paragraphs))

    # Cosine similarity between consecutive paragraph embeddings
    similarities = []
    for i in range(len(embeddings) - 1):
        a = np.array(embeddings[i])
        b = np.array(embeddings[i + 1])
        sim = float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-10))
        similarities.append(sim)

    if not similarities:
        return ["\n\n".join(paragraphs)]

    # Split at positions where similarity drops below mean - 0.5*std
    mean_sim = float(np.mean(similarities))
    std_sim = float(np.std(similarities))
    threshold = mean_sim - 0.5 * std_sim

    split_points = {0}
    for i, sim in enumerate(similarities):
        if sim < threshold:
            split_points.add(i + 1)
    split_points.add(len(paragraphs))

    split_points_sorted = sorted(split_points)
    raw_chunks: list[str] = []
    for start, end in zip(split_points_sorted, split_points_sorted[1:]):
        segment = "\n\n".join(paragraphs[start:end]).strip()
        if segment:
            raw_chunks.append(segment)

    # Each semantic chunk may still exceed CHUNK_SIZE — split those recursively
    final: list[str] = []
    for chunk in raw_chunks:
        word_count = len(chunk.split())
        if word_count > CHUNK_SIZE:
            final.extend(_recursive_split(chunk))
        elif word_count >= MIN_CHUNK_SIZE:
            final.append(chunk)

    return final if final else ["\n\n".join(paragraphs)]


def _recursive_split(text: str) -> list[str]:
    """
    Split on paragraphs → sentences → words until each piece fits in CHUNK_SIZE.
    Adds CHUNK_OVERLAP token overlap between consecutive chunks for context continuity.
    """
    words = text.split()
    if len(words) <= CHUNK_SIZE:
        # Index the whole thing as one chunk. The MIN_CHUNK_SIZE guard is meant to
        # drop tiny *fragments* produced while splitting a larger document — not to
        # reject an entire short document. Applying it here silently dropped every
        # single-block upload under MIN_CHUNK_SIZE words, so the ingest failed with
        # "no indexable content". Keep any non-empty document.
        return [text] if text.strip() else []

    chunks: list[str] = []
    start = 0
    while start < len(words):
        end = min(start + CHUNK_SIZE, len(words))
        chunk = " ".join(words[start:end])
        if len(words[start:end]) >= MIN_CHUNK_SIZE:
            chunks.append(chunk)
        if end >= len(words):
            break
        start = end - CHUNK_OVERLAP  # overlap between chunks
        if start <= 0:
            break

    return chunks


def _embed_batch(texts: list[str]) -> list[list[float]]:
    """Embed a list of texts via nomic-embed-text using Ollama's batch endpoint."""
    embeddings: list[list[float]] = []
    # Ollama embedding API is single-text — batch with sequential calls
    for text in texts:
        try:
            resp = httpx.post(
                f"{OLLAMA_BASE_URL}/api/embed",
                json={"model": EMBED_MODEL, "input": text},
                timeout=30,
            )
            resp.raise_for_status()
            data = resp.json()
            emb = data.get("embeddings", [data.get("embedding", [])])
            if isinstance(emb[0], list):
                embeddings.append(emb[0])
            else:
                embeddings.append(emb)
        except Exception as exc:
            logger.warning("Embedding failed for semantic chunking — falling back to recursive split: %s", exc)
            raise
    return embeddings
