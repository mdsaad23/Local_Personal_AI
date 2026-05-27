"""
HyDE — Hypothetical Document Embeddings.

For hard queries where the query itself has low lexical/semantic overlap with
the answer, generate a hypothetical answer first, embed that, and use it for
dense retrieval instead of the raw query.

When to use: query classifier sets hyde=True for analytical/reasoning queries.
Skip for factual lookups — HyDE adds ~1s latency from the generation call.
"""
from __future__ import annotations

import logging

import httpx

from config.settings import OLLAMA_BASE_URL, PRODUCTION_MODEL

logger = logging.getLogger(__name__)

_HYDE_PROMPT = """\
Write a short passage (2-4 sentences) that would directly answer the following question.
Write as if you are the author of a document that contains this information.
Do not say "I don't know" — write a plausible, specific answer.

Question: {query}

Passage:"""


def generate_hypothetical_document(query: str) -> str:
    """
    Generate a short hypothetical answer passage for the query.
    Returns the original query if generation fails.
    """
    prompt = _HYDE_PROMPT.format(query=query)
    try:
        resp = httpx.post(
            f"{OLLAMA_BASE_URL}/api/generate",
            json={
                "model": PRODUCTION_MODEL,
                "prompt": prompt,
                "stream": False,
                "options": {"temperature": 0.3, "num_predict": 150},
            },
            timeout=30,
        )
        resp.raise_for_status()
        hyp = resp.json().get("response", "").strip()
        if hyp:
            logger.debug("HyDE generated: %s …", hyp[:80])
            return hyp
    except Exception:
        logger.debug("HyDE generation failed — using raw query", exc_info=True)
    return query
