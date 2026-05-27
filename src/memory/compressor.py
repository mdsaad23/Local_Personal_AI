"""
Conversation compressor — fires when context usage hits 80% of CONTEXT_LENGTH.

Summarises the oldest turns into a compact narrative, replacing them with the
summary. This allows indefinitely long conversations without truncation.
"""
from __future__ import annotations

import logging
from typing import Any

import httpx

from config.settings import (
    CONTEXT_COMPRESSION_THRESHOLD,
    CONTEXT_LENGTH,
    OLLAMA_BASE_URL,
    PRODUCTION_MODEL,
)

logger = logging.getLogger(__name__)

_COMPRESS_PROMPT = """\
Summarise the following conversation excerpt into a compact paragraph.
Preserve: decisions made, facts stated, user preferences, action items.
Discard: greetings, filler, and repeated content.
Write in third-person past tense. Max 200 words.

CONVERSATION:
{conversation}

SUMMARY:"""

# Rough token estimate: 1 token ≈ 4 characters
_CHARS_PER_TOKEN = 4


def _estimate_tokens(messages: list[dict[str, str]]) -> int:
    total = sum(len(m.get("content", "")) for m in messages)
    return total // _CHARS_PER_TOKEN


def should_compress(messages: list[dict[str, str]]) -> bool:
    estimated = _estimate_tokens(messages)
    threshold = int(CONTEXT_LENGTH * CONTEXT_COMPRESSION_THRESHOLD)
    return estimated >= threshold


def compress(messages: list[dict[str, str]]) -> list[dict[str, str]]:
    """
    Compress a message list that has exceeded the context threshold.
    Keeps the last 4 turns intact (most recent context), summarises the rest.
    Returns a new message list: [system summary] + [last 4 turns].
    """
    if len(messages) <= 4:
        return messages

    keep_tail = 4
    to_compress = messages[:-keep_tail]
    tail = messages[-keep_tail:]

    convo_text = "\n".join(
        f"{m['role'].upper()}: {m['content']}" for m in to_compress
    )
    prompt = _COMPRESS_PROMPT.format(conversation=convo_text)

    try:
        resp = httpx.post(
            f"{OLLAMA_BASE_URL}/api/generate",
            json={
                "model": PRODUCTION_MODEL,
                "prompt": prompt,
                "stream": False,
                "options": {"temperature": 0, "num_predict": 300},
            },
            timeout=60,
        )
        resp.raise_for_status()
        summary = resp.json().get("response", "").strip()
    except Exception:
        logger.exception("Compression failed — keeping last %d messages", keep_tail * 2)
        return messages[-(keep_tail * 2):]

    summary_msg = {
        "role": "system",
        "content": f"[Conversation summary — earlier context compressed]\n{summary}",
    }
    logger.info("Compressed %d messages into summary", len(to_compress))
    return [summary_msg] + tail
