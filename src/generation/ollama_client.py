"""
Ollama wrapper with streaming, TTFT/TGS measurement, and context management.

All timing uses time.perf_counter() as per project convention.
TTFT = time to first token (measured at first streamed chunk).
TGS  = token generation speed = eval_count / eval_duration (from final chunk).
TTLC = time to last completion = total wall time from request to last token.
"""
from __future__ import annotations

import logging
import time
from collections.abc import Generator
from typing import Any

import httpx

from config.settings import (
    CONTEXT_LENGTH,
    KV_CACHE_TYPE,
    OLLAMA_BASE_URL,
    PRODUCTION_MODEL,
)

logger = logging.getLogger(__name__)

_GENERATE_URL = f"{OLLAMA_BASE_URL}/api/chat"


def stream_response(
    messages: list[dict[str, str]],
    model: str = PRODUCTION_MODEL,
    *,
    think: bool = False,
    image_b64: str | None = None,
) -> Generator[str, None, dict[str, Any]]:
    """
    Stream a chat response. Yields text tokens as they arrive.
    On completion, the generator return value contains timing metrics
    (access via StopIteration.value or send protocol).

    Args:
        messages: OpenAI-format message list [{role, content}, ...]
        model: Ollama model ID
        think: Enable Qwen 3 thinking mode for complex reasoning queries
        image_b64: Base64-encoded image to attach to the last user message

    Yields:
        str — each streamed text token

    Returns (via StopIteration.value):
        dict with keys: ttft_s, tgs, ttlc_s, eval_count, prompt_eval_count
    """
    options: dict[str, Any] = {
        "num_ctx": CONTEXT_LENGTH,
        "kv_cache_type": KV_CACHE_TYPE,
    }
    if think:
        options["think"] = True

    # Attach image to the last user message if provided
    messages_out = [dict(m) for m in messages]
    if image_b64:
        for msg in reversed(messages_out):
            if msg.get("role") == "user":
                msg["images"] = [image_b64]
                break

    payload = {
        "model": model,
        "messages": messages_out,
        "stream": True,
        "options": options,
    }

    metrics: dict[str, Any] = {
        "ttft_s": None,
        "tgs": None,
        "ttlc_s": None,
        "eval_count": 0,
        "prompt_eval_count": 0,
    }

    t_start = time.perf_counter()
    first_token = True

    try:
        with httpx.stream(
            "POST",
            _GENERATE_URL,
            json=payload,
            timeout=300,
        ) as resp:
            resp.raise_for_status()
            for line in resp.iter_lines():
                if not line:
                    continue
                import json
                try:
                    chunk = json.loads(line)
                except Exception:
                    continue

                msg = chunk.get("message", {})
                token = msg.get("content", "")

                if token and first_token:
                    metrics["ttft_s"] = time.perf_counter() - t_start
                    first_token = False

                if token:
                    yield token

                if chunk.get("done"):
                    metrics["ttlc_s"] = time.perf_counter() - t_start
                    eval_count = chunk.get("eval_count", 0)
                    eval_duration_ns = chunk.get("eval_duration", 0)
                    metrics["eval_count"] = eval_count
                    metrics["prompt_eval_count"] = chunk.get("prompt_eval_count", 0)
                    if eval_duration_ns > 0:
                        metrics["tgs"] = eval_count / (eval_duration_ns / 1e9)
                    break

    except httpx.HTTPStatusError as e:
        logger.error("Ollama HTTP error: %s — %s", e.response.status_code, e.response.text[:200])
        raise
    except Exception:
        logger.exception("Ollama stream failed")
        raise

    return metrics


def generate_sync(
    messages: list[dict[str, str]],
    model: str = PRODUCTION_MODEL,
    max_tokens: int = 1024,
    temperature: float = 0.0,
) -> str:
    """Non-streaming generation for internal calls (HyDE, relationship extraction)."""
    payload = {
        "model": model,
        "messages": messages,
        "stream": False,
        "options": {
            "num_ctx": CONTEXT_LENGTH,
            "num_predict": max_tokens,
            "temperature": temperature,
        },
    }
    resp = httpx.post(_GENERATE_URL, json=payload, timeout=120)
    resp.raise_for_status()
    return resp.json().get("message", {}).get("content", "")


def list_local_models() -> list[str]:
    """Return model IDs available in the local Ollama instance."""
    try:
        resp = httpx.get(f"{OLLAMA_BASE_URL}/api/tags", timeout=10)
        resp.raise_for_status()
        return [m["name"] for m in resp.json().get("models", [])]
    except Exception:
        logger.warning("Could not reach Ollama to list models")
        return []


def check_ollama_health() -> bool:
    try:
        resp = httpx.get(f"{OLLAMA_BASE_URL}/", timeout=5)
        return resp.status_code == 200
    except Exception:
        return False
