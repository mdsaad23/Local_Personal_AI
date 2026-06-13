"""
Ollama wrapper with async streaming, TTFT/TGS measurement, and context management.

All timing uses time.perf_counter() as per project convention.
TTFT = time to first token (measured at first streamed chunk).
TGS  = token generation speed = eval_count / eval_duration (from final chunk).
"""
from __future__ import annotations

import json
import logging
import time
from collections.abc import AsyncGenerator
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


async def stream_response(
    messages: list[dict[str, str]],
    model: str = PRODUCTION_MODEL,
    *,
    think: bool = False,
    image_b64: str | None = None,
) -> AsyncGenerator[tuple[str, dict[str, Any]], None]:
    """
    Async generator that streams chat tokens from Ollama.
    Yields (token, {}) for each token, then ("", metrics_dict) as the final item.

    Using httpx.AsyncClient keeps the event loop free during streaming —
    synchronous httpx.stream blocked the entire server.
    """
    options: dict[str, Any] = {
        "num_ctx": CONTEXT_LENGTH,
        "kv_cache_type": KV_CACHE_TYPE,
    }
    if think:
        options["think"] = True

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

    async with httpx.AsyncClient(timeout=300) as client:
        async with client.stream("POST", _GENERATE_URL, json=payload) as resp:
            resp.raise_for_status()
            async for line in resp.aiter_lines():
                if not line:
                    continue
                try:
                    chunk = json.loads(line)
                except Exception:
                    continue

                token = chunk.get("message", {}).get("content", "")

                if token and first_token:
                    metrics["ttft_s"] = time.perf_counter() - t_start
                    first_token = False

                if token:
                    yield token, {}

                if chunk.get("done"):
                    metrics["ttlc_s"] = time.perf_counter() - t_start
                    eval_count = chunk.get("eval_count", 0)
                    eval_duration_ns = chunk.get("eval_duration", 0)
                    metrics["eval_count"] = eval_count
                    metrics["prompt_eval_count"] = chunk.get("prompt_eval_count", 0)
                    if eval_duration_ns > 0:
                        metrics["tgs"] = round(eval_count / (eval_duration_ns / 1e9), 1)

                    # Fetch memory usage info from Ollama
                    try:
                        ps_resp = await client.get(f"{OLLAMA_BASE_URL}/api/ps")
                        if ps_resp.status_code == 200:
                            ps_data = ps_resp.json()
                            for m in ps_data.get("models", []):
                                if m.get("name") == model or m.get("model") == model:
                                    vram = m.get("size_vram", 0)
                                    total_size = m.get("size", 0)
                                    metrics["vram_bytes"] = vram
                                    metrics["ram_bytes"] = max(0, total_size - vram)
                                    break
                    except Exception as exc:
                        logger.warning(f"Failed to fetch model memory metrics: {exc}")

                    yield "", metrics
                    return


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
