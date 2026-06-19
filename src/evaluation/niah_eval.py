"""
Needle in a Haystack (NIAH) evaluation.

Tests raw model context window recall — bypasses the RAG pipeline entirely.
A unique fact (the "needle") is buried in filler text at varying depths and
context lengths. The model must retrieve it verbatim.

Matrix:
  - Context sizes: 1k, 4k, 8k, 16k, 32k tokens
  - Needle positions: 10% / 50% / 90% into the context
  - 15 cells per model
"""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass, asdict

import httpx

from config.settings import OLLAMA_BASE_URL

logger = logging.getLogger(__name__)

# ── Needle config ───────────────────────────────────────────────────────────────
NEEDLE       = "The secret verification passphrase for this benchmark is: TANGERINE-9847-FALCON."
NEEDLE_Q     = "What is the secret verification passphrase mentioned in the document? Reproduce it exactly."
NEEDLE_ANS   = "TANGERINE-9847-FALCON"

# ── Test matrix ────────────────────────────────────────────────────────────────
CONTEXT_SIZES_K = [1, 4, 8, 16, 32]   # approximate token targets (×1000)
POSITIONS       = [0.1, 0.5, 0.9]     # fraction into haystack where needle sits

# ── Filler text ────────────────────────────────────────────────────────────────
# Domain-neutral paragraphs — deliberately unrelated to the needle phrase.
_FILLER = (
    "The history of computing stretches back to mechanical calculators invented in "
    "the seventeenth century. Charles Babbage designed the Difference Engine in 1822, "
    "a mechanical device capable of computing polynomial functions. Ada Lovelace, "
    "collaborating with Babbage, wrote what is now regarded as the first algorithm "
    "intended to be processed by a machine. The twentieth century brought electronic "
    "computing with the ENIAC, completed in 1945, which filled an entire room and "
    "consumed 150 kilowatts of power. Transistors replaced vacuum tubes in the 1950s, "
    "dramatically reducing size and power consumption. Integrated circuits followed, "
    "then microprocessors, enabling the personal computer revolution of the 1980s. "
    "The internet connected these machines globally, and mobile computing brought "
    "powerful processors into every pocket. Graphics processing units, originally "
    "designed for rendering polygons in video games, proved ideal for the matrix "
    "multiplications required by neural networks, accelerating the current era of "
    "deep learning and large language models. "
)


def _build_haystack(target_tokens: int, needle_position: float) -> str:
    """
    Build a haystack of approximately target_tokens tokens with the needle
    inserted at needle_position (0.0 = beginning, 1.0 = end).
    Approximation: 1 token ≈ 4 characters.
    """
    target_chars = max(target_tokens * 4, len(_FILLER) * 2)
    reps = (target_chars // len(_FILLER)) + 2
    full = _FILLER * reps

    insert_at = int(target_chars * needle_position)
    pre  = full[:insert_at]
    post = full[insert_at:target_chars]

    return f"{pre}\n\n{NEEDLE}\n\n{post}"


def _query_ollama(model_id: str, context: str, timeout_s: int = 180) -> tuple[str, float]:
    """Return (answer, ttft_s)."""
    payload = {
        "model": model_id,
        "messages": [
            {
                "role": "system",
                "content": (
                    "You are a precise retrieval assistant. "
                    "Answer using ONLY what appears in the DOCUMENT below. "
                    "If the answer is not present, respond 'NOT FOUND'."
                ),
            },
            {
                "role": "user",
                "content": f"DOCUMENT:\n{context}\n\nQUESTION: {NEEDLE_Q}",
            },
        ],
        "stream": False,
        # Generous cap: reasoning models (DeepSeek-R1, etc.) spend tokens on a
        # <think> block before the actual answer — Ollama reports that as a
        # separate `thinking` field, but if num_predict is exhausted first,
        # `message.content` comes back empty. 2048 leaves room for CoT + the
        # one-line answer this task expects.
        "options": {"temperature": 0, "num_predict": 2048},
    }
    t0 = time.perf_counter()
    try:
        with httpx.Client(timeout=timeout_s) as client:
            resp = client.post(f"{OLLAMA_BASE_URL}/api/chat", json=payload)
            resp.raise_for_status()
        ttft = time.perf_counter() - t0
        answer = resp.json().get("message", {}).get("content", "")
        return answer.strip(), ttft
    except Exception as exc:
        logger.error("NIAH Ollama error (%s): %s", model_id, exc)
        return "", time.perf_counter() - t0


@dataclass
class NIAHResult:
    model_id:   str
    context_k:  int
    position:   float
    found:      bool
    answer:     str
    ttft_s:     float


def run_niah(
    model_id: str,
    context_sizes: list[int] | None = None,
    positions: list[float] | None = None,
    max_context_k: int = 32,
) -> list[NIAHResult]:
    """
    Run the full NIAH matrix for one model.
    max_context_k caps the largest haystack to the model's context limit.
    """
    sizes = [k for k in (context_sizes or CONTEXT_SIZES_K) if k <= max_context_k]
    pos_list = positions or POSITIONS
    results: list[NIAHResult] = []

    for k in sizes:
        for pos in pos_list:
            logger.info("NIAH | %s | %dk tokens | pos=%.0f%%", model_id, k, pos * 100)
            haystack = _build_haystack(k * 1000, pos)
            answer, ttft = _query_ollama(model_id, haystack)
            found = NEEDLE_ANS.lower() in answer.lower()
            results.append(NIAHResult(
                model_id=model_id,
                context_k=k,
                position=pos,
                found=found,
                answer=answer[:300],
                ttft_s=round(ttft, 3),
            ))
            logger.info("  found=%s | ttft=%.2fs | answer=%.80s", found, ttft, answer)

    return results


def results_to_dicts(results: list[NIAHResult]) -> list[dict]:
    return [asdict(r) for r in results]
