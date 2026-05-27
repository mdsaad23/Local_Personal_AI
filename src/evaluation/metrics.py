"""
Latency and hardware metrics collection.

Measurements:
  TTFT  — time to first token (seconds)
  TGS   — token generation speed (tokens/second)
  TTLC  — time to last completion (seconds)
  VRAM  — GPU memory used (MB) — via hipInfo.exe on Windows/Vulkan
"""
from __future__ import annotations

import logging
import subprocess
import time
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class RunMetrics:
    model: str
    query: str
    ttft_s: float | None = None
    tgs: float | None = None
    ttlc_s: float | None = None
    eval_count: int = 0
    prompt_eval_count: int = 0
    vram_used_mb: float | None = None
    error: str | None = None
    extra: dict = field(default_factory=dict)


def measure_vram_mb() -> float | None:
    """
    Read GPU VRAM usage on Windows via hipInfo.exe.
    Returns used VRAM in MB, or None if not available.
    """
    try:
        result = subprocess.run(
            ["hipInfo.exe"],
            capture_output=True, text=True, timeout=10,
        )
        for line in result.stdout.splitlines():
            if "Total Global Mem" in line or "memoryUsed" in line.lower():
                parts = line.split(":")
                if len(parts) >= 2:
                    val_str = parts[-1].strip().split()[0]
                    val = float(val_str)
                    # hipInfo reports in bytes — convert to MB
                    return val / (1024 * 1024) if val > 1_000_000 else val
    except FileNotFoundError:
        logger.debug("hipInfo.exe not found — VRAM measurement unavailable")
    except Exception:
        logger.debug("VRAM measurement failed", exc_info=True)
    return None


def collect_run_metrics(
    model: str,
    query: str,
    ollama_metrics: dict[str, Any],
) -> RunMetrics:
    """Build a RunMetrics object from Ollama streaming metrics."""
    return RunMetrics(
        model=model,
        query=query,
        ttft_s=ollama_metrics.get("ttft_s"),
        tgs=ollama_metrics.get("tgs"),
        ttlc_s=ollama_metrics.get("ttlc_s"),
        eval_count=ollama_metrics.get("eval_count", 0),
        prompt_eval_count=ollama_metrics.get("prompt_eval_count", 0),
        vram_used_mb=measure_vram_mb(),
    )
