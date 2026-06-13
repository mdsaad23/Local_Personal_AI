"""
Positional-recall (codeneedle) suite — adapter around the vendored
`local_llm_benchmarking` CLI.

For each model we shell out to `vendor/local_llm_benchmarking/bench.py run`
once per corpus, pointing it at Ollama's OpenAI-compatible endpoint
(OLLAMA_BASE_URL + /v1, which the vendored client appends itself). The CLI
writes a JSON dump per run; we parse it into a compact per-corpus summary.

The vendored tool measures whether a model can reproduce the first N lines of a
named function verbatim when the whole source file is stuffed into its context
— positional recall under long context, not named-entity lookup.

This module is sync (query-pipeline side) and never imports the vendored
`bench` package directly: it runs as a subprocess so the vendored repo stays an
isolated, swappable dependency.
"""
from __future__ import annotations

import json
import os
import re
import statistics
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Callable

from loguru import logger

from config.settings import (
    NEEDLE_CORPORA,
    NEEDLE_DIR,
    NEEDLE_MAX_TOKENS,
    NEEDLE_TIMEOUT_SEC,
    OLLAMA_BASE_URL,
)

LogFn = Callable[[str], None]
ProgressFn = Callable[[int, int, str], None]   # (step, total, label)
StopFn = Callable[[], bool]

# `[3/11] `func` — prompt …` is printed at the start of each query.
_PROGRESS_RE = re.compile(r"^\[(\d+)/(\d+)\]\s*`?([^`\s]+)?")

# Reference to the in-flight subprocess so a cancel request can kill it
# immediately rather than waiting for the current model query to return.
_active_proc: subprocess.Popen | None = None
_active_lock = threading.Lock()


def request_cancel() -> None:
    """Kill the currently-running benchmark subprocess, if any. Safe to call
    when nothing is running. Partial results already written stay intact."""
    with _active_lock:
        proc = _active_proc
    if proc and proc.poll() is None:
        try:
            proc.kill()
        except OSError:
            pass


def available_corpora() -> list[str]:
    """Corpus config stems shipped with the vendored benchmark."""
    corpora_dir = NEEDLE_DIR / "configs" / "corpora"
    if not corpora_dir.is_dir():
        return []
    return sorted(p.stem for p in corpora_dir.glob("*.toml"))


def _safe_stem(model_id: str) -> str:
    return "".join(c if c.isalnum() or c in "-_." else "_" for c in model_id)


def run_needle_corpus(
    model_id: str,
    corpus: str,
    output_dir: Path,
    *,
    base_url: str = OLLAMA_BASE_URL,
    max_tokens: int = NEEDLE_MAX_TOKENS,
    timeout_s: float = NEEDLE_TIMEOUT_SEC,
    log: LogFn | None = None,
    progress: ProgressFn | None = None,
    should_stop: StopFn | None = None,
) -> dict:
    """Run one model against one corpus. Returns a compact summary dict.

    Never raises for a benchmark failure — a crash, timeout, context-too-small,
    or user cancellation is captured in the returned dict's ``error`` field so
    the caller can keep going (or wind down) through the remaining models.
    """
    emit = log or (lambda _m: None)
    report = progress or (lambda _s, _t, _l: None)
    stopped = should_stop or (lambda: False)
    if stopped():
        return _error_summary(model_id, corpus, "cancelled")
    # Must be absolute: the subprocess runs with cwd=NEEDLE_DIR, so a relative
    # --dump would be written under the vendor dir instead of output_dir.
    output_dir = output_dir.resolve()
    dump_path = output_dir / f"needle_{corpus}__{_safe_stem(model_id)}.json"
    dump_path.parent.mkdir(parents=True, exist_ok=True)

    cmd = [
        sys.executable, "bench.py", "run",
        "--corpus", corpus,
        "--model", model_id,
        "--base-url", base_url,
        "--max-tokens", str(max_tokens),
        "--timeout", str(timeout_s),
        "--dump", str(dump_path),
    ]

    proc_env = {**os.environ, "PYTHONIOENCODING": "utf-8", "PYTHONUNBUFFERED": "1"}

    t0 = time.perf_counter()
    tail: list[str] = []
    try:
        proc = subprocess.Popen(
            cmd,
            cwd=str(NEEDLE_DIR),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
            env=proc_env,
        )
    except Exception as exc:
        msg = f"failed to launch bench.py: {exc}"
        emit(f"    ERROR: {msg}")
        logger.exception("needle: could not launch subprocess")
        return _error_summary(model_id, corpus, msg)

    with _active_lock:
        global _active_proc
        _active_proc = proc

    cancelled = False
    try:
        assert proc.stdout is not None
        deadline = t0 + timeout_s * 40  # generous wall cap across all queries in a corpus
        for line in proc.stdout:
            line = line.rstrip("\n")
            if line:
                tail.append(line)
                if len(tail) > 40:
                    tail.pop(0)
                m = _PROGRESS_RE.match(line.strip())
                if m:
                    step, total = int(m.group(1)), int(m.group(2))
                    fn = m.group(3) or ""
                    report(step, total, f"{corpus}: {fn}".rstrip(": "))
                # Forward only the salient progress lines to the UI; the tool is chatty.
                if line.lstrip().startswith(("[", "PASS", "FAIL", "ERROR", "❌", "⚠", "Source:", "Selected", "Pre-flight")):
                    emit(f"    {line.strip()}")
            if stopped():
                cancelled = True
                proc.kill()
                emit(f"    Cancelled corpus '{corpus}' — stopping safely.")
                break
            if time.perf_counter() > deadline:
                proc.kill()
                emit(f"    ERROR: corpus '{corpus}' exceeded wall-clock cap — killed")
                return _error_summary(model_id, corpus, "timeout (wall-clock cap exceeded)")

        proc.wait()
    finally:
        with _active_lock:
            _active_proc = None

    duration = time.perf_counter() - t0

    if cancelled or stopped():
        return _error_summary(model_id, corpus, "cancelled")

    if not dump_path.is_file():
        reason = tail[-1] if tail else f"exit code {proc.returncode}, no result dump"
        emit(f"    ERROR: no results ({reason})")
        return _error_summary(model_id, corpus, reason)

    try:
        data = json.loads(dump_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        emit(f"    ERROR: could not read dump: {exc}")
        return _error_summary(model_id, corpus, f"unreadable dump: {exc}")

    return _summarize(model_id, corpus, data, duration)


def _summarize(model_id: str, corpus: str, data: dict, duration: float) -> dict:
    results = data.get("results", [])
    n = len(results)
    passed = sum(1 for r in results if r.get("passed"))
    errored = sum(1 for r in results if r.get("error"))
    ratios = [
        r["primary_matched"] / r["primary_total"]
        for r in results
        if r.get("primary_total")
    ]
    latencies = [r["latency_s"] for r in results if r.get("latency_s") is not None]
    return {
        "model_id":     model_id,
        "corpus":       corpus,
        "n_functions":  n,
        "passed":       passed,
        "errored":      errored,
        "pass_rate":    round(passed / n, 3) if n else 0.0,
        "avg_recall":   round(statistics.mean(ratios), 3) if ratios else 0.0,
        "avg_latency_s": round(statistics.mean(latencies), 1) if latencies else None,
        "duration_s":   round(duration, 1),
        "error":        "",
    }


def _error_summary(model_id: str, corpus: str, message: str) -> dict:
    return {
        "model_id":     model_id,
        "corpus":       corpus,
        "n_functions":  0,
        "passed":       0,
        "errored":      0,
        "pass_rate":    0.0,
        "avg_recall":   0.0,
        "avg_latency_s": None,
        "duration_s":   0.0,
        "error":        message,
    }


def run_needle(
    model_id: str,
    output_dir: Path,
    *,
    corpora: list[str] | None = None,
    log: LogFn | None = None,
    progress: ProgressFn | None = None,
    should_stop: StopFn | None = None,
) -> list[dict]:
    """Run a model across all configured corpora. Returns one summary per corpus."""
    corpora = corpora or NEEDLE_CORPORA
    stopped = should_stop or (lambda: False)
    summaries: list[dict] = []
    for corpus in corpora:
        if stopped():
            summaries.append(_error_summary(model_id, corpus, "cancelled"))
            continue
        summaries.append(run_needle_corpus(
            model_id, corpus, output_dir,
            log=log, progress=progress, should_stop=should_stop,
        ))
    return summaries
