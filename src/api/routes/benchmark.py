"""
Benchmark runner API.

POST /api/benchmark/start   — launch background benchmark run
GET  /api/benchmark/status  — current state (running, progress, etc.)
GET  /api/benchmark/stream  — SSE log stream
POST /api/benchmark/stop    — request cancellation
"""
from __future__ import annotations

import json
import logging
import threading
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any

from fastapi import APIRouter
from fastapi.responses import StreamingResponse

from config.settings import BENCHMARKS_DIR

router = APIRouter(tags=["benchmark"])
logger = logging.getLogger(__name__)


# ── Shared state ───────────────────────────────────────────────────────────────

@dataclass
class BenchmarkState:
    running:       bool       = False
    stop_requested: bool      = False
    suites:        list[str]  = field(default_factory=list)
    models:        list[str]  = field(default_factory=list)
    current_suite: str        = ""
    current_model: str        = ""
    completed_models: int     = 0
    total_models:  int        = 0
    completed:     bool       = False
    error:         str        = ""
    started_at:    float      = 0.0
    finished_at:   float      = 0.0
    results_dir:   str        = ""
    logs:          list[str]  = field(default_factory=list)

_state = BenchmarkState()
_lock  = threading.Lock()


def _log(msg: str) -> None:
    with _lock:
        _state.logs.append(msg)
    logger.info("[benchmark] %s", msg)


# ── Background runner ──────────────────────────────────────────────────────────

def _run_benchmark(suites: list[str], model_ids: list[str], output_dir: Path) -> None:
    import yaml
    from config.settings import BENCHMARKS_DIR

    models_yaml = Path(__file__).parent.parent.parent.parent / "config" / "models.yaml"
    with models_yaml.open(encoding="utf-8") as f:
        all_models = yaml.safe_load(f).get("benchmark_models", [])

    if model_ids:
        models = [m for m in all_models if m["id"] in model_ids]
    else:
        models = all_models

    with _lock:
        _state.total_models  = len(models) * len(suites)
        _state.completed_models = 0
        _state.results_dir   = str(output_dir)

    _log(f"Starting benchmark: suites={suites} models={len(models)}")

    try:
        for suite in suites:
            if _state.stop_requested:
                _log("Stop requested — halting.")
                break

            with _lock:
                _state.current_suite = suite
            _log(f"━━━ Suite: {suite.upper()} ━━━")

            for model in models:
                if _state.stop_requested:
                    break

                model_id = model["id"]
                with _lock:
                    _state.current_model = model_id
                _log(f"Model: {model.get('name', model_id)}")

                try:
                    if suite == "tools":
                        from src.evaluation.tool_eval import run_tool_eval, results_to_dicts
                        results = run_tool_eval(model_id)
                        _save_json(output_dir / f"tools_{model_id.replace(':', '_')}.json",
                                   results_to_dicts(results))
                        supported = sum(r.tool_supported for r in results)
                        _log(f"  tools: {supported}/5 supported")

                    elif suite == "niah":
                        from src.evaluation.niah_eval import run_niah, results_to_dicts
                        ctx_k = model.get("context_k", 32)
                        results = run_niah(model_id, max_context_k=ctx_k)
                        _save_json(output_dir / f"niah_{model_id.replace(':', '_')}.json",
                                   results_to_dicts(results))
                        recall = sum(r.found for r in results) / max(len(results), 1)
                        _log(f"  niah: recall={recall:.1%} ({sum(r.found for r in results)}/{len(results)})")

                    elif suite == "coding":
                        from src.evaluation.coding_eval import run_coding_eval, results_to_dicts
                        result = run_coding_eval(model_id, n_problems=20)
                        _save_json(output_dir / f"coding_{model_id.replace(':', '_')}.json",
                                   results_to_dicts([result]))
                        _log(f"  coding: pass@1={result.pass_at_1:.1%} ({result.solved}/{result.n_problems})")

                    elif suite == "rag":
                        _log(f"  rag: skipping inline — run benchmark.py --suite rag separately")

                except Exception as exc:
                    _log(f"  ERROR: {exc}")
                    logger.exception("Suite %s failed for %s", suite, model_id)

                with _lock:
                    _state.completed_models += 1

        with _lock:
            _state.completed    = True
            _state.running      = False
            _state.finished_at  = time.time()
        _log(f"Benchmark complete. Results in: {output_dir}")

    except Exception as exc:
        with _lock:
            _state.error     = str(exc)
            _state.running   = False
            _state.completed = True
        _log(f"FATAL: {exc}")
        logger.exception("Benchmark runner crashed")


def _save_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, default=str), encoding="utf-8")


# ── Routes ────────────────────────────────────────────────────────────────────

@router.post("/benchmark/start")
async def start_benchmark(
    suites: list[str] | None = None,
    models: list[str] | None = None,
):
    with _lock:
        if _state.running:
            return {"status": "already_running", "message": "Benchmark is already running"}
        _state.__init__()   # reset
        _state.running      = True
        _state.started_at   = time.time()
        _state.suites       = suites or ["tools", "niah", "coding"]
        _state.models       = models or []

    output_dir = BENCHMARKS_DIR / f"run_{int(time.time())}"
    thread = threading.Thread(
        target=_run_benchmark,
        args=(_state.suites, _state.models, output_dir),
        daemon=True,
    )
    thread.start()
    return {"status": "started", "suites": _state.suites}


@router.post("/benchmark/stop")
async def stop_benchmark():
    with _lock:
        if not _state.running:
            return {"status": "not_running"}
        _state.stop_requested = True
    return {"status": "stop_requested"}


@router.get("/benchmark/status")
async def get_status():
    with _lock:
        return {
            "running":          _state.running,
            "completed":        _state.completed,
            "current_suite":    _state.current_suite,
            "current_model":    _state.current_model,
            "completed_models": _state.completed_models,
            "total_models":     _state.total_models,
            "error":            _state.error,
            "results_dir":      _state.results_dir,
            "elapsed_s":        round(time.time() - _state.started_at, 1) if _state.started_at else 0,
        }


@router.get("/benchmark/stream")
async def stream_logs():
    """SSE stream — sends log lines as they appear."""
    def generate():
        sent = 0
        while True:
            with _lock:
                logs     = _state.logs
                running  = _state.running
                complete = _state.completed

            while sent < len(logs):
                line = logs[sent]
                yield f"data: {json.dumps({'type': 'log', 'text': line})}\n\n"
                sent += 1

            if complete and sent >= len(logs):
                with _lock:
                    status = {
                        "type":             "done",
                        "error":            _state.error,
                        "results_dir":      _state.results_dir,
                        "completed_models": _state.completed_models,
                        "total_models":     _state.total_models,
                    }
                yield f"data: {json.dumps(status)}\n\n"
                break

            time.sleep(0.5)

    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@router.get("/benchmark/results")
async def list_results():
    """Return list of result JSON files from the most recent run."""
    if not BENCHMARKS_DIR.exists():
        return []
    files = sorted(BENCHMARKS_DIR.rglob("*.json"), key=lambda p: p.stat().st_mtime, reverse=True)
    results = []
    for f in files[:50]:
        try:
            data = json.loads(f.read_text(encoding="utf-8"))
            results.append({"file": f.name, "path": str(f), "data": data})
        except Exception:
            pass
    return results
