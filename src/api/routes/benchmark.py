"""
Benchmark runner API.

POST /api/benchmark/start    — launch background run for selected suites + models
POST /api/benchmark/stop     — request cancellation
GET  /api/benchmark/status   — current state (running, progress, etc.)
GET  /api/benchmark/stream   — SSE log stream
GET  /api/benchmark/models   — selectable models from config/models.yaml
GET  /api/benchmark/summary  — structured per-suite result tables for the UI
GET  /api/benchmark/results  — raw result JSON files (legacy)

Suites
------
  tools  — tool/function-calling tasks per model
  niah   — needle-in-a-haystack long-context recall
  coding — HumanEval+ pass@1
  needle — positional-recall (vendored codeneedle) per corpus
"""
from __future__ import annotations

import json
import logging
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml
from fastapi import APIRouter
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from config.settings import BENCHMARKS_DIR

router = APIRouter(tags=["benchmark"])
logger = logging.getLogger(__name__)

_MODELS_YAML = Path(__file__).parent.parent.parent.parent / "config" / "models.yaml"


# ── Suite metadata ───────────────────────────────────────────────────────────────
# Each suite exposes a label and the columns the UI should render. Rows are
# accumulated into BenchmarkState.results[suite] as models complete.

SUITE_META: dict[str, dict[str, Any]] = {
    "tools": {
        "label": "Tool Calling",
        "columns": [
            {"key": "model_name",   "label": "Model"},
            {"key": "tool_support", "label": "Tool support"},
            {"key": "fn_accuracy",  "label": "Fn accuracy"},
            {"key": "arg_accuracy", "label": "Arg accuracy"},
        ],
    },
    "niah": {
        "label": "Long Context (NIAH)",
        "columns": [
            {"key": "model_name", "label": "Model"},
            {"key": "recall",     "label": "Recall"},
            {"key": "found",      "label": "Found"},
        ],
    },
    "coding": {
        "label": "Coding (HumanEval+)",
        "columns": [
            {"key": "model_name", "label": "Model"},
            {"key": "pass_at_1",  "label": "pass@1"},
            {"key": "solved",     "label": "Solved"},
        ],
    },
    "needle": {
        "label": "Positional Recall (codeneedle)",
        "columns": [
            {"key": "model_name", "label": "Model"},
            {"key": "corpus",     "label": "Corpus"},
            {"key": "pass_rate",  "label": "Pass rate"},
            {"key": "avg_recall", "label": "Avg line recall"},
            {"key": "passed",     "label": "Functions passed"},
            {"key": "note",       "label": "Notes"},
        ],
    },
}


# ── Shared state ───────────────────────────────────────────────────────────────

@dataclass
class BenchmarkState:
    running:        bool       = False
    stop_requested: bool       = False
    run_id:         str        = ""
    suites:         list[str]  = field(default_factory=list)
    models:         list[str]  = field(default_factory=list)
    current_suite:  str        = ""
    current_model:  str        = ""
    completed_models: int      = 0
    total_models:   int        = 0
    # Per-model sub-progress (e.g. needle query 3/11). total=0 means indeterminate.
    current_step:   int        = 0
    current_total:  int        = 0
    current_label:  str        = ""
    completed:      bool       = False
    cancelled:      bool       = False
    error:          str        = ""
    started_at:     float      = 0.0
    finished_at:    float      = 0.0
    results_dir:    str        = ""
    logs:           list[str]  = field(default_factory=list)
    # suite -> list of row dicts (one row per model, or per model×corpus for needle)
    results:        dict[str, list[dict]] = field(default_factory=dict)

_state = BenchmarkState()
_lock  = threading.Lock()


def _log(msg: str) -> None:
    with _lock:
        _state.logs.append(msg)
    logger.info("[benchmark] %s", msg)


def _add_row(suite: str, row: dict) -> None:
    with _lock:
        _state.results.setdefault(suite, []).append(row)


def _set_progress(step: int, total: int, label: str) -> None:
    with _lock:
        _state.current_step  = step
        _state.current_total = total
        _state.current_label = label


def _should_stop() -> bool:
    return _state.stop_requested


def _load_all_models() -> list[dict]:
    with _MODELS_YAML.open(encoding="utf-8") as f:
        return yaml.safe_load(f).get("benchmark_models", [])


def _pct(x: float | None) -> str:
    return f"{x:.0%}" if x is not None else "n/a"


# ── Background runner ──────────────────────────────────────────────────────────

def _run_benchmark(suites: list[str], model_ids: list[str], output_dir: Path) -> None:
    all_models = _load_all_models()
    models = [m for m in all_models if m["id"] in model_ids] if model_ids else all_models

    with _lock:
        _state.total_models = len(models) * len(suites)
        _state.completed_models = 0
        _state.results_dir = str(output_dir)

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

                model_id   = model["id"]
                model_name = model.get("name", model_id)
                with _lock:
                    _state.current_model = model_id
                    _state.current_step = 0
                    _state.current_total = 0
                    _state.current_label = f"{suite}: {model_name}"
                _log(f"Model: {model_name}")

                try:
                    _run_one(suite, model, output_dir)
                except Exception as exc:
                    _log(f"  ERROR: {exc}")
                    logger.exception("Suite %s failed for %s", suite, model_id)

                with _lock:
                    _state.completed_models += 1
                    _state.current_step = 0
                    _state.current_total = 0

        _write_summary(output_dir)
        with _lock:
            _state.completed   = True
            _state.running     = False
            _state.cancelled   = _state.stop_requested
            _state.current_label = ""
            _state.finished_at = time.time()
        _log("Benchmark cancelled." if _state.cancelled
             else f"Benchmark complete. Results in: {output_dir}")

    except Exception as exc:
        with _lock:
            _state.error     = str(exc)
            _state.running   = False
            _state.completed = True
        _log(f"FATAL: {exc}")
        logger.exception("Benchmark runner crashed")


def _run_one(suite: str, model: dict, output_dir: Path) -> None:
    """Run a single suite for a single model, appending a normalized result row."""
    model_id   = model["id"]
    model_name = model.get("name", model_id)
    slug       = model_id.replace(":", "_")

    if suite == "tools":
        from src.evaluation.tool_eval import run_tool_eval, results_to_dicts
        results = run_tool_eval(model_id)
        _save_json(output_dir / f"tools_{slug}.json", results_to_dicts(results))
        n = len(results)
        supported = sum(r.tool_supported for r in results)
        fn  = sum(r.correct_fn for r in results)
        arg = sum(r.correct_args for r in results)
        _add_row("tools", {
            "model_id": model_id, "model_name": model_name,
            "tool_support": f"{supported}/{n}",
            "fn_accuracy":  f"{fn}/{n}",
            "arg_accuracy": f"{arg}/{n}",
        })
        _log(f"  tools: {supported}/{n} supported")

    elif suite == "niah":
        from src.evaluation.niah_eval import run_niah, results_to_dicts
        ctx_k = model.get("context_k", 32)
        results = run_niah(model_id, max_context_k=ctx_k)
        _save_json(output_dir / f"niah_{slug}.json", results_to_dicts(results))
        found = sum(r.found for r in results)
        n = len(results)
        recall = found / max(n, 1)
        _add_row("niah", {
            "model_id": model_id, "model_name": model_name,
            "recall": _pct(recall), "found": f"{found}/{n}",
        })
        _log(f"  niah: recall={recall:.1%} ({found}/{n})")

    elif suite == "coding":
        from src.evaluation.coding_eval import run_coding_eval, results_to_dicts
        result = run_coding_eval(model_id, n_problems=20)
        _save_json(output_dir / f"coding_{slug}.json", results_to_dicts([result]))
        _add_row("coding", {
            "model_id": model_id, "model_name": model_name,
            "pass_at_1": _pct(result.pass_at_1),
            "solved": f"{result.solved}/{result.n_problems}",
        })
        _log(f"  coding: pass@1={result.pass_at_1:.1%} ({result.solved}/{result.n_problems})")

    elif suite == "needle":
        from src.evaluation.needle_eval import run_needle
        summaries = run_needle(
            model_id, output_dir, log=_log,
            progress=_set_progress, should_stop=_should_stop,
        )
        _save_json(output_dir / f"needle_summary_{slug}.json", summaries)
        for s in summaries:
            if s["error"]:
                note = s["error"][:60]
            else:
                note = f"{s['errored']} errored" if s["errored"] else "ok"
            _add_row("needle", {
                "model_id": model_id, "model_name": model_name,
                "corpus": s["corpus"],
                "pass_rate": _pct(s["pass_rate"]) if not s["error"] else "—",
                "avg_recall": _pct(s["avg_recall"]) if not s["error"] else "—",
                "passed": f"{s['passed']}/{s['n_functions']}" if s["n_functions"] else "—",
                "note": note,
            })
            _log(f"  needle[{s['corpus']}]: pass={s['passed']}/{s['n_functions']}"
                 + (f" ({s['error'][:40]})" if s["error"] else ""))

    elif suite == "rag":
        _log("  rag: skipping inline — run benchmark.py --suite rag separately")


def _save_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, default=str), encoding="utf-8")


def _build_summary() -> dict:
    """Shape accumulated rows into per-suite tables for the UI. If empty, load latest."""
    with _lock:
        results = {s: list(rows) for s, rows in _state.results.items()}
        run_id = _state.run_id
        
    # If no run in memory, try to load the most recent summary.json from disk
    if not results and BENCHMARKS_DIR.exists():
        files = sorted(BENCHMARKS_DIR.rglob("summary.json"), key=lambda p: p.stat().st_mtime, reverse=True)
        if files:
            try:
                return json.loads(files[0].read_text(encoding="utf-8"))
            except Exception as exc:
                logger.error(f"Failed to load latest summary: {exc}")

    suites = {}
    for suite, rows in results.items():
        meta = SUITE_META.get(suite, {"label": suite, "columns": []})
        suites[suite] = {
            "label":   meta["label"],
            "columns": meta["columns"],
            "rows":    rows,
        }
    return {"run_id": run_id, "suites": suites}


def _write_summary(output_dir: Path) -> None:
    _save_json(output_dir / "summary.json", _build_summary())


# ── Request models ──────────────────────────────────────────────────────────────

class StartRequest(BaseModel):
    suites: list[str] = ["tools", "niah", "needle"]
    models: list[str] = []


# ── Routes ────────────────────────────────────────────────────────────────────

@router.get("/benchmark/models")
async def list_models():
    """Selectable models for the benchmark UI."""
    return [
        {
            "id":       m["id"],
            "name":     m.get("name", m["id"]),
            "family":   m.get("family", ""),
            "params_b": m.get("params_b"),
            "quant":    m.get("quant", ""),
        }
        for m in _load_all_models()
    ]


@router.post("/benchmark/start")
async def start_benchmark(req: StartRequest):
    with _lock:
        if _state.running:
            return {"status": "already_running", "message": "Benchmark is already running"}
        _state.__init__()  # reset
        _state.running    = True
        _state.started_at = time.time()
        _state.run_id     = f"run_{int(_state.started_at)}"
        _state.suites     = req.suites or ["tools", "niah", "needle"]
        _state.models     = req.models or []
        run_id = _state.run_id
        suites = _state.suites
        model_ids = _state.models

    output_dir = BENCHMARKS_DIR / run_id
    thread = threading.Thread(
        target=_run_benchmark,
        args=(suites, model_ids, output_dir),
        daemon=True,
    )
    thread.start()
    return {"status": "started", "run_id": run_id, "suites": suites, "models": model_ids}


@router.post("/benchmark/stop")
async def stop_benchmark():
    with _lock:
        if not _state.running:
            return {"status": "not_running"}
        _state.stop_requested = True
    # Kill any in-flight needle subprocess immediately so cancel is responsive;
    # already-collected results are preserved and the run winds down cleanly.
    from src.evaluation.needle_eval import request_cancel
    request_cancel()
    return {"status": "stop_requested"}


@router.get("/benchmark/status")
async def get_status():
    with _lock:
        return {
            "running":          _state.running,
            "completed":        _state.completed,
            "cancelled":        _state.cancelled,
            "current_suite":    _state.current_suite,
            "current_model":    _state.current_model,
            "current_step":     _state.current_step,
            "current_total":    _state.current_total,
            "current_label":    _state.current_label,
            "completed_models": _state.completed_models,
            "total_models":     _state.total_models,
            "error":            _state.error,
            "results_dir":      _state.results_dir,
            "elapsed_s":        round(time.time() - _state.started_at, 1) if _state.started_at else 0,
        }


@router.get("/benchmark/summary")
async def get_summary():
    """Structured per-suite result tables for the current/last run."""
    return _build_summary()


@router.get("/benchmark/stream")
async def stream_logs():
    """SSE stream — sends log lines as they appear."""
    def generate():
        sent = 0
        while True:
            with _lock:
                logs     = list(_state.logs)
                complete = _state.completed

            while sent < len(logs):
                yield f"data: {json.dumps({'type': 'log', 'text': logs[sent]})}\n\n"
                sent += 1

            if complete and sent >= len(logs):
                with _lock:
                    status = {
                        "type":             "done",
                        "error":            _state.error,
                        "cancelled":        _state.cancelled,
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


@router.get("/benchmark/history")
async def get_history():
    """Return a list of previous benchmark runs that have a summary."""
    if not BENCHMARKS_DIR.exists():
        return []
    history = []
    # Find all summary.json files
    files = sorted(BENCHMARKS_DIR.rglob("summary.json"), key=lambda p: p.stat().st_mtime, reverse=True)
    for f in files:
        run_id = f.parent.name
        # Simple extraction of timestamp from "run_1781363702"
        timestamp = 0
        try:
            if run_id.startswith("run_"):
                timestamp = int(run_id.split("_")[1])
        except ValueError:
            timestamp = int(f.stat().st_mtime)
        history.append({"run_id": run_id, "timestamp": timestamp})
    return history

@router.get("/benchmark/summary/{run_id}")
async def get_summary_by_id(run_id: str):
    """Return the summary.json for a specific run ID."""
    summary_path = BENCHMARKS_DIR / run_id / "summary.json"
    if summary_path.exists():
        try:
            return json.loads(summary_path.read_text(encoding="utf-8"))
        except Exception as exc:
            logger.error(f"Failed to load summary {run_id}: {exc}")
    return {}
