"""
Full benchmark harness — 9 model variants across multiple eval suites.

Suites
------
  rag     — Hybrid RAG: TTFT, TGS, VRAM, faithfulness, answer relevancy,
             hallucination, contextual precision/recall (default)
  niah    — Needle in a Haystack: context recall at 1k/4k/8k/16k/32k tokens
  tools   — Tool/function calling: 5 standard tasks per model
  coding  — HumanEval+ pass@1 via EvalPlus (20 problems per model by default)
  all     — Run all suites in sequence

Usage
-----
  python src/evaluation/benchmark.py                        # RAG suite only
  python src/evaluation/benchmark.py --suite niah
  python src/evaluation/benchmark.py --suite tools
  python src/evaluation/benchmark.py --suite coding --coding-n 20
  python src/evaluation/benchmark.py --suite all --skip-ragas

Output: CSV + summary JSON per suite in data/benchmarks/
"""
from __future__ import annotations

import argparse
import csv
import json
import logging
import sys
import time
from datetime import datetime
from itertools import groupby
from pathlib import Path

import statistics
import yaml

from config.settings import (
    BENCHMARK_QUERY_COUNT,
    BENCHMARK_TIMEOUT_SEC,
    BENCHMARK_WARMUP_RUNS,
    BENCHMARKS_DIR,
    OLLAMA_BASE_URL,
    PRODUCTION_MODEL,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


# ── Shared helpers ──────────────────────────────────────────────────────────────

def _load_models() -> list[dict]:
    models_path = Path(__file__).parent.parent.parent / "config" / "models.yaml"
    with models_path.open(encoding="utf-8") as f:
        return yaml.safe_load(f).get("benchmark_models", [])


def _ts() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def _write_csv(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _mean(vals: list) -> float | None:
    clean = [v for v in vals if v is not None]
    return statistics.mean(clean) if clean else None


# ── RAG suite ──────────────────────────────────────────────────────────────────

def _load_queries(corpus_path: Path) -> list[dict]:
    if not corpus_path.exists():
        logger.error("Test corpus not found: %s", corpus_path)
        logger.error("Run: python scripts/generate_test_corpus.py")
        sys.exit(1)
    queries = []
    with corpus_path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                queries.append(json.loads(line))
    return queries[:BENCHMARK_QUERY_COUNT]


def _run_single_rag(model_id: str, query: str) -> dict:
    from src.generation.ollama_client import stream_response
    from src.generation.router import retrieve_for_query, build_prompt
    from src.memory.episodic import retrieve_relevant
    from src.evaluation.metrics import measure_vram_mb

    chunks, route = retrieve_for_query(query)
    memories = retrieve_relevant(query, limit=3)
    messages = build_prompt(query, chunks, memories)

    answer_parts: list[str] = []
    ollama_metrics: dict = {}
    t0 = time.perf_counter()

    try:
        gen = stream_response(messages, model=model_id)
        while True:
            try:
                token = next(gen)
                answer_parts.append(token)
                if time.perf_counter() - t0 > BENCHMARK_TIMEOUT_SEC:
                    logger.warning("Query timed out after %ds", BENCHMARK_TIMEOUT_SEC)
                    break
            except StopIteration as e:
                ollama_metrics = e.value or {}
                break
    except Exception as exc:
        return {"error": str(exc), "answer": ""}

    return {
        "answer": "".join(answer_parts),
        "contexts": [c["text"] for c in chunks],
        "route": str(route),
        "ttft_s": ollama_metrics.get("ttft_s"),
        "tgs": ollama_metrics.get("tgs"),
        "ttlc_s": ollama_metrics.get("ttlc_s"),
        "eval_count": ollama_metrics.get("eval_count", 0),
        "prompt_eval_count": ollama_metrics.get("prompt_eval_count", 0),
        "vram_used_mb": measure_vram_mb(),
    }


def run_rag_suite(
    output_dir: Path,
    models: list[dict],
    skip_ragas: bool = False,
    skip_deepeval: bool = False,
) -> None:
    corpus_path = BENCHMARKS_DIR / "test_corpus.jsonl"
    queries = _load_queries(corpus_path)
    ts = _ts()
    csv_path = output_dir / f"rag_{ts}.csv"
    summary_path = output_dir / f"rag_{ts}_summary.json"

    fieldnames = [
        "model_id", "model_name", "query_index", "query", "route",
        "ttft_s", "tgs", "ttlc_s", "eval_count", "prompt_eval_count",
        "vram_used_mb",
        "faithfulness", "answer_relevancy",   # RAGAS
        "hallucination",                       # DeepEval
        "contextual_precision", "contextual_recall",  # DeepEval
        "error",
    ]

    all_rows: list[dict] = []

    with csv_path.open("w", newline="", encoding="utf-8") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()

        for model in models:
            model_id   = model["id"]
            model_name = model.get("name", model_id)
            logger.info("=== RAG suite: %s ===", model_name)

            for i in range(BENCHMARK_WARMUP_RUNS):
                logger.info("Warm-up %d/%d …", i + 1, BENCHMARK_WARMUP_RUNS)
                _run_single_rag(model_id, queries[0]["query"])

            for qi, item in enumerate(queries):
                logger.info("Query %d/%d: %.60s…", qi + 1, len(queries), item["query"])
                result = _run_single_rag(model_id, item["query"])

                faith = rel = 0.0
                hallucination = ctx_prec = ctx_rec = None

                if not result.get("error"):
                    if not skip_ragas:
                        from src.evaluation.ragas_eval import score_single
                        ragas = score_single(
                            item["query"], result["answer"], result.get("contexts", [])
                        )
                        faith = ragas["faithfulness"]
                        rel   = ragas["answer_relevancy"]

                    if not skip_deepeval:
                        from src.evaluation.deepeval_eval import score_all
                        de = score_all(
                            item["query"], result["answer"],
                            result.get("contexts", []),
                            item.get("answer", ""),
                        )
                        hallucination = de.get("hallucination")
                        ctx_prec      = de.get("contextual_precision")
                        ctx_rec       = de.get("contextual_recall")

                row = {
                    "model_id":            model_id,
                    "model_name":          model_name,
                    "query_index":         qi,
                    "query":               item["query"],
                    "route":               result.get("route", ""),
                    "ttft_s":              result.get("ttft_s"),
                    "tgs":                 result.get("tgs"),
                    "ttlc_s":              result.get("ttlc_s"),
                    "eval_count":          result.get("eval_count", 0),
                    "prompt_eval_count":   result.get("prompt_eval_count", 0),
                    "vram_used_mb":        result.get("vram_used_mb"),
                    "faithfulness":        faith,
                    "answer_relevancy":    rel,
                    "hallucination":       hallucination,
                    "contextual_precision": ctx_prec,
                    "contextual_recall":   ctx_rec,
                    "error":               result.get("error", ""),
                }
                writer.writerow(row)
                csvfile.flush()
                all_rows.append(row)
                logger.info(
                    "  TTFT=%.2fs TGS=%.1f tok/s Faith=%.2f Rel=%.2f Hall=%s",
                    result.get("ttft_s") or 0,
                    result.get("tgs") or 0,
                    faith, rel,
                    f"{hallucination:.2f}" if hallucination is not None else "n/a",
                )

    _write_rag_summary(all_rows, summary_path)
    logger.info("RAG suite done → %s", csv_path)


def _write_rag_summary(rows: list[dict], path: Path) -> None:
    summary: dict = {}
    sorted_rows = sorted(rows, key=lambda r: r["model_id"])
    for model_id, group in groupby(sorted_rows, key=lambda r: r["model_id"]):
        g = [r for r in group if not r.get("error")]
        if not g:
            continue
        summary[model_id] = {
            "model_name":            g[0]["model_name"],
            "query_count":           len(g),
            "avg_tgs":               _mean([r["tgs"] for r in g]),
            "avg_ttft_s":            _mean([r["ttft_s"] for r in g]),
            "avg_faithfulness":      _mean([r["faithfulness"] for r in g]),
            "avg_answer_relevancy":  _mean([r["answer_relevancy"] for r in g]),
            "avg_hallucination":     _mean([r["hallucination"] for r in g]),
            "avg_contextual_precision": _mean([r["contextual_precision"] for r in g]),
            "avg_contextual_recall": _mean([r["contextual_recall"] for r in g]),
            "avg_vram_mb":           _mean([r["vram_used_mb"] for r in g]),
        }
    path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")


# ── NIAH suite ─────────────────────────────────────────────────────────────────

def run_niah_suite(output_dir: Path, models: list[dict]) -> None:
    from src.evaluation.niah_eval import run_niah, results_to_dicts

    ts = _ts()
    all_rows: list[dict] = []

    for model in models:
        model_id = model["id"]
        # Respect the model's expected context window
        context_k = model.get("context_k", 32)
        logger.info("=== NIAH suite: %s (max %dk) ===", model.get("name", model_id), context_k)
        results = run_niah(model_id, max_context_k=context_k)
        for r in results_to_dicts(results):
            r["model_name"] = model.get("name", model_id)
            all_rows.append(r)

    fieldnames = ["model_id", "model_name", "context_k", "position", "found", "ttft_s", "answer"]
    csv_path = output_dir / f"niah_{ts}.csv"
    _write_csv(csv_path, all_rows, fieldnames)

    # Summary: recall matrix per model
    summary: dict = {}
    for model_id, group in groupby(
        sorted(all_rows, key=lambda r: r["model_id"]), key=lambda r: r["model_id"]
    ):
        g = list(group)
        summary[model_id] = {
            "model_name":    g[0]["model_name"],
            "total_cells":   len(g),
            "recall":        round(sum(r["found"] for r in g) / len(g), 3),
            "recall_by_k":   {
                str(k): round(sum(r["found"] for r in g if r["context_k"] == k)
                              / max(1, sum(1 for r in g if r["context_k"] == k)), 3)
                for k in sorted({r["context_k"] for r in g})
            },
        }

    (output_dir / f"niah_{ts}_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    logger.info("NIAH suite done → %s", csv_path)


# ── Tool-calling suite ─────────────────────────────────────────────────────────

def run_tools_suite(output_dir: Path, models: list[dict]) -> None:
    from src.evaluation.tool_eval import run_tool_eval, results_to_dicts

    ts = _ts()
    all_rows: list[dict] = []

    for model in models:
        model_id = model["id"]
        logger.info("=== Tool suite: %s ===", model.get("name", model_id))
        results = run_tool_eval(model_id)
        for r in results_to_dicts(results):
            r["model_name"] = model.get("name", model_id)
            all_rows.append(r)

    fieldnames = [
        "model_id", "model_name", "task_id",
        "tool_supported", "correct_fn", "correct_args", "latency_s", "raw_call",
    ]
    csv_path = output_dir / f"tools_{ts}.csv"
    _write_csv(csv_path, all_rows, fieldnames)

    summary: dict = {}
    for model_id, group in groupby(
        sorted(all_rows, key=lambda r: r["model_id"]), key=lambda r: r["model_id"]
    ):
        g = list(group)
        n = len(g)
        summary[model_id] = {
            "model_name":     g[0]["model_name"],
            "tool_support":   f"{sum(r['tool_supported'] for r in g)}/{n}",
            "fn_accuracy":    f"{sum(r['correct_fn'] for r in g)}/{n}",
            "arg_accuracy":   f"{sum(r['correct_args'] for r in g)}/{n}",
            "supports_tools": any(r["tool_supported"] for r in g),
        }

    (output_dir / f"tools_{ts}_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    logger.info("Tool suite done → %s", csv_path)


# ── Coding suite ───────────────────────────────────────────────────────────────

def run_coding_suite(
    output_dir: Path,
    models: list[dict],
    n_problems: int = 20,
    dataset: str = "humaneval",
) -> None:
    from src.evaluation.coding_eval import run_coding_eval, results_to_dicts

    ts = _ts()
    all_rows: list[dict] = []

    for model in models:
        model_id = model["id"]
        logger.info("=== Coding suite: %s (%d problems) ===", model.get("name", model_id), n_problems)
        result = run_coding_eval(model_id, dataset=dataset, n_problems=n_problems)
        row = results_to_dicts([result])[0]
        row["model_name"] = model.get("name", model_id)
        all_rows.append(row)

    fieldnames = ["model_id", "model_name", "dataset", "n_problems", "pass_at_1",
                  "solved", "duration_s", "error"]
    csv_path = output_dir / f"coding_{ts}.csv"
    _write_csv(csv_path, all_rows, fieldnames)

    summary = {r["model_id"]: {
        "model_name": r["model_name"],
        "pass_at_1":  r["pass_at_1"],
        "solved":     f"{r['solved']}/{r['n_problems']}",
        "duration_s": r["duration_s"],
    } for r in all_rows}

    (output_dir / f"coding_{ts}_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    logger.info("Coding suite done → %s", csv_path)


# ── Main ───────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Run benchmark matrix")
    parser.add_argument("--output",     type=Path, default=BENCHMARKS_DIR)
    parser.add_argument(
        "--suite",
        choices=["rag", "niah", "tools", "coding", "all"],
        default="rag",
        help="Which benchmark suite(s) to run",
    )
    parser.add_argument("--skip-ragas",    action="store_true", help="Skip RAGAS scoring")
    parser.add_argument("--skip-deepeval", action="store_true", help="Skip DeepEval metrics")
    parser.add_argument("--coding-n",      type=int, default=20,
                        help="Number of HumanEval+ problems per model (default: 20)")
    parser.add_argument("--coding-dataset", default="humaneval",
                        choices=["humaneval", "mbpp"])
    parser.add_argument("--models",        nargs="*",
                        help="Restrict to specific model IDs (default: all from models.yaml)")
    args = parser.parse_args()

    output_dir = args.output
    output_dir.mkdir(parents=True, exist_ok=True)

    all_models = _load_models()
    if args.models:
        models = [m for m in all_models if m["id"] in args.models]
        logger.info("Filtering to %d models: %s", len(models), args.models)
    else:
        models = all_models

    suites = ["rag", "niah", "tools", "coding"] if args.suite == "all" else [args.suite]

    for suite in suites:
        logger.info("━━━ Starting suite: %s ━━━", suite.upper())
        if suite == "rag":
            run_rag_suite(
                output_dir, models,
                skip_ragas=args.skip_ragas,
                skip_deepeval=args.skip_deepeval,
            )
        elif suite == "niah":
            run_niah_suite(output_dir, models)
        elif suite == "tools":
            run_tools_suite(output_dir, models)
        elif suite == "coding":
            run_coding_suite(
                output_dir, models,
                n_problems=args.coding_n,
                dataset=args.coding_dataset,
            )

    logger.info("All suites complete. Results in: %s", output_dir)


if __name__ == "__main__":
    main()
