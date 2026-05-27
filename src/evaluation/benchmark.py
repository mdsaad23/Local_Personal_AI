"""
Full benchmark harness — 9 model variants × 3 query types × TTFT/TGS/VRAM/RAGAS.

Usage:
    python src/evaluation/benchmark.py --output data/benchmarks/

Each model is loaded, warmed up (BENCHMARK_WARMUP_RUNS discarded), then
timed across BENCHMARK_QUERY_COUNT queries sampled from the test corpus.
Results saved as CSV + summary JSON.
"""
from __future__ import annotations

import argparse
import csv
import json
import logging
import sys
import time
from datetime import datetime
from pathlib import Path

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


def _load_models() -> list[dict]:
    models_path = Path(__file__).parent.parent.parent / "config" / "models.yaml"
    with models_path.open(encoding="utf-8") as f:
        return yaml.safe_load(f).get("benchmark_models", [])


def _load_queries(corpus_path: Path) -> list[dict]:
    """Load test queries from JSONL file. Each line: {query, answer, contexts}."""
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


def _run_single_query(model_id: str, query: str) -> dict:
    """Run one query against one model. Returns timing + answer metrics."""
    from src.generation.ollama_client import stream_response
    from src.generation.router import retrieve_for_query, build_prompt
    from src.memory.episodic import retrieve_relevant
    from src.evaluation.metrics import measure_vram_mb

    chunks, route = retrieve_for_query(query)
    memories = retrieve_relevant(query, limit=3)
    messages = build_prompt(query, chunks, memories)

    answer_parts = []
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
    except Exception as e:
        return {"error": str(e), "answer": ""}

    answer = "".join(answer_parts)
    contexts = [c["text"] for c in chunks]

    return {
        "answer": answer,
        "contexts": contexts,
        "route": str(route),
        "ttft_s": ollama_metrics.get("ttft_s"),
        "tgs": ollama_metrics.get("tgs"),
        "ttlc_s": ollama_metrics.get("ttlc_s"),
        "eval_count": ollama_metrics.get("eval_count", 0),
        "prompt_eval_count": ollama_metrics.get("prompt_eval_count", 0),
        "vram_used_mb": measure_vram_mb(),
    }


def run_benchmark(output_dir: Path, skip_ragas: bool = False) -> None:
    import httpx

    models = _load_models()
    corpus_path = BENCHMARKS_DIR / "test_corpus.jsonl"
    queries = _load_queries(corpus_path)

    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_path = output_dir / f"benchmark_{timestamp}.csv"
    summary_path = output_dir / f"benchmark_{timestamp}_summary.json"

    fieldnames = [
        "model_id", "model_name", "query_index", "query", "route",
        "ttft_s", "tgs", "ttlc_s", "eval_count", "prompt_eval_count",
        "vram_used_mb", "faithfulness", "answer_relevancy", "error",
    ]

    all_rows = []

    with csv_path.open("w", newline="", encoding="utf-8") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()

        for model in models:
            model_id = model["id"]
            model_name = model.get("name", model_id)
            logger.info("=== Benchmarking: %s ===", model_name)

            # Warm-up runs (discarded)
            for i in range(BENCHMARK_WARMUP_RUNS):
                logger.info("Warm-up %d/%d …", i + 1, BENCHMARK_WARMUP_RUNS)
                _run_single_query(model_id, queries[0]["query"])

            for qi, item in enumerate(queries):
                logger.info("Query %d/%d: %.60s …", qi + 1, len(queries), item["query"])
                result = _run_single_query(model_id, item["query"])

                # RAGAS scoring
                faith = 0.0
                relevancy = 0.0
                if not skip_ragas and not result.get("error"):
                    from src.evaluation.ragas_eval import score_single
                    scores = score_single(
                        item["query"], result["answer"], result.get("contexts", [])
                    )
                    faith = scores["faithfulness"]
                    relevancy = scores["answer_relevancy"]

                row = {
                    "model_id": model_id,
                    "model_name": model_name,
                    "query_index": qi,
                    "query": item["query"],
                    "route": result.get("route", ""),
                    "ttft_s": result.get("ttft_s"),
                    "tgs": result.get("tgs"),
                    "ttlc_s": result.get("ttlc_s"),
                    "eval_count": result.get("eval_count", 0),
                    "prompt_eval_count": result.get("prompt_eval_count", 0),
                    "vram_used_mb": result.get("vram_used_mb"),
                    "faithfulness": faith,
                    "answer_relevancy": relevancy,
                    "error": result.get("error", ""),
                }
                writer.writerow(row)
                csvfile.flush()
                all_rows.append(row)
                logger.info(
                    "  TTFT=%.2fs TGS=%.1f tok/s Faith=%.2f Rel=%.2f",
                    result.get("ttft_s") or 0,
                    result.get("tgs") or 0,
                    faith, relevancy,
                )

    # Write summary JSON
    _write_summary(all_rows, summary_path)
    logger.info("Benchmark complete. Results: %s", csv_path)
    logger.info("Summary: %s", summary_path)


def _write_summary(rows: list[dict], path: Path) -> None:
    import statistics
    from itertools import groupby

    summary = {}
    sorted_rows = sorted(rows, key=lambda r: r["model_id"])
    for model_id, group in groupby(sorted_rows, key=lambda r: r["model_id"]):
        g = [r for r in group if not r.get("error")]
        if not g:
            continue
        tgs_vals = [r["tgs"] for r in g if r.get("tgs")]
        ttft_vals = [r["ttft_s"] for r in g if r.get("ttft_s")]
        summary[model_id] = {
            "model_name": g[0]["model_name"],
            "query_count": len(g),
            "avg_tgs": statistics.mean(tgs_vals) if tgs_vals else None,
            "avg_ttft_s": statistics.mean(ttft_vals) if ttft_vals else None,
            "avg_faithfulness": statistics.mean(r["faithfulness"] for r in g),
            "avg_answer_relevancy": statistics.mean(r["answer_relevancy"] for r in g),
            "avg_vram_mb": statistics.mean(r["vram_used_mb"] for r in g if r.get("vram_used_mb")) or None,
        }

    path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run benchmark matrix")
    parser.add_argument("--output", type=Path, default=BENCHMARKS_DIR)
    parser.add_argument("--skip-ragas", action="store_true",
                        help="Skip RAGAS scoring (offline mode)")
    args = parser.parse_args()
    run_benchmark(args.output, skip_ragas=args.skip_ragas)
