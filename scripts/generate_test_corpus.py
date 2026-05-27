"""
Generate a synthetic QA test corpus for benchmarking.

Pulls chunks from LanceDB, generates questions + reference answers using the
production model, and writes JSONL to data/benchmarks/test_corpus.jsonl.

Usage:
    python scripts/generate_test_corpus.py --count 50
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from config.settings import BENCHMARKS_DIR, BENCHMARK_QUERY_COUNT
from src.generation.ollama_client import generate_sync

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
logger = logging.getLogger(__name__)

_QA_PROMPT = """\
Given the following document excerpt, generate ONE factual question and a precise answer.
The question should be answerable from the excerpt alone.

FORMAT (JSON only):
{{"question": "...", "answer": "...", "difficulty": "easy|medium|hard"}}

EXCERPT:
{excerpt}

JSON:"""


def generate_qa_pair(text: str) -> dict | None:
    prompt = _QA_PROMPT.format(excerpt=text[:1500])
    response = generate_sync(
        [{"role": "user", "content": prompt}],
        temperature=0.5,
        max_tokens=300,
    )
    try:
        start = response.find("{")
        end = response.rfind("}") + 1
        if start == -1 or end == 0:
            return None
        return json.loads(response[start:end])
    except Exception:
        return None


def main(count: int) -> None:
    from src.retrieval.dense import _get_table

    output_path = BENCHMARKS_DIR / "test_corpus.jsonl"
    BENCHMARKS_DIR.mkdir(parents=True, exist_ok=True)

    table = _get_table()
    try:
        df = table.to_pandas()
    except Exception as e:
        logger.error("Could not load chunks from LanceDB: %s", e)
        logger.error("Ingest some documents first.")
        sys.exit(1)

    if df.empty:
        logger.error("No chunks found. Ingest documents before generating test corpus.")
        sys.exit(1)

    # Sample diverse chunks
    sample = df.sample(min(count * 2, len(df)), random_state=42)

    generated = []
    for _, row in sample.iterrows():
        if len(generated) >= count:
            break
        text = row.get("text", "")
        if len(text.split()) < 50:
            continue
        qa = generate_qa_pair(text)
        if not qa or not qa.get("question"):
            continue
        item = {
            "query": qa["question"],
            "answer": qa["answer"],
            "difficulty": qa.get("difficulty", "medium"),
            "contexts": [text],
            "source": row.get("source", ""),
        }
        generated.append(item)
        logger.info("[%d/%d] %s", len(generated), count, qa["question"][:80])

    with output_path.open("w", encoding="utf-8") as f:
        for item in generated:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    logger.info("Wrote %d QA pairs to %s", len(generated), output_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--count", type=int, default=BENCHMARK_QUERY_COUNT)
    args = parser.parse_args()
    main(args.count)
