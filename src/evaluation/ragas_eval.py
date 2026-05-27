"""
RAGAS evaluation via DeepSeek V3 as judge.

Scores each (query, answer, context_chunks) triple on:
  - faithfulness: answer is grounded in retrieved context
  - answer_relevancy: answer actually addresses the query

DeepSeek V3 is used as the independent judge — it is never one of the models
under evaluation, preventing judge bias.
"""
from __future__ import annotations

import logging
from typing import Any

from config.settings import DEEPSEEK_API_KEY, DEEPSEEK_BASE_URL, DEEPSEEK_MODEL

logger = logging.getLogger(__name__)


def _get_ragas_config():
    """Build RAGAS LLM/embedder config pointing at DeepSeek."""
    from ragas.llms import LangchainLLMWrapper
    from langchain_openai import ChatOpenAI

    if not DEEPSEEK_API_KEY:
        raise RuntimeError(
            "DEEPSEEK_API_KEY is not set. RAGAS evaluation requires DeepSeek V3 as judge. "
            "Set it in .env — see .env.example."
        )

    llm = ChatOpenAI(
        model=DEEPSEEK_MODEL,
        api_key=DEEPSEEK_API_KEY,
        base_url=DEEPSEEK_BASE_URL,
        temperature=0,
    )
    return LangchainLLMWrapper(llm)


def score_single(
    query: str,
    answer: str,
    contexts: list[str],
) -> dict[str, float]:
    """
    Score one (query, answer, contexts) triple.
    Returns {"faithfulness": 0-1, "answer_relevancy": 0-1}.
    """
    try:
        from ragas import evaluate
        from ragas.metrics import faithfulness, answer_relevancy
        from datasets import Dataset

        ragas_llm = _get_ragas_config()
        faithfulness.llm = ragas_llm
        answer_relevancy.llm = ragas_llm

        ds = Dataset.from_dict({
            "question": [query],
            "answer": [answer],
            "contexts": [contexts],
        })
        result = evaluate(ds, metrics=[faithfulness, answer_relevancy])
        scores = result.to_pandas().iloc[0].to_dict()
        return {
            "faithfulness": float(scores.get("faithfulness", 0.0)),
            "answer_relevancy": float(scores.get("answer_relevancy", 0.0)),
        }
    except Exception:
        logger.exception("RAGAS scoring failed for query: %.60s", query)
        return {"faithfulness": 0.0, "answer_relevancy": 0.0}


def score_batch(
    items: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """
    Score a list of dicts, each with keys: query, answer, contexts (list[str]).
    Returns the same list with faithfulness and answer_relevancy added.
    """
    results = []
    for i, item in enumerate(items):
        logger.info("RAGAS scoring %d/%d", i + 1, len(items))
        scores = score_single(
            item["query"], item["answer"], item.get("contexts", [])
        )
        results.append({**item, **scores})
    return results
