"""
DeepEval hallucination + contextual metrics.

Extends the RAGAS faithfulness/relevancy scores with:
  - Hallucination score: fraction of response claims NOT grounded in context
    (lower is better; 0.0 = no hallucination)
  - Contextual precision: retrieved chunks that are actually relevant
  - Contextual recall: answer coverage across retrieved chunks

Uses the production Qwen3 model as judge (local, no API cost).
DeepEval is optional — if not installed the scores default to None.

Install: pip install deepeval
"""
from __future__ import annotations

import logging
from typing import Any

from config.settings import PRODUCTION_MODEL

logger = logging.getLogger(__name__)


def _build_judge():
    """Return a DeepEval-compatible Ollama judge model wrapper."""
    try:
        from deepeval.models import OllamaModel
        return OllamaModel(model=PRODUCTION_MODEL)
    except ImportError:
        return None


def score_hallucination(
    query: str,
    answer: str,
    contexts: list[str],
    judge=None,
) -> float | None:
    """
    Hallucination score for one (query, answer, contexts) triple.
    Returns 0.0–1.0 where 0.0 = no hallucination, 1.0 = fully hallucinated.
    Returns None if deepeval is not installed or scoring fails.
    """
    try:
        from deepeval.metrics import HallucinationMetric
        from deepeval.test_case import LLMTestCase

        _judge = judge or _build_judge()
        if _judge is None:
            return None

        metric = HallucinationMetric(threshold=0.5, model=_judge, verbose_mode=False)
        test_case = LLMTestCase(
            input=query,
            actual_output=answer,
            context=contexts,
        )
        metric.measure(test_case)
        # DeepEval reports hallucination as fraction of hallucinated claims (0–1)
        return round(float(metric.score), 4)
    except ImportError:
        logger.debug("deepeval not installed — skipping hallucination metric")
        return None
    except Exception:
        logger.exception("DeepEval hallucination scoring failed")
        return None


def score_contextual(
    query: str,
    answer: str,
    contexts: list[str],
    expected_output: str = "",
    judge=None,
) -> dict[str, float | None]:
    """
    Contextual precision and recall for one triple.
    Returns {"contextual_precision": 0–1, "contextual_recall": 0–1}.
    Both return None if deepeval is not installed or scoring fails.
    """
    out: dict[str, float | None] = {
        "contextual_precision": None,
        "contextual_recall": None,
    }
    try:
        from deepeval.metrics import ContextualPrecisionMetric, ContextualRecallMetric
        from deepeval.test_case import LLMTestCase

        _judge = judge or _build_judge()
        if _judge is None:
            return out

        expected = expected_output or answer  # use answer as proxy if no ground truth

        test_case = LLMTestCase(
            input=query,
            actual_output=answer,
            expected_output=expected,
            retrieval_context=contexts,
        )

        precision_metric = ContextualPrecisionMetric(
            threshold=0.5, model=_judge, verbose_mode=False
        )
        recall_metric = ContextualRecallMetric(
            threshold=0.5, model=_judge, verbose_mode=False
        )

        precision_metric.measure(test_case)
        recall_metric.measure(test_case)

        out["contextual_precision"] = round(float(precision_metric.score), 4)
        out["contextual_recall"]    = round(float(recall_metric.score), 4)
    except ImportError:
        logger.debug("deepeval not installed — skipping contextual metrics")
    except Exception:
        logger.exception("DeepEval contextual scoring failed")
    return out


def score_all(
    query: str,
    answer: str,
    contexts: list[str],
    expected_output: str = "",
) -> dict[str, Any]:
    """
    Run all DeepEval metrics for one (query, answer, contexts) triple.
    Returns a flat dict — all keys are None if deepeval is not available.
    """
    judge = _build_judge()
    hallucination = score_hallucination(query, answer, contexts, judge)
    contextual = score_contextual(query, answer, contexts, expected_output, judge)
    return {
        "hallucination": hallucination,
        **contextual,
    }
