"""
Reciprocal Rank Fusion — merges dense, sparse, and graph result lists.

RRF(d) = Σ 1 / (k + rank_i(d))   where k=60 (standard constant)

Why RRF over score normalisation:
  Each retriever produces scores on incompatible scales (distance, BM25,
  hop count). Normalising across scales introduces noise. RRF only uses rank
  order, which is stable regardless of score magnitude.
"""
from __future__ import annotations

from collections import defaultdict
from typing import Any

from config.settings import RRF_K


def reciprocal_rank_fusion(
    *result_lists: list[dict[str, Any]],
    k: int = RRF_K,
) -> list[dict[str, Any]]:
    """
    Fuse multiple ranked result lists into a single list sorted by RRF score.
    Each list must have 'chunk_id' and 'text' keys.
    """
    rrf_scores: dict[str, float] = defaultdict(float)
    chunk_by_id: dict[str, dict[str, Any]] = {}

    for result_list in result_lists:
        for rank, chunk in enumerate(result_list):
            cid = chunk.get("chunk_id") or chunk.get("text", "")[:64]
            rrf_scores[cid] += 1.0 / (k + rank + 1)
            if cid not in chunk_by_id:
                chunk_by_id[cid] = chunk

    fused = sorted(
        [
            {**chunk_by_id[cid], "rrf_score": score}
            for cid, score in rrf_scores.items()
        ],
        key=lambda x: x["rrf_score"],
        reverse=True,
    )
    return fused
