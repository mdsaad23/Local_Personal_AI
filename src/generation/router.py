"""
Adaptive query router — classifies the query and dispatches to the right pipeline.

Route classes:
  DIRECT   — general knowledge, no retrieval needed (greetings, math, coding help)
  RAG      — document lookup, standard hybrid retrieval
  GRAPH    — relational / multi-hop query (who/what is connected to X)
  HYDE_RAG — analytical query where HyDE improves recall

The classifier uses a lightweight heuristic first; falls back to LLM
classification for ambiguous queries. This adds <5ms for heuristic path,
~300ms for LLM path.
"""
from __future__ import annotations

import logging
import re
from enum import Enum
from typing import Any

from config.settings import HYDE_ENABLED

logger = logging.getLogger(__name__)


class RouteType(str, Enum):
    DIRECT   = "direct"
    RAG      = "rag"
    GRAPH    = "graph"
    HYDE_RAG = "hyde_rag"


# Patterns that strongly suggest relational / graph queries
_GRAPH_PATTERNS = re.compile(
    r"\b(who|connected|relationship|linked|associated|related|knows|works with|"
    r"between .+ and|across .+ document|all mention|every|mentioned in)\b",
    re.IGNORECASE,
)

# Patterns that suggest no retrieval is needed
_DIRECT_PATTERNS = re.compile(
    r"\b(hello|hi|thanks|what is \d|how do i|write code|calculate|convert|"
    r"translate|explain .* concept|what does .* mean in general)\b",
    re.IGNORECASE,
)

# Analytical / synthesis queries benefit from HyDE
_HYDE_PATTERNS = re.compile(
    r"\b(summarise|summarize|compare|contrast|analyse|analyze|what are the key|"
    r"implications of|impact of|recommend|best approach|pros and cons)\b",
    re.IGNORECASE,
)


def classify(query: str) -> RouteType:
    """Heuristic-first query classification."""
    q = query.strip()

    if _DIRECT_PATTERNS.search(q) and len(q.split()) < 8:
        return RouteType.DIRECT

    if _GRAPH_PATTERNS.search(q):
        return RouteType.GRAPH

    if HYDE_ENABLED and _HYDE_PATTERNS.search(q):
        return RouteType.HYDE_RAG

    return RouteType.RAG


def build_prompt(
    query: str,
    chunks: list[dict[str, Any]],
    memories: list[str],
    *,
    system_suffix: str = "",
) -> list[dict[str, str]]:
    """
    Assemble the final message list for the LLM.
    Relevant memories → system prompt. Retrieved chunks → context block.
    """
    system_parts = [
        "You are a helpful personal AI assistant with access to the user's document knowledge base.",
        "Answer using the provided CONTEXT. If the context does not contain the answer, say so clearly.",
        "Always cite the source document name when referencing retrieved content.",
    ]

    if memories:
        mem_block = "\n".join(f"- {m}" for m in memories)
        system_parts.append(f"\nRELEVANT PAST CONTEXT (from previous conversations):\n{mem_block}")

    if system_suffix:
        system_parts.append(system_suffix)

    system_msg = {"role": "system", "content": "\n".join(system_parts)}

    if chunks:
        context_lines = []
        for i, c in enumerate(chunks):
            source = c.get("source", "unknown")
            page = c.get("page", "")
            page_str = f" p.{page}" if page else ""
            context_lines.append(f"[{i+1}] [{source}{page_str}]\n{c['text']}")
        context_block = "\n\n---\n\n".join(context_lines)
        user_content = f"CONTEXT:\n{context_block}\n\nQUESTION: {query}"
    else:
        user_content = query

    return [system_msg, {"role": "user", "content": user_content}]


def retrieve_for_query(query: str) -> tuple[list[dict[str, Any]], RouteType]:
    """
    Run the retrieval pipeline appropriate for this query type.
    Returns (chunks, route_used).
    """
    from src.retrieval.dense import search_dense
    from src.retrieval.sparse import search_sparse
    from src.retrieval.graph_retrieval import search_graph
    from src.retrieval.fusion import reciprocal_rank_fusion
    from src.retrieval.reranker import rerank
    from src.retrieval.hyde import generate_hypothetical_document

    route = classify(query)
    logger.info("Route: %s for query: %.80s", route, query)

    if route == RouteType.DIRECT:
        return [], route

    if route == RouteType.GRAPH:
        graph_results = search_graph(query)
        dense_results = search_dense(query)
        fused = reciprocal_rank_fusion(dense_results, graph_results)
        return rerank(query, fused), route

    if route == RouteType.HYDE_RAG:
        hyp_doc = generate_hypothetical_document(query)
        dense_results = search_dense(hyp_doc)
        sparse_results = search_sparse(query)
        fused = reciprocal_rank_fusion(dense_results, sparse_results)
        return rerank(query, fused), route

    # Default: RAG
    dense_results = search_dense(query)
    sparse_results = search_sparse(query)
    fused = reciprocal_rank_fusion(dense_results, sparse_results)
    return rerank(query, fused), route
