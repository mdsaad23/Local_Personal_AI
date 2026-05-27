"""
GraphRAG retrieval — entity-anchored multi-hop graph traversal.

Strategy:
  1. Extract named entities from the query (spaCy).
  2. Find matching nodes in the NetworkX graph (text-match, case-insensitive).
  3. Traverse up to GRAPH_MAX_HOPS hops from each seed node.
  4. Collect doc_ids from all reached nodes + edges.
  5. Retrieve those chunks from LanceDB for the fusion step.
"""
from __future__ import annotations

import logging
from typing import Any

from config.settings import GRAPH_MAX_HOPS, GRAPH_TOP_K_ENTITIES

logger = logging.getLogger(__name__)


def _find_seed_nodes(g, query_entities: list[str]) -> list[str]:
    """Case-insensitive text match between query entities and graph node labels."""
    lower_entities = {e.lower() for e in query_entities}
    seeds = []
    for node_id, attrs in g.nodes(data=True):
        node_text = attrs.get("text", "").lower()
        if any(e in node_text or node_text in e for e in lower_entities):
            seeds.append(node_id)
    return seeds[:GRAPH_TOP_K_ENTITIES]


def _traverse(g, seed_nodes: list[str], max_hops: int) -> set[str]:
    """BFS from seed nodes up to max_hops. Returns set of node IDs."""
    visited: set[str] = set()
    frontier = set(seed_nodes)
    for _ in range(max_hops):
        if not frontier:
            break
        next_frontier: set[str] = set()
        for node in frontier:
            if node in visited:
                continue
            visited.add(node)
            next_frontier.update(g.successors(node))
            next_frontier.update(g.predecessors(node))
        frontier = next_frontier - visited
    return visited


def search_graph(query: str, top_k: int = 10) -> list[dict[str, Any]]:
    """
    Graph-based retrieval. Returns chunks whose documents are reachable
    from query entities in the knowledge graph.
    """
    from src.ingestion.graph_builder import get_graph, _extract_entities
    from src.retrieval.dense import _get_table

    g = get_graph()
    if g.number_of_nodes() == 0:
        logger.debug("Graph is empty — skipping graph retrieval")
        return []

    try:
        entities = _extract_entities(query)
    except Exception:
        logger.warning("NER failed during graph retrieval")
        return []

    if not entities:
        return []

    entity_texts = [e["text"] for e in entities]
    seeds = _find_seed_nodes(g, entity_texts)
    if not seeds:
        return []

    reached = _traverse(g, seeds, GRAPH_MAX_HOPS)

    # Collect doc_ids from reached nodes
    doc_ids: set[str] = set()
    for node_id in reached:
        for did in g.nodes[node_id].get("doc_ids", []):
            doc_ids.add(did)

    if not doc_ids:
        return []

    # Fetch chunks from LanceDB for those doc_ids
    try:
        table = _get_table()
        df = table.to_pandas()
        filtered = df[df["doc_id"].isin(doc_ids)].head(top_k)
        results = filtered.to_dict(orient="records")
    except Exception:
        logger.exception("Failed to fetch graph-retrieved chunks from LanceDB")
        return []

    return [
        {**r, "score": 1.0, "retrieval": "graph"}
        for r in results
    ]
