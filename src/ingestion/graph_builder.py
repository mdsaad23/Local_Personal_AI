"""
Entity extraction and graph construction for GraphRAG.

Pipeline per chunk:
  1. spaCy NER  → named entities (PERSON, ORG, GPE, DATE, MONEY …)
  2. LLM call   → subject-verb-object relationships between entity pairs
  3. NetworkX   → add nodes + edges; persist to GRAPH_PATH as JSON

Graph schema is forward-compatible with Phase 2 (email/calendar):
  Node types: PERSON, ORG, GPE, EVENT, TASK, DOCUMENT, CONCEPT
  Edge attrs:  relation, doc_id, chunk_id, confidence
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import httpx
import networkx as nx

from config.settings import (
    GRAPH_PATH,
    OLLAMA_BASE_URL,
    PRODUCTION_MODEL,
    SPACY_MODEL,
)

logger = logging.getLogger(__name__)

_graph: nx.DiGraph | None = None


# ---------------------------------------------------------------------------
# Graph persistence
# ---------------------------------------------------------------------------

def _load_graph() -> nx.DiGraph:
    global _graph
    if _graph is not None:
        return _graph
    if GRAPH_PATH.exists():
        data = json.loads(GRAPH_PATH.read_text(encoding="utf-8"))
        # NB: omit the edge-key kwarg — its name changed across NetworkX
        # versions (3.3 uses `link`, 3.4+ uses `edges`). The default key
        # ("links") is stable, and save/load here both rely on it.
        _graph = nx.node_link_graph(data)
    else:
        _graph = nx.DiGraph()
    return _graph


def _save_graph(g: nx.DiGraph) -> None:
    tmp = GRAPH_PATH.with_suffix(".tmp")
    data = nx.node_link_data(g)
    tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(GRAPH_PATH)  # atomic rename — crash-safe


def get_graph() -> nx.DiGraph:
    return _load_graph()


# ---------------------------------------------------------------------------
# NER via spaCy
# ---------------------------------------------------------------------------

_nlp = None

def _get_nlp():
    global _nlp
    if _nlp is None:
        import spacy
        try:
            _nlp = spacy.load(SPACY_MODEL)
        except OSError:
            logger.error(
                "spaCy model '%s' not found. Run: python -m spacy download %s",
                SPACY_MODEL, SPACY_MODEL,
            )
            raise
    return _nlp


def _extract_entities(text: str) -> list[dict[str, str]]:
    nlp = _get_nlp()
    doc = nlp(text[:100_000])  # spaCy limit guard
    seen: set[str] = set()
    entities: list[dict[str, str]] = []
    for ent in doc.ents:
        key = (ent.text.strip(), ent.label_)
        if key not in seen and ent.text.strip():
            seen.add(key)
            entities.append({"text": ent.text.strip(), "label": ent.label_})
    return entities


# ---------------------------------------------------------------------------
# Relationship extraction via LLM
# ---------------------------------------------------------------------------

_REL_PROMPT = """\
You are an information extraction assistant. Given the TEXT and a list of ENTITIES, extract subject-verb-object relationships.

Rules:
- Only use entities from the ENTITIES list as subject or object.
- Relation must be a short verb phrase (max 5 words).
- Return JSON array only. No prose. Example: [{{"subject": "Alice", "relation": "works at", "object": "ACME Corp"}}]
- If no clear relationships exist, return [].

ENTITIES: {entities}

TEXT:
{text}

JSON:"""


def _extract_relationships(
    text: str, entities: list[dict[str, str]]
) -> list[dict[str, str]]:
    if len(entities) < 2:
        return []

    entity_names = [e["text"] for e in entities[:20]]  # cap to avoid huge prompts
    prompt = _REL_PROMPT.format(
        entities=", ".join(entity_names),
        text=text[:2000],
    )
    try:
        resp = httpx.post(
            f"{OLLAMA_BASE_URL}/api/generate",
            json={"model": PRODUCTION_MODEL, "prompt": prompt, "stream": False,
                  "options": {"temperature": 0}},
            timeout=60,
        )
        resp.raise_for_status()
        raw = resp.json().get("response", "").strip()
        # Extract JSON array from response
        start = raw.find("[")
        end = raw.rfind("]") + 1
        if start == -1 or end == 0:
            return []
        return json.loads(raw[start:end])
    except Exception:
        logger.debug("Relationship extraction failed — skipping", exc_info=True)
        return []


# ---------------------------------------------------------------------------
# Graph update
# ---------------------------------------------------------------------------

def build_graph_from_chunks(chunks: list[dict[str, Any]]) -> None:
    """
    Process a list of embedded chunks, extract entities + relationships,
    and update the persistent graph. Call once per document after embedding.
    """
    g = _load_graph()
    modified = False

    for chunk in chunks:
        text = chunk.get("text", "")
        doc_id = chunk.get("doc_id", "")
        chunk_index = chunk.get("chunk_index", 0)
        source = chunk.get("source", "")

        try:
            entities = _extract_entities(text)
        except Exception:
            logger.warning("NER failed on chunk %s:%d", source, chunk_index)
            continue

        # Add entity nodes
        for ent in entities:
            node_id = f"{ent['label']}:{ent['text']}"
            if not g.has_node(node_id):
                g.add_node(node_id, label=ent["label"], text=ent["text"],
                           doc_ids=[doc_id], sources=[source])
            else:
                node = g.nodes[node_id]
                if doc_id not in node.get("doc_ids", []):
                    node.setdefault("doc_ids", []).append(doc_id)
                if source not in node.get("sources", []):
                    node.setdefault("sources", []).append(source)
            modified = True

        # Extract and add relationships
        relationships = _extract_relationships(text, entities)
        for rel in relationships:
            subj = rel.get("subject", "").strip()
            obj = rel.get("object", "").strip()
            relation = rel.get("relation", "").strip()
            if not (subj and obj and relation):
                continue
            # Find matching node IDs (text match)
            subj_id = next(
                (n for n in g.nodes if g.nodes[n].get("text") == subj), None
            )
            obj_id = next(
                (n for n in g.nodes if g.nodes[n].get("text") == obj), None
            )
            if subj_id and obj_id:
                g.add_edge(subj_id, obj_id, relation=relation,
                           doc_id=doc_id, chunk_index=chunk_index,
                           confidence=1.0)
                modified = True

    if modified:
        _save_graph(g)
        logger.info("Graph updated: %d nodes, %d edges", g.number_of_nodes(), g.number_of_edges())
