"""
Dense vector retrieval via LanceDB ANN search.

Stores embedded chunks in a LanceDB table. Query with an embedding vector
to get top-K nearest neighbours.
"""
from __future__ import annotations

import logging
from typing import Any

import lancedb
import pyarrow as pa

from config.settings import LANCEDB_DIR, TOP_K_DENSE
from src.ingestion.embedder import embed_query

logger = logging.getLogger(__name__)

_TABLE_NAME = "chunks"
_db: lancedb.DBConnection | None = None
_table = None


def _get_table():
    global _db, _table
    if _table is not None:
        return _table

    _db = lancedb.connect(str(LANCEDB_DIR))

    schema = pa.schema([
        pa.field("chunk_id",       pa.string()),
        pa.field("doc_id",         pa.string()),
        pa.field("text",           pa.string()),
        pa.field("source",         pa.string()),
        pa.field("source_path",    pa.string()),
        pa.field("page",           pa.int32()),
        pa.field("section",        pa.string()),
        pa.field("file_type",      pa.string()),
        pa.field("chunk_index",    pa.int32()),
        pa.field("embedding",      pa.list_(pa.float32(), 768)),
    ])

    if _TABLE_NAME in _db.table_names():
        _table = _db.open_table(_TABLE_NAME)
    else:
        _table = _db.create_table(_TABLE_NAME, schema=schema)

    return _table


def store_chunks(chunks: list[dict[str, Any]]) -> None:
    """Write embedded chunks to LanceDB. Skips chunks already present (by chunk_id)."""
    if not chunks:
        return

    table = _get_table()

    rows = []
    for c in chunks:
        emb = c.get("embedding")
        if not emb:
            continue
        rows.append({
            "chunk_id":    c.get("chunk_id", ""),
            "doc_id":      c.get("doc_id", ""),
            "text":        c.get("text", ""),
            "source":      c.get("source", ""),
            "source_path": c.get("source_path", ""),
            "page":        int(c.get("page", 1)),
            "section":     c.get("section", ""),
            "file_type":   c.get("file_type", ""),
            "chunk_index": int(c.get("chunk_index", 0)),
            "embedding":   [float(x) for x in emb],
        })

    if rows:
        table.add(rows)
        logger.info("Stored %d chunks in LanceDB", len(rows))


def search_dense(query: str, top_k: int = TOP_K_DENSE) -> list[dict[str, Any]]:
    """ANN search. Returns top_k chunks sorted by vector similarity."""
    try:
        embedding = embed_query(query)
    except RuntimeError:
        logger.error("Dense search aborted — could not embed query")
        return []

    table = _get_table()
    try:
        results = (
            table.search(embedding)
                 .limit(top_k)
                 .to_list()
        )
    except Exception:
        logger.exception("LanceDB search failed")
        return []

    return [
        {
            "chunk_id":    r.get("chunk_id"),
            "doc_id":      r.get("doc_id"),
            "text":        r.get("text"),
            "source":      r.get("source"),
            "source_path": r.get("source_path"),
            "page":        r.get("page"),
            "section":     r.get("section"),
            "file_type":   r.get("file_type"),
            "chunk_index": r.get("chunk_index"),
            "score":       float(r.get("_distance", 0.0)),
            "retrieval":   "dense",
        }
        for r in results
    ]


def delete_doc(doc_id: str) -> None:
    """Remove all chunks for a document from LanceDB."""
    table = _get_table()
    try:
        table.delete(f"doc_id = '{doc_id}'")
        logger.info("Deleted doc %s from LanceDB", doc_id)
    except Exception:
        logger.exception("Failed to delete doc %s from LanceDB", doc_id)


def list_documents() -> list[dict[str, Any]]:
    """Return one row per unique doc_id with source name and chunk count."""
    table = _get_table()
    try:
        rows = table.to_pandas()[["doc_id", "source", "file_type"]].drop_duplicates("doc_id")
        counts = table.to_pandas().groupby("doc_id").size().reset_index(name="chunk_count")
        merged = rows.merge(counts, on="doc_id")
        return merged.to_dict(orient="records")
    except Exception:
        logger.exception("Failed to list documents")
        return []
