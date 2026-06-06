from __future__ import annotations

import tempfile
from pathlib import Path

from fastapi import APIRouter, HTTPException, UploadFile, File
from pydantic import BaseModel

router = APIRouter(tags=["documents"])


class DocumentOut(BaseModel):
    doc_id: str
    source: str
    file_type: str
    chunk_count: int


class IngestResult(BaseModel):
    doc_id: str
    source: str
    sections: int
    chunks: int


@router.get("/documents", response_model=list[DocumentOut])
async def list_documents():
    from src.retrieval.dense import list_documents as _list
    return _list()


@router.post("/documents", response_model=IngestResult)
async def upload_document(file: UploadFile = File(...)):
    import logging
    from src.ingestion.parsers import parse_file
    from src.ingestion.chunker import chunk_pages
    from src.ingestion.embedder import embed_chunks
    from src.retrieval.dense import store_chunks, delete_doc
    from src.retrieval.sparse import add_chunks_to_bm25
    from src.ingestion.graph_builder import build_graph_from_chunks

    logger = logging.getLogger(__name__)
    suffix = Path(file.filename or "upload").suffix
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(await file.read())
        tmp_path = Path(tmp.name)

    try:
        sections = parse_file(tmp_path)
        if not sections:
            raise HTTPException(status_code=422, detail="No content extracted from file")

        chunks = chunk_pages(sections)
        embedded = embed_chunks(chunks)
        doc_id = sections[0].get("doc_id", "") if sections else ""

        # Remove any previous partial ingest for this doc_id before storing
        if doc_id:
            try:
                delete_doc(doc_id)
            except Exception:
                pass

        store_chunks(embedded)
        add_chunks_to_bm25(embedded)

        # Graph building is best-effort — spaCy may not be installed
        try:
            build_graph_from_chunks(embedded)
        except Exception as exc:
            logger.warning("Graph build skipped for %s: %s", file.filename, exc)

        return IngestResult(
            doc_id=doc_id,
            source=file.filename or "upload",
            sections=len(sections),
            chunks=len(embedded),
        )
    finally:
        tmp_path.unlink(missing_ok=True)


@router.post("/documents/{doc_id}/reingest", response_model=IngestResult)
async def reingest_document(doc_id: str):
    """Re-run graph building on chunks already stored in LanceDB."""
    import logging
    from src.retrieval.dense import _get_table
    from src.ingestion.graph_builder import build_graph_from_chunks

    logger = logging.getLogger(__name__)
    try:
        tbl = _get_table()
        rows = tbl.search().where(f"doc_id = '{doc_id}'").to_list()
        if not rows:
            raise HTTPException(status_code=404, detail="Document not found in vector store")

        chunks = [{"text": r["text"], "doc_id": r["doc_id"],
                   "source": r.get("source", ""), "chunk_id": r.get("chunk_id", "")}
                  for r in rows]
        build_graph_from_chunks(chunks)
        return IngestResult(
            doc_id=doc_id,
            source=rows[0].get("source", doc_id),
            sections=0,
            chunks=len(chunks),
        )
    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("Reingest failed for %s", doc_id)
        raise HTTPException(status_code=500, detail=str(exc))


@router.delete("/documents/{doc_id}", status_code=204)
async def delete_document(doc_id: str):
    from src.retrieval.dense import delete_doc
    delete_doc(doc_id)
