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
    from src.ingestion.parsers import parse_file
    from src.ingestion.chunker import chunk_pages
    from src.ingestion.embedder import embed_chunks
    from src.retrieval.dense import store_chunks
    from src.retrieval.sparse import add_chunks_to_bm25
    from src.ingestion.graph_builder import build_graph_from_chunks

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
        store_chunks(embedded)
        add_chunks_to_bm25(embedded)
        build_graph_from_chunks(embedded)
        doc_id = sections[0].get("doc_id", "") if sections else ""
        return IngestResult(
            doc_id=doc_id,
            source=file.filename or "upload",
            sections=len(sections),
            chunks=len(embedded),
        )
    finally:
        tmp_path.unlink(missing_ok=True)


@router.delete("/documents/{doc_id}", status_code=204)
async def delete_document(doc_id: str):
    from src.retrieval.dense import delete_doc
    delete_doc(doc_id)
