from __future__ import annotations

import asyncio
import logging
import tempfile
from pathlib import Path

from fastapi import APIRouter, BackgroundTasks, HTTPException, UploadFile, File
from pydantic import BaseModel

router = APIRouter(tags=["documents"])
logger = logging.getLogger(__name__)

# Track in-progress ingestions: key (filename hint or content doc_id) → progress record.
# Each record: {stage, progress (0-100), detail, error}.
_ingestion_status: dict[str, dict] = {}

# Stage → percent-complete, so the UI can render a real progress bar. building_graph
# (spaCy NER + LLM relation extraction) is the slow tail, hence the wide 80→100 band.
_STAGE_PROGRESS: dict[str, int] = {
    "queued":         5,
    "parsing":        20,
    "chunking":       40,
    "embedding":      60,
    "storing":        80,
    "building_graph": 92,
    "complete":       100,
}


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
    status: str = "complete"


def _run_ingestion(tmp_path: Path, filename: str, doc_id_hint: str) -> None:
    """Blocking ingestion — runs in a thread pool so the event loop stays free."""
    from src.ingestion.parsers import parse_file
    from src.ingestion.chunker import chunk_pages
    from src.ingestion.embedder import embed_chunks
    from src.retrieval.dense import store_chunks, delete_doc
    from src.retrieval.sparse import add_chunks_to_bm25
    from src.ingestion.graph_builder import build_graph_from_chunks

    # The client only knows doc_id_hint (the filename) until ingestion finishes,
    # so every status update must be written under the hint. Once the real
    # content-hash doc_id is known we mirror status under it too.
    keys = [doc_id_hint]
    stage = "queued"

    def _set(new_stage: str, detail: str = "") -> None:
        nonlocal stage
        stage = new_stage
        record = {
            "stage":    new_stage,
            "progress": _STAGE_PROGRESS.get(new_stage, 0),
            "detail":   detail,
            "error":    "",
        }
        for k in keys:
            _ingestion_status[k] = record
        logger.info("Ingest [%s] %s — %s%s", filename, new_stage,
                    f"{record['progress']}%", f" ({detail})" if detail else "")

    def _fail(message: str) -> None:
        # Preserve which stage we were in so the UI shows where it broke.
        record = {
            "stage":    "error",
            "progress": _STAGE_PROGRESS.get(stage, 0),
            "detail":   "",
            "error":    message,
            "failed_stage": stage,
        }
        for k in keys:
            _ingestion_status[k] = record
        logger.error("Ingest [%s] FAILED at %s — %s", filename, stage, message)

    try:
        _set("parsing")
        sections = parse_file(tmp_path)
        if not sections:
            _fail("No content extracted from the file (unreadable or empty).")
            return

        doc_id = sections[0].get("doc_id", doc_id_hint)
        if doc_id not in keys:
            keys.append(doc_id)

        # parse_file derives `source` from the temp file name — restore the
        # original upload filename so the document list shows a real name.
        for section in sections:
            section["source"] = filename

        _set("chunking", f"{len(sections)} section(s)")
        chunks = chunk_pages(sections)

        _set("embedding", f"{len(chunks)} chunk(s)")
        embedded = embed_chunks(chunks)
        if not embedded:
            _fail("No indexable content — the document produced no embeddable text.")
            logger.warning("No chunks produced for %s — nothing stored", filename)
            return

        _set("storing", f"{len(embedded)} chunk(s)")
        try:
            delete_doc(doc_id)
        except Exception:
            pass
        store_chunks(embedded)
        add_chunks_to_bm25(embedded)

        _set("building_graph")
        try:
            build_graph_from_chunks(embedded)
        except Exception as exc:
            logger.warning("Graph build skipped for %s: %s", filename, exc)

        _set("complete", f"{len(embedded)} chunk(s) indexed")
        logger.info("Ingestion complete: %s (%d chunks)", filename, len(embedded))

    except Exception as exc:
        logger.exception("Ingestion failed for %s", filename)
        _fail(f"{type(exc).__name__}: {exc}")
    finally:
        tmp_path.unlink(missing_ok=True)


@router.get("/documents", response_model=list[DocumentOut])
async def list_documents():
    from src.retrieval.dense import list_documents as _list
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(None, _list)


@router.get("/documents/status/{doc_id}")
async def ingestion_status(doc_id: str):
    record = _ingestion_status.get(doc_id)
    if record is None:
        return {"doc_id": doc_id, "stage": "unknown", "progress": 0, "detail": "", "error": ""}
    # `status` kept for backward compatibility with older clients (= stage string).
    return {"doc_id": doc_id, "status": record["stage"], **record}


@router.post("/documents", response_model=IngestResult)
async def upload_document(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
):
    suffix = Path(file.filename or "upload").suffix
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(await file.read())
        tmp_path = Path(tmp.name)

    # Use filename as a temporary key until real doc_id is known after parsing
    doc_id_hint = file.filename or "upload"
    _ingestion_status[doc_id_hint] = {
        "stage": "queued", "progress": _STAGE_PROGRESS["queued"], "detail": "", "error": "",
    }

    # Run blocking ingestion in thread pool — returns immediately to the client
    loop = asyncio.get_event_loop()
    background_tasks.add_task(
        loop.run_in_executor, None, _run_ingestion, tmp_path, file.filename or "upload", doc_id_hint
    )

    return IngestResult(
        doc_id=doc_id_hint,
        source=file.filename or "upload",
        sections=0,
        chunks=0,
        status="ingesting",
    )


@router.post("/documents/{doc_id}/reingest", response_model=IngestResult)
async def reingest_document(doc_id: str):
    """Re-run graph building on chunks already stored in LanceDB."""
    from src.retrieval.dense import _get_table
    from src.ingestion.graph_builder import build_graph_from_chunks

    try:
        loop = asyncio.get_event_loop()

        def _do():
            tbl = _get_table()
            rows = tbl.search().where(f"doc_id = '{doc_id}'").to_list()
            if not rows:
                raise ValueError("not_found")
            chunks = [{"text": r["text"], "doc_id": r["doc_id"],
                       "source": r.get("source", ""), "chunk_id": r.get("chunk_id", "")}
                      for r in rows]
            build_graph_from_chunks(chunks)
            return rows

        rows = await loop.run_in_executor(None, _do)
        return IngestResult(
            doc_id=doc_id,
            source=rows[0].get("source", doc_id),
            sections=0,
            chunks=len(rows),
            status="complete",
        )
    except ValueError:
        raise HTTPException(status_code=404, detail="Document not found in vector store")
    except Exception as exc:
        logger.exception("Reingest failed for %s", doc_id)
        raise HTTPException(status_code=500, detail=str(exc))


@router.delete("/documents/{doc_id}", status_code=204)
async def delete_document(doc_id: str):
    from src.retrieval.dense import delete_doc
    from src.retrieval.sparse import rebuild_index_from_lancedb

    def _do():
        delete_doc(doc_id)
        rebuild_index_from_lancedb()

    loop = asyncio.get_event_loop()
    await loop.run_in_executor(None, _do)
