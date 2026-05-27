"""
Folder watcher — monitors DOCUMENTS_DIR and triggers ingestion on new files.

Uses watchdog. Run as a standalone process:
    python src/ingestion/watcher.py

Pipeline per new file:
  parse → chunk → embed → store in LanceDB + BM25 + graph
"""
from __future__ import annotations

import logging
import sys
import time
from pathlib import Path

from watchdog.events import FileCreatedEvent, FileMovedEvent, FileSystemEventHandler
from watchdog.observers import Observer

from config.settings import DOCUMENTS_DIR, SUPPORTED_EXTENSIONS

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)

# Imported lazily to avoid slow startup when running just the watcher
_pipeline_ready = False


def _run_pipeline(path: Path) -> None:
    global _pipeline_ready
    if not _pipeline_ready:
        # Deferred imports — keeps watcher startup fast
        _pipeline_ready = True

    from src.ingestion.parsers import parse_file
    from src.ingestion.chunker import chunk_pages
    from src.ingestion.embedder import embed_chunks
    from src.retrieval.dense import store_chunks
    from src.retrieval.sparse import add_chunks_to_bm25
    from src.ingestion.graph_builder import build_graph_from_chunks

    logger.info("Ingesting: %s", path.name)
    try:
        sections = parse_file(path)
        if not sections:
            logger.warning("No content extracted from %s", path.name)
            return

        chunks = chunk_pages(sections)
        embedded = embed_chunks(chunks)
        if not embedded:
            logger.warning("Embedding failed for all chunks in %s", path.name)
            return

        store_chunks(embedded)
        add_chunks_to_bm25(embedded)
        build_graph_from_chunks(embedded)

        logger.info(
            "Ingested %s → %d sections → %d chunks → %d embedded",
            path.name, len(sections), len(chunks), len(embedded),
        )
    except Exception:
        logger.exception("Pipeline failed for %s", path.name)


class _IngestionHandler(FileSystemEventHandler):
    def _handle(self, path_str: str) -> None:
        path = Path(path_str)
        if path.suffix.lower() not in SUPPORTED_EXTENSIONS:
            return
        if path.name.startswith("."):
            return
        # Small delay — wait for file write to complete before parsing
        time.sleep(1)
        _run_pipeline(path)

    def on_created(self, event: FileCreatedEvent) -> None:
        if not event.is_directory:
            self._handle(event.src_path)

    def on_moved(self, event: FileMovedEvent) -> None:
        if not event.is_directory:
            self._handle(event.dest_path)


def run_watcher(watch_dir: Path = DOCUMENTS_DIR) -> None:
    watch_dir.mkdir(parents=True, exist_ok=True)
    observer = Observer()
    observer.schedule(_IngestionHandler(), str(watch_dir), recursive=False)
    observer.start()
    logger.info("Watching %s for new documents …", watch_dir)
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        observer.stop()
    observer.join()


if __name__ == "__main__":
    run_watcher()
