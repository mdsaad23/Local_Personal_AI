"""
Document parsers for every supported file type.

PDF / DOCX  → Docling (structure-preserving, full-doc markdown export)
MD / TXT    → direct read
Images      → Ollama vision model description (minicpm-v)

Each parse_* function returns a list of section dicts:
    {text, source, source_path, page, section, doc_id, file_type}

A document may yield multiple sections (one per heading / page group).
The chunker further splits each section into fixed-size overlapping chunks.
"""
from __future__ import annotations

import base64
import hashlib
import re
from pathlib import Path
from typing import Any

import httpx
from loguru import logger

from config.settings import (
    IMAGE_EXTENSIONS,
    OLLAMA_BASE_URL,
    SUPPORTED_EXTENSIONS,
    VISION_MODEL,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(65536), b""):
            h.update(block)
    return h.hexdigest()


def _section(
    text: str,
    path: Path,
    doc_id: str,
    file_type: str,
    page: int = 1,
    section: str = "",
    **extra: Any,
) -> dict[str, Any]:
    return {
        "text": text.strip(),
        "source": path.name,
        "source_path": str(path),
        "page": page,
        "section": section,
        "doc_id": doc_id,
        "file_type": file_type,
        **extra,
    }


# ---------------------------------------------------------------------------
# PDF via Docling
# ---------------------------------------------------------------------------

def parse_pdf(path: Path, doc_id: str) -> list[dict[str, Any]]:
    from docling.document_converter import DocumentConverter

    converter = DocumentConverter()
    result = converter.convert(str(path))
    doc = result.document

    # Export entire document as markdown — Docling preserves heading hierarchy,
    # tables, and captions. The chunker handles the size splitting.
    markdown = doc.export_to_markdown()
    if not markdown.strip():
        return []

    # Split on top-level headings to create logical sections for metadata
    sections = _split_markdown_sections(markdown, path, doc_id, "pdf")
    return sections if sections else [_section(markdown, path, doc_id, "pdf")]


# ---------------------------------------------------------------------------
# DOCX via python-docx
# ---------------------------------------------------------------------------

def parse_docx(path: Path, doc_id: str) -> list[dict[str, Any]]:
    import docx as python_docx

    doc = python_docx.Document(str(path))
    sections: list[dict[str, Any]] = []
    current_heading = ""
    current_lines: list[str] = []

    def flush() -> None:
        text = "\n".join(current_lines).strip()
        if text:
            sections.append(_section(
                text, path, doc_id, "docx",
                page=len(sections) + 1,
                section=current_heading,
            ))

    for para in doc.paragraphs:
        style = para.style.name
        if style.startswith("Heading") and para.text.strip():
            flush()
            current_heading = para.text.strip()
            current_lines = []
        elif para.text.strip():
            current_lines.append(para.text.strip())

    flush()
    return sections


# ---------------------------------------------------------------------------
# Markdown
# ---------------------------------------------------------------------------

def parse_md(path: Path, doc_id: str) -> list[dict[str, Any]]:
    raw = path.read_text(encoding="utf-8", errors="replace")
    sections = _split_markdown_sections(raw, path, doc_id, "md")
    return sections if sections else [_section(raw, path, doc_id, "md")]


# ---------------------------------------------------------------------------
# Plain text
# ---------------------------------------------------------------------------

def parse_txt(path: Path, doc_id: str) -> list[dict[str, Any]]:
    text = path.read_text(encoding="utf-8", errors="replace").strip()
    return [_section(text, path, doc_id, "txt")] if text else []


# ---------------------------------------------------------------------------
# Images — describe with vision model
# ---------------------------------------------------------------------------

_IMAGE_PROMPT = (
    "Describe this image comprehensively for a document search index. "
    "Transcribe all visible text exactly. Summarise any charts, tables, or "
    "diagrams with their key values. Describe people, objects, and overall "
    "context. Be thorough — this description is the only representation of "
    "the image in the retrieval system."
)


def parse_image(path: Path, doc_id: str) -> list[dict[str, Any]]:
    b64 = base64.b64encode(path.read_bytes()).decode()
    try:
        resp = httpx.post(
            f"{OLLAMA_BASE_URL}/api/generate",
            json={"model": VISION_MODEL, "prompt": _IMAGE_PROMPT,
                  "images": [b64], "stream": False},
            timeout=120,
        )
        resp.raise_for_status()
        description = resp.json().get("response", "").strip()
    except Exception as exc:
        logger.warning(f"Vision model failed for {path.name}: {exc}. Falling back to filename.")
        description = ""

    text = f"[IMAGE: {path.name}]\n{description or path.name}"
    return [_section(text, path, doc_id, path.suffix.lstrip("."), section="image", is_image=True)]


# ---------------------------------------------------------------------------
# Markdown section splitter (shared by PDF and MD parsers)
# ---------------------------------------------------------------------------

def _split_markdown_sections(
    markdown: str, path: Path, doc_id: str, file_type: str
) -> list[dict[str, Any]]:
    """Split markdown on H1/H2 headings into logical sections."""
    parts = re.split(r"(?m)^(#{1,2} .+)$", markdown)
    sections: list[dict[str, Any]] = []
    current_heading = ""
    current_chunks: list[str] = []

    def flush(heading: str, parts_list: list[str]) -> None:
        text = "\n".join(parts_list).strip()
        if text:
            sections.append(_section(
                text, path, doc_id, file_type,
                page=len(sections) + 1,
                section=heading.lstrip("#").strip(),
            ))

    for part in parts:
        if re.match(r"^#{1,2} ", part):
            flush(current_heading, current_chunks)
            current_heading = part
            current_chunks = []
        else:
            current_chunks.append(part)

    flush(current_heading, current_chunks)
    return sections


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def parse_file(path: Path) -> list[dict[str, Any]]:
    """
    Dispatch to the correct parser based on file extension.
    Returns a list of section dicts ready for the chunker.
    Returns [] if the file is unsupported, missing, or parsing fails.
    """
    if not path.exists() or not path.is_file():
        logger.error(f"Not found: {path}")
        return []

    suffix = path.suffix.lower()
    if suffix not in SUPPORTED_EXTENSIONS:
        logger.warning(f"Unsupported type: {path.name}")
        return []

    doc_id = _sha256(path)
    logger.info(f"Parsing [{suffix}] {path.name}")

    try:
        if suffix == ".pdf":
            return parse_pdf(path, doc_id)
        elif suffix == ".docx":
            return parse_docx(path, doc_id)
        elif suffix == ".md":
            return parse_md(path, doc_id)
        elif suffix == ".txt":
            return parse_txt(path, doc_id)
        elif suffix in IMAGE_EXTENSIONS:
            return parse_image(path, doc_id)
    except Exception:
        logger.exception(f"Parser crashed on {path.name}")

    return []
