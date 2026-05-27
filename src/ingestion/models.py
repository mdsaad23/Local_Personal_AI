"""Shared data models for the ingestion pipeline."""
from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class ParsedDocument:
    doc_id: str                      # SHA256 of raw file bytes
    source_path: Path
    file_type: str                   # pdf, docx, md, txt, image
    title: str
    markdown_content: str            # full content as markdown
    page_count: int = 1
    metadata: dict = field(default_factory=dict)


@dataclass
class Chunk:
    chunk_id: str
    doc_id: str
    text: str
    source_path: str
    page_num: int
    section_heading: str
    chunk_index: int
    char_count: int
    embedding: list[float] | None = None

    @classmethod
    def make(
        cls,
        text: str,
        doc_id: str,
        source_path: str,
        chunk_index: int,
        page_num: int = 0,
        section_heading: str = "",
    ) -> "Chunk":
        chunk_id = hashlib.sha256(
            f"{doc_id}:{chunk_index}:{text[:64]}".encode()
        ).hexdigest()[:16]
        return cls(
            chunk_id=chunk_id,
            doc_id=doc_id,
            text=text,
            source_path=source_path,
            page_num=page_num,
            section_heading=section_heading,
            chunk_index=chunk_index,
            char_count=len(text),
        )


@dataclass
class Entity:
    text: str
    label: str       # spaCy label: PERSON, ORG, DATE, GPE, MONEY …
    doc_id: str
    chunk_id: str
    start_char: int = 0
    end_char: int = 0


@dataclass
class Relationship:
    source: str      # entity text
    relation: str    # verb / relation type
    target: str      # entity text
    doc_id: str
    confidence: float = 1.0
