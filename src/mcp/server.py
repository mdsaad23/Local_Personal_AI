"""
MCP server — exposes the knowledge base to Claude Desktop.

Tools:
  query_kb   — ask a question against the full RAG pipeline
  list_docs  — list ingested documents
  add_doc    — ingest a file by path

Run with: python src/mcp/server.py
Configure in Claude Desktop: add the server entry to claude_desktop_config.json
"""
from __future__ import annotations

import logging
import sys
from pathlib import Path

from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import TextContent, Tool

logging.basicConfig(level=logging.INFO, stream=sys.stderr)
logger = logging.getLogger(__name__)

app = Server("local-ai-kb")


@app.list_tools()
async def list_tools() -> list[Tool]:
    return [
        Tool(
            name="query_kb",
            description=(
                "Query the local knowledge base using hybrid RAG retrieval "
                "(dense vector + BM25 + GraphRAG + reranker). "
                "Returns an answer grounded in the user's documents."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "The question to answer"},
                },
                "required": ["query"],
            },
        ),
        Tool(
            name="list_docs",
            description="List all documents currently ingested in the knowledge base.",
            inputSchema={"type": "object", "properties": {}},
        ),
        Tool(
            name="add_doc",
            description="Ingest a document file into the knowledge base by providing its file path.",
            inputSchema={
                "type": "object",
                "properties": {
                    "file_path": {
                        "type": "string",
                        "description": "Absolute path to the document file to ingest",
                    },
                },
                "required": ["file_path"],
            },
        ),
    ]


@app.call_tool()
async def call_tool(name: str, arguments: dict) -> list[TextContent]:
    if name == "query_kb":
        return await _query_kb(arguments.get("query", ""))
    elif name == "list_docs":
        return await _list_docs()
    elif name == "add_doc":
        return await _add_doc(arguments.get("file_path", ""))
    else:
        return [TextContent(type="text", text=f"Unknown tool: {name}")]


async def _query_kb(query: str) -> list[TextContent]:
    if not query.strip():
        return [TextContent(type="text", text="Empty query.")]

    from src.generation.router import retrieve_for_query, build_prompt
    from src.generation.ollama_client import generate_sync
    from src.memory.episodic import retrieve_relevant

    chunks, route = retrieve_for_query(query)
    memories = retrieve_relevant(query, limit=3)
    messages = build_prompt(query, chunks, memories)

    try:
        answer = generate_sync(messages)
    except Exception as e:
        return [TextContent(type="text", text=f"Generation failed: {e}")]

    sources = list({c.get("source", "unknown") for c in chunks})
    source_line = f"\n\n*Sources: {', '.join(sources)}*" if sources else ""
    return [TextContent(type="text", text=answer + source_line)]


async def _list_docs() -> list[TextContent]:
    from src.retrieval.dense import list_documents
    docs = list_documents()
    if not docs:
        return [TextContent(type="text", text="No documents ingested yet.")]
    lines = [f"- **{d['source']}** ({d['file_type']}, {d['chunk_count']} chunks)" for d in docs]
    return [TextContent(type="text", text="\n".join(lines))]


async def _add_doc(file_path: str) -> list[TextContent]:
    path = Path(file_path)
    if not path.exists():
        return [TextContent(type="text", text=f"File not found: {file_path}")]

    from src.ingestion.parsers import parse_file
    from src.ingestion.chunker import chunk_pages
    from src.ingestion.embedder import embed_chunks
    from src.retrieval.dense import store_chunks
    from src.retrieval.sparse import add_chunks_to_bm25
    from src.ingestion.graph_builder import build_graph_from_chunks

    try:
        sections = parse_file(path)
        chunks = chunk_pages(sections)
        embedded = embed_chunks(chunks)
        store_chunks(embedded)
        add_chunks_to_bm25(embedded)
        build_graph_from_chunks(embedded)
        return [TextContent(
            type="text",
            text=f"Ingested: {path.name} → {len(sections)} sections → {len(embedded)} chunks",
        )]
    except Exception as e:
        return [TextContent(type="text", text=f"Ingestion failed: {e}")]


async def main() -> None:
    async with stdio_server() as (read_stream, write_stream):
        await app.run(read_stream, write_stream, app.create_initialization_options())


if __name__ == "__main__":
    import asyncio
    asyncio.run(main())
