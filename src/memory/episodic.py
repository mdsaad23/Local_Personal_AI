"""
Episodic memory — extracts structured facts from conversations and retrieves
relevant past memories to inject into new sessions.

Uses Mem0 with a local SQLite backend. Facts are extracted with the production
LLM and stored as searchable memories keyed to a user_id.
"""
from __future__ import annotations

import logging
from typing import Any

from config.settings import MEMORY_DB_PATH, EPISODIC_INJECT_COUNT, OLLAMA_BASE_URL, PRODUCTION_MODEL

logger = logging.getLogger(__name__)

_DEFAULT_USER = "local_user"
_mem0_client = None


def _get_client():
    global _mem0_client
    if _mem0_client is not None:
        return _mem0_client

    try:
        from mem0 import Memory

        config = {
            "llm": {
                "provider": "ollama",
                "config": {
                    "model": PRODUCTION_MODEL,
                    "ollama_base_url": OLLAMA_BASE_URL,
                    "temperature": 0,
                    "max_tokens": 2000,
                },
            },
            "embedder": {
                "provider": "ollama",
                "config": {
                    "model": "nomic-embed-text",
                    "ollama_base_url": OLLAMA_BASE_URL,
                },
            },
            "vector_store": {
                "provider": "chroma",
                "config": {
                    "collection_name": "episodic_memory",
                    "path": str(MEMORY_DB_PATH.parent / "mem0_chroma"),
                },
            },
            "history_db_path": str(MEMORY_DB_PATH),
        }
        _mem0_client = Memory.from_config(config)
        logger.info("Mem0 client initialised")
    except Exception:
        logger.exception("Failed to initialise Mem0 — episodic memory disabled")
        _mem0_client = None

    return _mem0_client


def extract_and_store(messages: list[dict[str, str]], session_id: str) -> None:
    """
    Extract facts from a list of {role, content} messages and store them.
    Call at end of session or periodically during long conversations.
    """
    client = _get_client()
    if client is None:
        return
    try:
        client.add(messages, user_id=_DEFAULT_USER, metadata={"session_id": session_id})
        logger.info("Episodic memories extracted for session %s", session_id)
    except Exception:
        logger.exception("Memory extraction failed for session %s", session_id)


def retrieve_relevant(query: str, limit: int = EPISODIC_INJECT_COUNT) -> list[str]:
    """
    Return up to `limit` past memory strings relevant to the query.
    Injected into the system prompt at session start.
    """
    client = _get_client()
    if client is None:
        return []
    try:
        results = client.search(query, user_id=_DEFAULT_USER, limit=limit)
        memories = results.get("results", results) if isinstance(results, dict) else results
        return [m.get("memory", str(m)) for m in memories if m]
    except Exception:
        logger.exception("Memory retrieval failed")
        return []


def get_all_memories() -> list[dict[str, Any]]:
    """Return all stored memories (for UI memory browser)."""
    client = _get_client()
    if client is None:
        return []
    try:
        results = client.get_all(user_id=_DEFAULT_USER)
        memories = results.get("results", results) if isinstance(results, dict) else results
        return list(memories)
    except Exception:
        logger.exception("Failed to retrieve all memories")
        return []


def delete_memory(memory_id: str) -> None:
    client = _get_client()
    if client is None:
        return
    try:
        client.delete(memory_id=memory_id)
    except Exception:
        logger.exception("Failed to delete memory %s", memory_id)
