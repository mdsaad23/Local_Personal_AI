"""
Chat routes — session management and SSE streaming.

POST /api/chats/{session_id}/stream
  - Accepts multipart/form-data: message (str) + image (file, optional)
  - Streams SSE events: token | sources | done | error
  - Auto-routes to vision model when an image is attached
"""
from __future__ import annotations

import base64
import json
import logging
from typing import AsyncGenerator

from fastapi import APIRouter, Form, HTTPException, UploadFile, File
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from config.settings import VISION_MODEL
from src.api.state import state

router = APIRouter(tags=["chat"])
logger = logging.getLogger(__name__)

# Best vision-capable model in priority order (checked against pulled models)
_VISION_PREFERENCE = ["gemma4:26b-a4b-it-q4_K_M", "llama4:scout", "minicpm-v"]


def _pick_vision_model() -> str:
    from src.generation.ollama_client import list_local_models
    pulled = list_local_models()
    for m in _VISION_PREFERENCE:
        if m in pulled:
            return m
    return VISION_MODEL


# ── Session CRUD ──────────────────────────────────────────────────────────────

class SessionOut(BaseModel):
    session_id: str
    title: str
    started_at: float
    turn_count: int
    last_message: str | None = None
    last_activity: float | None = None


class MessageOut(BaseModel):
    role: str
    content: str
    has_image: bool
    metrics: dict | None = None
    timestamp: float


@router.get("/chats", response_model=list[SessionOut])
async def list_chats():
    from src.memory.session import get_all_sessions
    return get_all_sessions()


@router.post("/chats", response_model=SessionOut)
async def create_chat():
    from src.memory.session import new_session, get_all_sessions
    sid = new_session()
    sessions = get_all_sessions()
    for s in sessions:
        if s["session_id"] == sid:
            return s
    return {"session_id": sid, "title": "New conversation",
            "started_at": 0, "turn_count": 0}


@router.delete("/chats/{session_id}", status_code=204)
async def delete_chat(session_id: str):
    from src.memory.session import delete_session
    delete_session(session_id)


@router.get("/chats/{session_id}/messages", response_model=list[MessageOut])
async def get_messages(session_id: str):
    from src.memory.session import get_messages
    return get_messages(session_id)


# ── Streaming inference ───────────────────────────────────────────────────────

async def _sse_stream(
    session_id: str,
    message: str,
    image_b64: str | None,
) -> AsyncGenerator[str, None]:
    """Yields SSE-formatted events for a single turn."""

    def _event(data: dict) -> str:
        return f"data: {json.dumps(data)}\n\n"

    try:
        import asyncio
        from src.generation.router import retrieve_for_query, build_prompt
        from src.generation.ollama_client import stream_response
        from src.memory.episodic import retrieve_relevant
        from src.memory.session import add_message, get_messages
        from src.memory.compressor import should_compress, compress

        # Choose model: vision model if image attached, else current active model
        model = _pick_vision_model() if image_b64 else state.current_model

        # Run all blocking retrieval in thread pool — keeps event loop free
        if image_b64:
            from src.generation.router import RouteType
            chunks, route = [], RouteType.DIRECT
            memories = []
        else:
            loop = asyncio.get_event_loop()
            chunks, route = await loop.run_in_executor(
                None, retrieve_for_query, message
            )
            memories = await loop.run_in_executor(
                None, retrieve_relevant, message, 3
            )

        # Emit retrieved sources before generation starts
        if chunks:
            sources = [
                {"source": c.get("source", ""), "page": c.get("page", ""),
                 "section": c.get("section", ""), "score": c.get("rerank_score")}
                for c in chunks
            ]
            yield _event({"type": "sources", "sources": sources})

        # Build context-aware prompt (CPU-only, fast)
        history = get_messages(session_id)
        history_msgs = [{"role": m["role"], "content": m["content"]} for m in history]
        prompt_msgs = build_prompt(message, chunks, memories)
        system_msg = prompt_msgs[0]
        user_turn = prompt_msgs[-1]
        full_messages = [system_msg] + history_msgs + [user_turn]

        if should_compress(full_messages):
            full_messages = compress(full_messages)

        # Stream tokens — async generator keeps event loop free
        full_response: list[str] = []
        metrics: dict = {}
        async for token, meta in stream_response(full_messages, model=model, image_b64=image_b64):
            if token:
                full_response.append(token)
                yield _event({"type": "token", "content": token})
            elif meta:
                metrics = meta

        response_text = "".join(full_response)

        # Persist both turns
        add_message(session_id, "user", message, has_image=bool(image_b64))
        saved_metrics = {
            "ttft": metrics.get("ttft_s"),
            "tgs": metrics.get("tgs"),
            "tokens": metrics.get("eval_count"),
            "route": str(route),
            "model": model,
        }
        add_message(session_id, "assistant", response_text, metrics=saved_metrics)

        yield _event({
            "type": "done",
            **saved_metrics
        })

    except Exception as exc:
        import traceback
        msg = str(exc) or f"{type(exc).__name__} (no message)"
        logger.error("Chat stream error: %s\n%s", msg, traceback.format_exc())
        yield _event({"type": "error", "message": msg})


@router.post("/chats/{session_id}/stream")
async def stream_message(
    session_id: str,
    message: str = Form(...),
    image: UploadFile | None = File(default=None),
):
    image_b64: str | None = None
    if image and image.filename:
        raw = await image.read()
        image_b64 = base64.b64encode(raw).decode()

    return StreamingResponse(
        _sse_stream(session_id, message, image_b64),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )
