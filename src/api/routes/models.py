import json
import logging
from typing import AsyncGenerator

import httpx
from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from config.settings import OLLAMA_BASE_URL
from src.api.state import state
from src.generation.ollama_client import list_local_models

router = APIRouter(tags=["models"])
logger = logging.getLogger(__name__)


class ModelsOut(BaseModel):
    models: list[str]
    current_model: str


class SwitchModelReq(BaseModel):
    model_id: str


@router.get("/models", response_model=ModelsOut)
async def get_models():
    models = list_local_models()
    return ModelsOut(models=models, current_model=state.current_model)


async def _switch_model_stream(new_model: str) -> AsyncGenerator[str, None]:
    def _event(data: dict) -> str:
        return f"data: {json.dumps(data)}\n\n"

    try:
        old_model = state.current_model
        
        # 1. Unload old model
        if old_model != new_model:
            yield _event({"status": "unloading", "model": old_model})
            async with httpx.AsyncClient(timeout=30) as client:
                try:
                    await client.post(
                        f"{OLLAMA_BASE_URL}/api/generate",
                        json={"model": old_model, "keep_alive": 0}
                    )
                except httpx.ReadTimeout:
                    pass  # Timeouts are fine, Ollama handles it in background
                except Exception as e:
                    logger.warning(f"Error unloading model {old_model}: {e}")

        # 2. Load new model
        yield _event({"status": "loading", "model": new_model})
        async with httpx.AsyncClient(timeout=120) as client:
            try:
                # Preload new model by keeping it alive
                await client.post(
                    f"{OLLAMA_BASE_URL}/api/generate",
                    json={"model": new_model, "keep_alive": -1}
                )
            except httpx.ReadTimeout:
                pass # Can timeout if model is huge, but it still loads
            except Exception as e:
                logger.error(f"Error loading model {new_model}: {e}")
                yield _event({"status": "error", "message": f"Failed to load: {e}"})
                return

        # 3. Update state and finish
        state.current_model = new_model
        yield _event({"status": "ready", "model": new_model})

    except Exception as exc:
        msg = str(exc) or f"{type(exc).__name__} (no message)"
        logger.error(f"Model switch error: {msg}")
        yield _event({"status": "error", "message": msg})


@router.post("/models/active")
async def switch_active_model(req: SwitchModelReq):
    models = list_local_models()
    if req.model_id not in models:
        raise HTTPException(status_code=400, detail="Model not found")

    return StreamingResponse(
        _switch_model_stream(req.model_id),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )
