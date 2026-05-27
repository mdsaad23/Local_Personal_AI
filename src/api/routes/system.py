from __future__ import annotations

from fastapi import APIRouter
from pydantic import BaseModel

from config.settings import PRODUCTION_MODEL, EMBED_MODEL, VISION_MODEL

router = APIRouter(tags=["system"])


class HealthResponse(BaseModel):
    ollama: bool
    models: list[str]
    production_model: str
    vision_model: str
    embed_model: str


@router.get("/health", response_model=HealthResponse)
async def health():
    from src.generation.ollama_client import check_ollama_health, list_local_models
    ok = check_ollama_health()
    models = list_local_models() if ok else []
    return HealthResponse(
        ollama=ok,
        models=models,
        production_model=PRODUCTION_MODEL,
        vision_model=VISION_MODEL,
        embed_model=EMBED_MODEL,
    )
