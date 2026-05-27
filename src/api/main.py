"""
FastAPI application entry point.

Development:  uvicorn src.api.main:app --reload --port 8000
Production:   uvicorn src.api.main:app --port 8000
              (serves the Vite build from src/api/static/)

The Vite dev server proxies /api/* to this server, so no CORS is needed in prod.
In dev mode, CORS is allowed from localhost:5173.
"""
from __future__ import annotations

import sys
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.api.routes import chat, documents, system

app = FastAPI(title="Local AI Assistant", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://localhost:4173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(chat.router,      prefix="/api")
app.include_router(documents.router, prefix="/api")
app.include_router(system.router,    prefix="/api")

# Serve the Vite production build (after `npm run build`)
_static = Path(__file__).parent / "static"
if _static.exists() and any(_static.iterdir()):
    app.mount("/assets", StaticFiles(directory=str(_static / "assets")), name="assets")

    @app.get("/{full_path:path}", include_in_schema=False)
    async def spa_fallback(full_path: str):
        return FileResponse(str(_static / "index.html"))
