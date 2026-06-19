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

# On this machine, the local WMI service hangs indefinitely on Win32_OperatingSystem
# queries. Python 3.12's platform.uname()/win32_ver() calls into WMI via the _wmi
# module, and pandas (imported transitively by lancedb -> pyarrow) calls
# platform.machine() at import time, which deadlocks the whole startup. Forcing
# _wmi_query to fail makes platform.py fall back to its non-WMI implementation.
import platform
platform._wmi_query = lambda *a, **k: (_ for _ in ()).throw(OSError("WMI disabled: hangs on this host"))

# Verify that the server is running inside the correct virtual environment and has critical RAG libraries
try:
    import docling
    import lancedb
    import spacy
    import sentence_transformers
except ImportError as e:
    print("\n" + "="*80, file=sys.stderr)
    print(f"CRITICAL STARTUP ERROR: Missing required library: {e}", file=sys.stderr)
    print("This usually happens when running the server outside the virtual environment (.venv).", file=sys.stderr)
    print("Please activate the virtual environment and run the server again:", file=sys.stderr)
    print("  Windows PowerShell:  .venv\\Scripts\\Activate.ps1", file=sys.stderr)
    print("  Windows CMD:         .venv\\Scripts\\activate.bat", file=sys.stderr)
    print("="*80 + "\n", file=sys.stderr)
    sys.exit(1)

from src.api.routes import chat, documents, system, benchmark, models

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
app.include_router(benchmark.router, prefix="/api")
app.include_router(models.router,    prefix="/api")



# Serve the Vite production build (after `npm run build`)
_static = Path(__file__).parent / "static"
if _static.exists() and any(_static.iterdir()):
    app.mount("/assets", StaticFiles(directory=str(_static / "assets")), name="assets")

    @app.get("/{full_path:path}", include_in_schema=False)
    async def spa_fallback(full_path: str):
        return FileResponse(str(_static / "index.html"))
