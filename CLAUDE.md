# Local AI Personal Assistant — P04A

## Overview
Fully offline personal knowledge base + AI assistant. Local LLMs only (Ollama). Zero cloud for inference. Phase 1 of a long-term agentic assistant — future phases add email, calendar, WhatsApp, Telegram with action execution. Part of Mohammad Saad's 17-project AI engineering portfolio.

Owner: Mohammad Saad — AI engineering + procurement. LinkedIn posts follow: business problem → technical solution → measured result.

---

## Hardware
- GPU: AMD Radeon RX 9070 XT (RDNA 4), 16GB VRAM — **Vulkan backend** (221.5 tok/s confirmed)
- `OLLAMA_VULKAN=1` + `HSA_OVERRIDE_GFX_VERSION=12.0.1` set in user env (persistent after reboot)
- OS: Windows 11 Pro
- Python: **3.12.10 venv** — system is 3.14.3, do NOT use it (torch/lancedb/docling have no 3.14 wheels)

---

## Tech Stack
| Layer | Tool |
|---|---|
| Model server | Ollama (Vulkan/AMD) |
| Production model | Qwen 3 14B Q4_K_M |
| Vision model | minicpm-v |
| Embedding | nomic-embed-text (local, Ollama) |
| Vector store | LanceDB (embedded, file-based) |
| Sparse index | BM25 (rank_bm25) |
| Reranker | ms-marco-MiniLM-L6-v2 (cross-encoder) |
| Graph store | NetworkX + JSON |
| NER | spaCy (en_core_web_sm) |
| Memory | Mem0 + SQLite |
| Orchestration | LlamaIndex |
| Document parsing | Docling |
| Folder watcher | watchdog |
| Backend API | FastAPI + uvicorn (SSE streaming) |
| UI | React + TypeScript + Vite + Tailwind |
| MCP server | mcp Python SDK |
| Eval judge | DeepSeek V3 API (evaluation only) |
| Eval framework | RAGAS |

---

## Project Structure
```
├── config/
│   ├── settings.py             # Single source of truth — all paths + model names
│   └── models.yaml             # Benchmark model matrix
├── data/
│   ├── documents/              # Drop zone (watchdog monitors)
│   ├── processed/              # Ingestion metadata (SHA256 hashes)
│   └── db/
│       ├── lancedb/            # Vector store
│       ├── bm25/               # BM25 index (pickle)
│       ├── graph/              # NetworkX graph (graph.json)
│       ├── memory.db           # SQLite: Mem0 episodic memory
│       └── benchmarks/         # Benchmark CSVs + plots
├── src/
│   ├── ingestion/              # parsers, chunker, embedder, watcher, graph_builder
│   ├── retrieval/              # dense, sparse, graph_retrieval, reranker, fusion, hyde
│   ├── memory/                 # episodic, compressor, session
│   ├── generation/             # ollama_client, router
│   ├── evaluation/             # benchmark, metrics, ragas_eval, niah/tool/coding/deepeval
│   ├── api/                    # FastAPI backend — routes: chat, documents, system, benchmark
│   │   ├── main.py             # app entry; serves Vite build from api/static in prod
│   │   └── routes/             # chat.py (SSE), documents.py, system.py, benchmark.py
│   └── mcp/                    # server.py (Claude Desktop tools)
├── ui/
│   ├── web/                    # React + TypeScript + Vite SPA (the live UI)
│   │   └── src/components/     # ChatWindow, DocumentPanel, BenchmarkPanel, Sidebar
│   └── (legacy Streamlit: app.py + pages/ — superseded by ui/web, see D14)
├── scripts/
│   ├── start_ollama.ps1        # Always use this — starts Ollama with Vulkan GPU
│   ├── start_api.ps1           # Start FastAPI backend (uvicorn, port 8000)
│   ├── verify_gpu.py           # Run first to confirm GPU
│   ├── pull_models.ps1         # Pull all benchmark models (~45 GB)
│   └── generate_test_corpus.py
└── docs/
    ├── decisions.md            # Full decision log — update after every phase
    └── linkedin-posts.md       # All generated posts (append only)
```

---

## Key Commands
```powershell
# Activate venv
.venv\Scripts\activate

# Start Ollama with GPU (always use this script, not plain 'ollama serve')
.\scripts\start_ollama.ps1

# Verify GPU (expect >40 tok/s)
$env:PYTHONIOENCODING = "utf-8"; python scripts/verify_gpu.py

# Pull benchmark models
.\scripts\pull_models.ps1

# Run the app — backend + frontend (two terminals)
.\scripts\start_api.ps1                       # FastAPI on http://localhost:8000
cd ui/web; npm run dev                         # Vite dev server on http://localhost:5173
# Production: cd ui/web; npm run build  → copy build into src/api/static, then start_api only

# Run full benchmark
python src/evaluation/benchmark.py --output data/benchmarks/

# Start folder watcher
python src/ingestion/watcher.py

# Start MCP server
python src/mcp/server.py
```

---

## Coding Conventions
- All config via `config/settings.py` — no hardcoded paths or model names elsewhere
- Timing: `time.perf_counter()` only — never `time.time()`
- Streaming: always stream from Ollama, measure TTFT at first chunk
- Document IDs: SHA256 hash of file content — deduplication key
- DB operations: typed try/except — no bare `except`
- Ingestion pipeline: async. Query pipeline: sync. Do not mix.
- Type hints on all function signatures
- No `print()` in production code — use `loguru` logger

---

## Decision Log Protocol
**`docs/decisions.md` is the live record of every architectural and tooling decision.**

Update it when:
- A new technology is chosen over alternatives
- A phase is completed and results differ from expectations
- A known limitation is discovered or worked around
- A design choice is revisited or reversed

Entry format (one block per decision):
```
### D## — Short title
**Decision:** what was chosen vs rejected
**Why:** reasoning at decision time
**Pros:** what this enables
**Cons/Limitations:** what it costs
**Impact:** how it shapes the build
**What's lacking:** gap vs ideal
**Ideal state:** what better looks like without current constraints
**Updated:** date — reason for update (add if revisiting an existing entry)
```

---

## Current Scope (P04A)
1. Benchmark harness: 9 model variants × TTFT + TGS + VRAM + RAGAS
2. Document ingestion: PDF, DOCX, MD, TXT, images
3. Hybrid retrieval: dense + BM25 + GraphRAG + cross-encoder reranker
4. Episodic memory: Mem0 + SQLite, cross-session persistence
5. Web UI (React + FastAPI): chat + document manager + graph view + benchmark viewer
6. MCP server: Claude Desktop integration

## Future Development (out of scope — design for extensibility)
- Phase 2: Gmail, Outlook, Google Calendar, Notion, WhatsApp, Telegram as document sources
- Phase 3: Action engine — identify action items, execute or remind, human-in-the-loop confirmation
- Architecture implication: graph schema includes Person/Organization/Event/Task nodes from day one; ingestion pipeline is source-agnostic; MCP tools are additive

---

## LinkedIn Post Protocol
After every completed phase, invoke the `linkedin-post-writer` agent.

```
Phase completed: [name]
Key metrics: [at least one number — required]
Notes: [raw observations, surprises, failures]
```

### Phase checklist
- [x] GPU verification + Vulkan fix (221.5 tok/s) → **POST DUE**
- [ ] Embedding model live and tested
- [ ] Document ingestion pipeline complete
- [ ] Semantic chunker benchmarked vs fixed-size
- [ ] LanceDB + BM25 operational
- [ ] Hybrid retrieval + reranker live
- [ ] GraphRAG entity extraction working
- [ ] Episodic memory persisting across sessions
- [ ] Streamlit UI running
- [ ] MCP server connected to Claude Desktop
- [ ] Full benchmark matrix complete
- [ ] Quantisation comparison table published
- [ ] Full offline test passed

### Bridge Rule (non-negotiable)
Every post opens with a business problem a CPO understands. Technical solution follows. A specific number closes. Never purely technical. Never purely strategic.

### Daily format
| Day | Format |
|---|---|
| Monday | Hot Take |
| Tuesday | Technical Build |
| Wednesday | Framework |
| Thursday | Strategic |
| Friday | Reflection |
| Saturday | Data Story |
| Sunday | Industry Insight |
