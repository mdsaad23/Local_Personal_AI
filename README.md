# Local AI Personal Assistant (P04A)

A **fully offline** personal knowledge base and AI assistant. Drop your documents in, ask
questions, and get answers grounded in your own files — with **zero cloud calls for
inference**. Every model runs locally on your own GPU via [Ollama](https://ollama.com).

This is Phase 1 of a long-term agentic assistant. Future phases add email, calendar, and
messaging as document sources, plus an action-execution engine.

---

## Table of Contents
1. [How to Run](#how-to-run)
2. [What This Is & Why It Exists](#what-this-is--why-it-exists)
3. [Architecture](#architecture)
4. [Models — What & Why](#models--what--why)
5. [System Requirements](#system-requirements)
6. [The Retrieval Pipeline](#the-retrieval-pipeline)
7. [Memory](#memory)
8. [Evaluation & Benchmarking](#evaluation--benchmarking)
9. [Tech Stack](#tech-stack)
10. [Configuration](#configuration)
11. [Project Structure](#project-structure)
12. [Known Limitations](#known-limitations)

---

## How to Run

### Prerequisites
- **Windows 11** (developed/tested here; the Python side is cross-platform, the helper
  scripts are PowerShell)
- **Python 3.12** — *not* 3.13/3.14 (torch, lancedb, docling have no wheels for newer
  versions yet)
- **Node.js 18+** (for the React frontend)
- **[Ollama](https://ollama.com)** installed
- A GPU with **≥16 GB VRAM** strongly recommended (see [System Requirements](#system-requirements))

### 1. Python environment
```powershell
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
python -m spacy download en_core_web_sm    # NER model for GraphRAG
```

### 2. Configuration
```powershell
copy .env.example .env
```
Then edit `.env`. The only key you may need is `DEEPSEEK_API_KEY` — and **only** for the
benchmark/evaluation suite (RAGAS judge). Daily use needs no API keys at all.
`config/settings.py` is the single source of truth; `.env` overrides its defaults.

### 3. Start Ollama (with GPU)
```powershell
.\scripts\start_ollama.ps1     # starts Ollama with the Vulkan backend + correct env vars
python scripts\verify_gpu.py   # optional — confirms GPU is active (expect >40 tok/s)
```
> Always use `start_ollama.ps1`, not plain `ollama serve`. It sets the Vulkan/AMD env vars
> and bypasses the tray app, which otherwise resets them.

### 4. Pull the models
```powershell
# Minimum to run the assistant:
ollama pull qwen3:14b-q4_K_M     # production LLM (~9 GB)
ollama pull nomic-embed-text     # embeddings (~0.3 GB)
ollama pull minicpm-v            # vision model for image ingestion (~5 GB)

# Optional — the full benchmark matrix (~45 GB):
.\scripts\pull_models.ps1
```

### 5. Run the app
The app is a **FastAPI backend + React frontend**. In development, run both:

```powershell
# Terminal 1 — backend (http://localhost:8000)
.\scripts\start_api.ps1

# Terminal 2 — frontend dev server (http://localhost:5173)
cd ui\web
npm install
npm run dev
```
Open **http://localhost:5173**.

**Production / single-server mode:** build the frontend once and let FastAPI serve it:
```powershell
cd ui\web; npm run build
# copy the build output into src/api/static/, then just run start_api.ps1
# the app is now available at http://localhost:8000
```

### Subsequent Runs (app already set up)

Once you've done the initial setup above, restarting the app is quick:

```powershell
# 1. Activate the venv
.venv\Scripts\activate

# 2. Start Ollama (with GPU — do this first, it takes a few seconds to warm up)
.\scripts\start_ollama.ps1

# 3. Start FastAPI backend in one terminal
.\scripts\start_api.ps1

# 4. Start React dev server in another terminal
cd ui\web
npm run dev
```

Open **http://localhost:5173** in your browser. The backend will automatically reload your knowledge base (LanceDB, BM25, graph) and memory from disk.

**Quick check:** Ollama should print "Listening on 127.0.0.1:11434" (or your `OLLAMA_HOST` setting). FastAPI should print "Application startup complete" with the port (default 8000). If you see either stuck or not printing, check `config/settings.py` for port conflicts.

### 6. (Optional) MCP server for Claude Desktop
Expose the knowledge base as tools inside Claude Desktop. Add to
`%APPDATA%\Claude\claude_desktop_config.json`:
```json
{
  "mcpServers": {
    "local-ai-kb": {
      "command": "python",
      "args": ["C:/path/to/Local Personal AI/src/mcp/server.py"],
      "env": { "PYTHONPATH": "C:/path/to/Local Personal AI" }
    }
  }
}
```
This adds three tools: `query_kb`, `list_docs`, and `add_doc`.

---

## What This Is & Why It Exists

**The problem.** Personal and professional knowledge lives scattered across PDFs, contracts,
reports, and notes. Cloud AI assistants can read them — but that means uploading private,
often confidential, documents to a third party. For sensitive material (procurement
contracts, financial reports, personal records), that's a non-starter.

**The objective.** Build an assistant that is as capable as a cloud RAG system but runs
**entirely on local hardware** — no document, embedding, or query ever leaves the machine.
The only network call in the entire system is an optional one to a judge model used purely
to *score* benchmark quality; it is never part of a user-facing answer.

**What it does today (Phase 1 scope):**
1. **Ingests** PDF, DOCX, Markdown, TXT, and images (PNG/JPG/WebP/GIF).
2. **Retrieves** with a 4-layer hybrid pipeline: dense vectors + BM25 + knowledge-graph
   traversal, fused and cross-encoder reranked.
3. **Remembers** across sessions via episodic memory.
4. **Answers** by streaming tokens from a local 14B LLM, grounded in your documents and
   showing its sources.
5. **Benchmarks** 9 model variants for speed (TTFT/TGS), VRAM, and answer quality.
6. **Integrates** with Claude Desktop via an MCP server.

**Designed for extensibility.** The knowledge graph already includes
Person / Organization / Event / Task node types, the ingestion pipeline is source-agnostic,
and MCP tools are additive — so future phases (email/calendar/messaging ingestion, then an
action engine) bolt on without re-architecture.

---

## Architecture

```
                       ┌──────────────────────────────────────────────┐
                       │  React + TypeScript SPA (ui/web, port 5173)    │
                       │  chat · documents · graph · benchmarks         │
                       └───────────────────────┬──────────────────────┘
                                                │  REST + SSE (/api/*)
                       ┌────────────────────────▼──────────────────────┐
                       │  FastAPI backend (src/api, port 8000)          │
                       │  routes: chat · documents · system · benchmark │
                       └───┬───────────────┬─────────────────┬─────────┘
                           │               │                 │
              ┌────────────▼───┐   ┌───────▼────────┐  ┌─────▼──────────┐
              │  INGESTION     │   │   RETRIEVAL     │  │   MEMORY        │
              │  Docling parse │   │  Dense (LanceDB)│  │  Mem0 + SQLite  │
              │  semantic chunk│   │  Sparse (BM25)  │  │  episodic facts │
              │  embed         │   │  Graph (NetworkX)│ └─────────────────┘
              │  graph build   │   │  RRF fusion     │
              └────────────────┘   │  cross-encoder  │
                                   │  rerank         │
                                   └───────┬─────────┘
                                           │
                                   ┌───────▼─────────┐
                                   │  GENERATION      │
                                   │  query router    │
                                   │  Ollama (Vulkan) │
                                   └──────────────────┘
                                           │
                                   ┌───────▼─────────┐
                                   │  Ollama server   │  ← all model inference, local GPU
                                   └──────────────────┘
```

- **Ingestion pipeline is async**; the **query pipeline is sync** (run in a thread pool so
  the FastAPI event loop stays free). Document uploads return immediately and ingest in the
  background, with a pollable status endpoint.
- **Streaming everywhere**: the LLM streams tokens over Server-Sent Events; TTFT is measured
  at the first chunk.

---

## Models — What & Why

All models run **locally through Ollama**. Each was chosen under a hard 16 GB VRAM budget.

| Role | Model | Size | Why this one |
|---|---|---|---|
| **Production LLM** | **Qwen 3 14B Q4_K_M** | ~9 GB | Best instruction-following + reasoning in the 14B tier. Hybrid *thinking / non-thinking* modes let the router enable chain-of-thought only for hard multi-hop queries. Strong Arabic/English (relevant for UAE procurement docs). Q4_K_M fits in 9 GB, leaving headroom for the KV cache. |
| **Vision** | **minicpm-v** | ~5 GB | Describes/transcribes images at ingestion so they become searchable text. Auto-selected for chat turns that include an image. |
| **Embedding** | **nomic-embed-text** | ~0.3 GB | 768-dim, runs on **CPU** via Ollama — so it never competes with the 14B model for VRAM. Stable, well-integrated with LanceDB. |
| **Reranker** | **ms-marco-MiniLM-L6-v2** (cross-encoder) | ~0.1 GB | Re-scores `(query, chunk)` pairs directly. Adds ~200 ms but eliminates ~40% of false positives and, crucially, doesn't depend on embedding quality — it compensates for the embedder's recall ceiling. Runs on CPU/torch. |
| **Eval judge** *(benchmark only)* | **DeepSeek-V3** API | cloud | Used **only** to grade benchmark answers. Using the model under test as its own judge is methodologically invalid; DeepSeek-V3 is an independent, frontier-class, cheap judge (~$0.03 per full run). This is the *only* non-local component, and it never touches a user-facing answer. |

### Why local-only inference?
Privacy is the entire point. 16 GB VRAM makes local 14B inference viable, so there's no
quality reason to reach for the cloud for everyday Q&A. Trade-off: you can't use frontier
cloud models for inference, and performance is tied to your GPU.

### Why Q4_K_M quantization?
Best quality-per-byte for GGUF. `_K_M` = K-means quantization at medium group size, which
protects the most important weights — roughly **1–2% perplexity loss vs FP16** while fitting
a 14B model in ~9 GB. TGS is bandwidth-bound, so smaller weights can even be *faster*.

| Level | Bytes/weight | Perplexity loss | VRAM (14B) |
|---|---|---|---|
| FP16 | 2.0 | 0% | ~28 GB |
| Q8_0 | 1.0 | ~0.1% | ~14 GB |
| Q5_K_M | 0.625 | ~0.5% | ~11 GB |
| **Q4_K_M** | **0.5** | **~1–2%** | **~9 GB** |
| Q3_K_M | 0.375 | ~5–8% | ~7 GB |

### Benchmark matrix (`config/models.yaml`)
The benchmark compares 9 variants spanning the size/quant/architecture space:

| Model | Params | Quant | ~VRAM | Notes |
|---|---|---|---|---|
| Llama 3.2 3B | 3B | Q4_K_M / Q8_0 | 2.5 / 3.5 GB | Speed baseline |
| Llama 3.1 8B | 8B | Q4_K_M / Q8_0 | 5.0 / 9.0 GB | Standard RAG reference |
| Phi-4 14B | 14B | Q4_K_M | 8.5 GB | Strong reasoning |
| **Qwen 3 14B** | 14B | Q4_K_M | 9.0 GB | **Production model** |
| Gemma 4 (A4B) | 26B total / 4B active | Q4_K_M | ~14 GB | **MoE** — the dense-vs-MoE datapoint |
| Mistral 7B v0.3 | 7B | Q4_K_M | 4.5 GB | Sliding-window attention profile |

> *Llama 4 Scout was evaluated and removed: 67 GB on disk (109B total weights). At 16 GB
> VRAM it forced heavy RAM offload, making throughput unusable and the comparison unfair.*

---

## System Requirements

### Reference hardware (what this was built and measured on)
- **GPU:** AMD Radeon RX 9070 XT (RDNA 4), 16 GB VRAM
- **Backend:** **Vulkan** (`OLLAMA_VULKAN=1` + `HSA_OVERRIDE_GFX_VERSION=12.0.1`)
- **OS:** Windows 11 Pro
- **Python:** 3.12

### Measured performance (Vulkan)
| Model | Throughput |
|---|---|
| Llama 3.2 3B | **221.5 tok/s** (vs 26.5 tok/s on CPU — 8× speedup) |
| Qwen 3 14B Q4_K_M | **~51 tok/s** end-to-end (including RAG retrieval) |

### Minimum / recommended
| | Minimum | Recommended |
|---|---|---|
| **VRAM** | 8 GB (run an 8B model; drop the 14B) | **16 GB** (run Qwen 3 14B + reranker) |
| **System RAM** | 16 GB | 32 GB |
| **Disk** | ~15 GB (production models only) | ~70 GB (full benchmark matrix) |

### A note on AMD GPUs
This project uses Ollama's **Vulkan** backend, not HIP/ROCm. On RDNA 4 (gfx1201), Ollama's
HIP device list didn't yet include the 9070 XT, and ROCm tooling (`rocm-smi`) doesn't ship
on Windows. Vulkan is the confirmed-working path and delivers ~70–85% of HIP performance
with no proprietary SDK. NVIDIA users can run the same stack on CUDA with no code changes —
Ollama abstracts the backend. The KV cache is set to `q4_0` and context to 16K by default to
keep a 14B model comfortably within 16 GB; bump these in `.env` if you have more VRAM.

---

## The Retrieval Pipeline

No single retrieval method covers every query type, so four run in parallel and are fused:

1. **Dense (LanceDB ANN)** — semantic similarity via `nomic-embed-text` vectors. Catches
   paraphrases and conceptual matches.
2. **Sparse (BM25)** — keyword/lexical match. Catches exact IDs, names, and rare terms that
   dense search misses.
3. **GraphRAG (NetworkX)** — entities (spaCy NER) and LLM-extracted relationships form a
   knowledge graph. Answers relational/multi-hop questions ("who is connected to X?") that
   neither dense nor sparse can.
4. **Fusion + Rerank** — results are merged with **Reciprocal Rank Fusion** (RRF, k=60, no
   score normalization needed), then a **cross-encoder reranker** re-scores the top
   candidates and keeps the best 5.

**Adaptive query router** (`src/generation/router.py`) classifies each query first
(heuristic, <5 ms; LLM fallback ~300 ms) and picks a path:
- `DIRECT` — greetings/math/coding → no retrieval
- `RAG` — standard document lookup
- `GRAPH` — relational/multi-hop → graph traversal
- `HYDE_RAG` — analytical queries (HyDE; currently disabled by default as it doubles TTFT)

**Chunking** (`src/ingestion/chunker.py`) is **semantic** — it splits where embedding
similarity drops (topic shifts) rather than at fixed sizes, with recursive
paragraph/sentence fallback. Target 512 tokens, 64-token overlap, sub-128-token fragments
discarded.

**Document parsing** uses **Docling** (IBM) for PDF/DOCX — it preserves heading hierarchy,
tables (as structured markdown), and multi-column layouts, rather than flattening to raw
text. Markdown/TXT are read directly; images are described by the vision model.

---

## Memory

Episodic memory via **Mem0 + SQLite**. Unlike a plain conversation buffer, Mem0 extracts
*structured facts* with embeddings, so the assistant:
- **Persists across sessions** — a new session already has context from past ones.
- **Injects by relevance** — top-K relevant memories, not a full history dump.

Working memory compresses automatically once a conversation reaches ~80% of the context
window.

---

## Evaluation & Benchmarking

The benchmark harness measures every model in the matrix on three axes:
- **Speed** — TTFT (time to first token) and TGS (tokens/sec), all timed with
  `time.perf_counter()` and warm-up runs discarded.
- **Resource** — VRAM usage.
- **Quality** — graded by an independent judge across multiple frameworks:
  - **RAGAS** — faithfulness & answer relevance
  - **DeepEval** — hallucination, contextual precision/recall
  - **NIAH** (Needle in a Haystack) — long-context retrieval
  - **BFCL** — tool/function-calling accuracy
  - **EvalPlus / HumanEval+** — coding ability

```powershell
python src/evaluation/benchmark.py --output data/benchmarks/
```
Results (CSVs + plots) land in `data/benchmarks/` and render in the UI's benchmark panel.

---

## Tech Stack

| Layer | Tool |
|---|---|
| Model server | Ollama (Vulkan / AMD) |
| Production LLM | Qwen 3 14B Q4_K_M |
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
| Frontend | React 19 + TypeScript + Vite + Tailwind |
| MCP server | mcp Python SDK |
| Eval judge | DeepSeek-V3 API (evaluation only) |
| Eval frameworks | RAGAS, DeepEval, NIAH, BFCL, EvalPlus |

> The repo still contains a legacy Streamlit UI (`ui/app.py`, `ui/pages/`). It has been
> **superseded by the React app** (`ui/web/`) — Streamlit's full-rerun model couldn't do
> token streaming or non-blocking uploads. See `docs/decisions.md` (D14) for the rationale.

---

## Configuration

Everything is centralized in **`config/settings.py`** (no hardcoded paths or model names
elsewhere). Key knobs, all overridable via `.env`:

| Setting | Default | Meaning |
|---|---|---|
| `PRODUCTION_MODEL` | `qwen3:14b-q4_K_M` | Main chat model |
| `CONTEXT_LENGTH` | `16384` | Context window (tuned for 16 GB VRAM) |
| `KV_CACHE_TYPE` | `q4_0` | KV cache quantization |
| `CHUNK_SIZE` / `OVERLAP` / `MIN` | 512 / 64 / 128 | Chunking (tokens) |
| `TOP_K_DENSE` / `SPARSE` / `RERANK` | 10 / 10 / 5 | Retrieval candidate counts |
| `RRF_K` | 60 | Fusion constant |
| `EPISODIC_INJECT_COUNT` | 5 | Memories injected per session |
| `OLLAMA_MODELS` | `D:/ollama_models` | Where Ollama stores model weights |

> Models are stored on `D:\ollama_models` to avoid filling the C: drive (the benchmark
> matrix is ~45–70 GB). Change `OLLAMA_MODELS` to match your machine.

---

## Project Structure

```
├── config/
│   ├── settings.py          # Single source of truth — paths, models, tunables
│   └── models.yaml          # Benchmark model matrix
├── data/
│   ├── documents/           # Drop zone (watchdog monitors)
│   ├── processed/           # Ingestion metadata (SHA256 hashes)
│   └── db/                  # lancedb/ · bm25/ · graph/ · memory.db · benchmarks/
├── src/
│   ├── ingestion/           # parsers, chunker, embedder, watcher, graph_builder
│   ├── retrieval/           # dense, sparse, graph_retrieval, reranker, fusion, hyde
│   ├── memory/              # episodic, compressor, session
│   ├── generation/          # ollama_client, router
│   ├── evaluation/          # benchmark, metrics, ragas/niah/tool/coding/deepeval
│   ├── api/                 # FastAPI backend — main.py + routes/
│   └── mcp/                 # server.py (Claude Desktop tools)
├── ui/
│   ├── web/                 # React + TS + Vite SPA (the live UI)
│   └── app.py, pages/       # legacy Streamlit (superseded — see D14)
├── scripts/
│   ├── start_ollama.ps1     # Start Ollama with Vulkan GPU (always use this)
│   ├── start_api.ps1        # Start FastAPI backend
│   ├── verify_gpu.py        # Confirm GPU is active
│   ├── pull_models.ps1      # Pull the full benchmark matrix
│   └── generate_test_corpus.py
└── docs/
    ├── decisions.md         # Full architectural decision log (D01–D17)
    └── linkedin-posts.md
```

---

## Known Limitations

- **Embedding recall ceiling.** `nomic-embed-text` (~62 MTEB) trails best-in-class
  embedders; the cross-encoder reranker compensates for precision, but some relevant chunks
  may never be retrieved. Nomic Embed v2 is the planned zero-friction upgrade.
- **GraphRAG entity dedup.** The same entity in different surface forms isn't yet merged;
  relationship extraction quality is bounded by spaCy NER + the LLM extractor.
- **Single-user / local scale.** LanceDB + NetworkX + SQLite are file-based and load into
  memory — ideal for a personal corpus, not a multi-user deployment. (Migration targets:
  Qdrant + Neo4j.)
- **Benchmarking needs internet** for the DeepSeek judge — the only break in the otherwise
  fully-offline design, and not part of any user-facing flow.
- **Local HTTP can be flaky under load** on some Windows setups (`127.0.0.1` connections
  intermittently reset) — a known environment quirk, not a server bug.

---

*Part of Mohammad Saad's AI engineering portfolio. Decision rationale for every tooling and
architectural choice lives in [`docs/decisions.md`](docs/decisions.md).*
