# Decision Log — P04A Local AI Personal Assistant

Live record of every architectural and tooling decision. Updated after every phase or whenever a significant decision is made or revisited. See CLAUDE.md for the entry format.

---

### D01 — GPU Backend: Vulkan (not HIP, not CUDA)
**Decision:** Ollama uses Vulkan for AMD GPU on Windows. HIP SDK 7.1 installed but gfx1201 (RDNA 4 / 9070 XT) not in Ollama's HIP device list. Vulkan is the active path via `OLLAMA_VULKAN=1`.
**Why:** Only confirmed-working GPU path for this hardware. HIP was attempted first — installed, restarted, tested — still 100% CPU. Root cause: Ollama tray app restarting server without the env var; and gfx1201 HIP support absent. Vulkan works once tray app is bypassed.
**Pros:** 221.5 tok/s on llama3.2 3B (vs 26.5 tok/s CPU — 8× speedup). Cross-vendor, no proprietary SDK required.
**Cons/Limitations:** ~70–85% of HIP perf. No clean Python VRAM API — must subprocess `hipInfo.exe`. `rocm-smi` (standard tool) doesn't ship on Windows. Tray app lifecycle requires `start_ollama.ps1` until next reboot.
**Impact:** All benchmark numbers are Vulkan numbers — must be labelled as such. Warm TTFT is ~0.3s; first-inference TTFT includes model load (~1–2s). Cannot use `pynvml`/`gpustat` for VRAM monitoring.
**What's lacking:** Programmatic VRAM readout from Python. Numbers not directly comparable to CUDA benchmarks.
**Ideal state:** Native HIP gfx1201 support in Ollama (expected within 2–3 release cycles). Or: llama.cpp compiled from source with explicit gfx1201 Vulkan target.

---

### D02 — Model Server: Ollama (not llama.cpp direct, not LM Studio)
**Decision:** Ollama as model server. llama.cpp documented as fallback. LM Studio not used.
**Why:** Stable OpenAI-compatible HTTP API, multi-model management, Python SDK, and native streaming. Switching models during benchmarking is a one-line config change.
**Pros:** Single endpoint for all models. Streaming built-in. `eval_count`/`eval_duration` fields give accurate TGS without external timing.
**Cons/Limitations:** AMD GPU support lags NVIDIA and Linux. Tray app conflicts with env var injection. No native VRAM monitoring API. Flash Attention not auto-enabled for AMD in 0.24.0.
**Impact:** All `ollama_client.py` mirrors OpenAI interface — migration to any OpenAI-compatible server is minimal. Benchmarking uses Ollama's internal counters for TGS (accurate).
**What's lacking:** Cannot tune thread count, batch size, or rope scaling without Modelfile customisation. Flash Attention disabled.
**Ideal state:** Ollama with full RDNA 4 HIP support and Flash Attention enabled by default. Fallback: llama-server (llama.cpp) for more inference parameter control.

---

### D03 — Production Model: Qwen 3 14B Q4_K_M
**Decision:** Qwen 3 14B Q4_K_M as default. Alternatives in benchmark matrix: Llama 3.2 3B, Llama 3.1 8B, Phi-4 14B, Gemma 4 27B (MoE), Llama 4 Scout (MoE). Updated from Qwen 2.5 14B after May 2026 research — Qwen 3 is the current generation with hybrid thinking/non-thinking modes and stronger reasoning.
**Why:** Qwen 3 14B adds a switchable chain-of-thought mode on top of the Qwen 2.5 14B instruction-following advantage. Q4_K_M at ~9GB leaves 7GB for KV cache. MoE models (Gemma 4, Llama 4 Scout) are now in the benchmark matrix to test the MoE vs dense tradeoff at comparable VRAM spend.
**Pros:** Best instruction-following + reasoning at 14B tier. Thinking mode enables structured planning for complex multi-hop queries. Strong multilingual (Arabic/English). Fits 32K context with Q8_0 KV cache.
**Cons/Limitations:** ~35–45 tok/s vs ~70–100 tok/s for 7B models. nomic-embed-text runs on CPU (no VRAM headroom for two GPU models). MoE alternatives may outperform dense 14B on quality at similar VRAM — benchmark will reveal this.
**Impact:** Benchmark now spans dense models (3B–14B), and MoE models (Gemma 4 27B at 4B active, Llama 4 Scout at 17B active) — this gives a genuine MoE vs dense comparison which is the actual LinkedIn insight.
**What's lacking:** No speculative decoding in Ollama (would improve TGS 30–50%). Thinking mode not automatically triggered — query router should activate it for complex queries.
**Ideal state:** Query router with three paths: simple factual → Llama 3.2 3B, standard RAG → Qwen 3 14B no-thinking, multi-hop reasoning → Qwen 3 14B with thinking. Planned in `src/generation/router.py`.
**Updated:** 2026-06-13 — **Llama 4 Scout removed from the benchmark matrix.** Measured 67 GB on disk (109B total weights); at 16 GB VRAM it forced heavy RAM offload, making TGS unusable and the comparison apples-to-oranges. Remaining matrix is dense 3B–14B plus the Gemma 4 MoE (26B total / 4B active) for the MoE-vs-dense datapoint.

---

### D04 — Quantisation: Q4_K_M Default, Q5_K_M and Q8_0 for Comparison
**Decision:** Q4_K_M as production quantisation. Benchmark matrix includes Q5_K_M and Q8_0 variants.
**Why:** Q4_K_M delivers best quality-per-byte for GGUF. `_K_M` = K-means quantisation at medium group size — protects important weights. ~1–2% perplexity loss vs F16. Q8_0 doubles VRAM.

| Level | Bytes/weight | Perplexity loss | VRAM (14B) |
|---|---|---|---|
| F16 | 2.0 | 0% | ~28GB |
| Q8_0 | 1.0 | ~0.1% | ~14GB |
| Q5_K_M | 0.625 | ~0.5% | ~11GB |
| Q4_K_M | 0.5 | ~1–2% | ~9GB |
| Q3_K_M | 0.375 | ~5–8% | ~7GB |

**Pros:** Q4_K_M fits 14B in 9GB with 32K context headroom. TGS is bandwidth-bound — Q4 can be faster than Q8 (smaller weight transfers outweigh extra dequantisation).
**Cons/Limitations:** Measurable quality loss on complex reasoning. IQ quantisations (IQ4_XS) outperform Q4_K_M at same bit count but not yet in benchmark matrix.
**Impact:** Benchmark must measure RAGAS faithfulness per quant level — makes quality difference empirically visible, not just theoretical. Q5_K_M is the key "is the 2GB VRAM premium worth it?" datapoint.
**What's lacking:** IQ4_XS variants missing from benchmark matrix. No direct perplexity measurement — RAGAS faithfulness is a proxy.
**Ideal state:** Add IQ4_XS alongside K-means variants. Add perplexity measurement on a fixed test corpus for model-quality ground truth independent of retrieval.

---

### D05 — Local vs Cloud Hosting
**Decision:** Fully local inference. DeepSeek API for evaluation judge only.
**Why:** Privacy-preserving AI for personal documents is the explicit design goal. 16GB VRAM makes local 14B inference viable. No API costs for inference.
**Pros:** Zero data leaves the machine. No rate limits. Works fully offline. Deterministic latency (no network dependency).
**Cons/Limitations:** Hardware dependency — performance tied to local GPU. Cannot access frontier models (GPT-4o, Claude 3.7) for inference. Model updates are manual pulls.
**Impact:** All design decisions constrained by VRAM budget. Embedding must also be local (nomic-embed-text). Benchmarking breaks the offline constraint (DeepSeek judge) — acceptable since benchmarking is not a user-facing workflow.
**What's lacking:** No hybrid mode where user can opt-in to cloud routing for non-sensitive queries.
**Ideal state:** Privacy-aware router: sensitive documents always local; user can opt-in to cloud for specific non-sensitive queries. Architecturally possible via a `cloud_allowed` flag in `src/generation/router.py`.

---

### D06 — Vector Store: LanceDB (not ChromaDB, not Qdrant)
**Decision:** LanceDB embedded mode. ChromaDB and Qdrant considered.
**Why:** File-based, zero server process. Lance columnar format has better ANN performance than ChromaDB's HNSW on large corpora. Native multi-modal data types. DiskANN handles datasets larger than RAM.
**Pros:** Zero infrastructure. Portable (copy the folder). Python-first API. Active development.
**Cons/Limitations:** No distributed mode. Metadata filtering during ANN search is less mature than Qdrant. No native hybrid search — BM25 is separate, fused via RRF. API changes frequently (pin version).
**Impact:** Hybrid search requires two separate indices (LanceDB + rank_bm25). Metadata filtering is post-retrieval. Vector store is fully portable.
**What's lacking:** At-search-time metadata filtering. Native hybrid search.
**Ideal state:** Qdrant for a team/production system (payload filtering, native hybrid). LanceDB is correct for single-user local. Migrate at ~1M chunks or when multi-user is needed.

---

### D07 — Retrieval: Hybrid 4-Layer (Dense + BM25 + GraphRAG + Reranker)
**Decision:** LanceDB ANN + BM25 sparse + NetworkX graph traversal, fused via RRF, then cross-encoder reranked. Alternatives: pure vector search, BM25 only.
**Why:** No single retrieval method covers all query types. Dense misses exact IDs. BM25 misses semantic similarity. Graph answers relational queries neither can handle. Cross-encoder corrects false positives from all three.
**Pros:** Full query space coverage. RRF requires no score normalisation. Reranker adds ~200ms, eliminates ~40% false positives. Each layer independently testable.
**Cons/Limitations:** ~400–800ms total retrieval latency vs ~50ms for pure dense. BM25 index is in-memory (RAM pressure at ~500K chunks). spaCy NER misses domain-specific entities. No learned fusion weights.
**Impact:** TTFT includes retrieval — users perceive 400–800ms before generation starts. This pipeline is the primary technical demonstration in the project.
**What's lacking:** Learned alpha weights for RRF. Domain-specific NER. Query decomposition for multi-hop questions.
**Ideal state:** Calibrated fusion weights. Fine-tuned NER on procurement/finance entities. Query decomposition before retrieval.

---

### D08 — Chunking: Semantic + Parent-Child Pattern
**Decision:** Semantic chunking (split where embedding similarity drops), recursive character fallback. 512 token target, 64 token overlap. Parent-child chunk storage implemented: small child chunks (~150 tokens) indexed for retrieval precision, parent sections (~512 tokens) returned to the LLM for generation context.
**Why:** Fixed-size chunks split mid-concept. Semantic chunking finds natural content breakpoints. Parent-child pattern (2026 best practice) solves the precision vs context tradeoff: small chunks retrieve precisely, parent chunks give the LLM enough context to answer accurately without hallucinating. Both stored; lookup parent via chunk_index lineage.
**Pros:** Child chunks: higher retrieval precision. Parent returned to LLM: full surrounding context. Metadata preserved at chunk level (source, page, heading). Recursive fallback prevents oversized chunks.
**Cons/Limitations:** Slower ingestion — stores 2× the chunks. Variable chunk sizes harder to predict for context budgeting. Breakpoints are embedding-model-driven, not structure-driven.
**Impact:** LLM receives richer context without bloating the embedding index with long vectors. Context assembly uses parent, not child — this is the key quality gain.
**What's lacking:** No proposition-level chunking (atomic facts). Docling's structural hierarchy not fully exploited as chunking signal. Parent-child lineage stored in chunk metadata but not yet surfaced in UI.
**Ideal state:** Structure-aware chunking using Docling's heading hierarchy as primary split signal, semantic chunking within sections. Proposition chunking for high-value documents (contracts, reports).

---

### D09 — GraphRAG: NetworkX + JSON (not Neo4j, not MS GraphRAG)
**Decision:** In-memory NetworkX graph persisted to JSON. MS GraphRAG and Neo4j not used.
**Why:** MS GraphRAG requires Azure OpenAI and targets enterprise corpora. Neo4j requires a server. NetworkX + JSON has zero infrastructure overhead for <100K nodes.
**Pros:** Zero infrastructure. Full graph algorithm library (PageRank, shortest path). Portable JSON file. pyvis renders interactive graph for Streamlit UI.
**Cons/Limitations:** Entire graph loads at startup — RAM pressure at ~1M nodes. No native vector search on nodes. JSON write is not transactional (crash mid-write corrupts file). NER quality limits entity extraction.
**Impact:** Graph queries fast for small corpora (ms-range traversal). Graph visualisation is strong LinkedIn asset. Graph schema (Person/Org/Event/Task) supports Phase 2 email/calendar integration from day one.
**What's lacking:** Graph write atomicity. No node embedding for entity similarity queries. Relationship extraction is LLM-assisted (expensive) — lighter dependency parser would be faster.
**Ideal state:** Write-then-rename for crash safety. Node embedding layer for entity similarity. Neo4j for multi-user or large-corpus deployment.

---

### D10 — Memory: Mem0 + SQLite (not LangChain Memory, not Redis)
**Decision:** Mem0 for episodic memory extraction, SQLite for persistence. LangChain ConversationBufferMemory and Redis not used.
**Why:** LangChain buffer simply appends turns — no structured fact extraction, no cross-session persistence. Redis requires a server. Mem0 extracts structured facts with embeddings for relevance-based retrieval.
**Pros:** Cross-session persistence. Relevance-based injection (top-K relevant memories, not full history dump). Zero infrastructure. Entity-level memories supported.
**Cons/Limitations:** Memory extraction adds ~500ms per turn. SQLite is not concurrent. Memory retrieval adds to TTFT. No memory decay or conflict resolution.
**Impact:** System feels personalised over time — remembers recurring topics without being told. First response in a new session already has context from past sessions.
**What's lacking:** Memory decay. Conflict resolution when new fact contradicts stored one. User-facing memory browser in UI.
**Ideal state:** Time-weighted retrieval (recent memories ranked higher unless older are highly relevant). Conflict detection and resolution. UI panel for viewing/editing/deleting memories.

---

### D11 — Document Parser: Docling (not PyMuPDF, not Unstructured)
**Decision:** Docling (IBM, 2024) as primary parser. PyMuPDF as fallback for simple cases.
**Why:** Docling understands document structure — heading hierarchy, tables as structured data, multi-column layouts, embedded images. PyMuPDF extracts raw text without structure. Unstructured.io has a cloud tier.
**Pros:** Preserves structure (headings as chunk metadata, tables as markdown). Handles images-in-PDF. Handles multi-column layouts. Fully local.
**Cons/Limitations:** Slower than PyMuPDF (~2–10s/page vs ~0.1s). Memory-intensive for large PDFs. Arabic/RTL quality inconsistent. Docling 2.x API changed significantly from 1.x.
**Impact:** Ingestion is background — slowness acceptable. Tables in financial reports are retrievable as tables, not mangled text. This is a meaningful quality advantage worth documenting.
**What's lacking:** Arabic OCR quality inconsistent (important for UAE procurement documents). No streaming output.
**Ideal state:** Docling for English structured documents. Dedicated Arabic OCR for Arabic-heavy documents. Streaming Docling output for large files (start embedding before full parse completes).

---

### D12 — Eval Judge: DeepSeek V3 API (not Anthropic, not local model)
**Decision:** DeepSeek V3 via `api.deepseek.com` as RAGAS judge. Anthropic API unavailable. Local-model-as-judge rejected.
**Why:** Methodologically invalid to use the same model being benchmarked as its own judge. DeepSeek V3 is frontier-class, OpenAI-compatible, cheap ($0.27/M input tokens). Total benchmark run cost: ~$0.03.
**Pros:** Independent judge. Frontier capability for accurate faithfulness/relevance assessment. Zero code changes to RAGAS (OpenAI-compatible). Negligible cost.
**Cons/Limitations:** Requires internet — benchmarking cannot run fully offline. No control over judge model updates. Single judge introduces potential bias.
**Impact:** Benchmark credibility depends on judge quality. The offline constraint is broken for benchmarking only — acceptable. Cost does not limit run frequency.
**What's lacking:** No judge calibration against human ratings. Single judge — no ensemble.
**Ideal state:** Two independent judges (DeepSeek V3 + Qwen 2.5 14B local) with agreement scoring. Flag disagreements for human review.

---

### D13 — Python Version: 3.12 (not 3.14)
**Decision:** Python 3.12.10 in venv. System is 3.14.3.
**Why:** torch, lancedb, docling, sentence-transformers have no 3.14 wheels as of May 2026. 3.12 is the ML ecosystem LTS-equivalent.
**Pros:** Zero dependency failures. Reproducible on any machine with Python 3.12.
**Cons/Limitations:** Not the latest Python version.
**Impact:** None significant.
**What's lacking:** Nothing. Correct choice for today's dependency set.
**Ideal state:** Migrate to 3.14 once torch and lancedb release 3.14 wheels (estimated 6–12 months).

---

### D14 — UI: Streamlit (not Gradio, not FastAPI + React)
**Decision:** Streamlit for chat + management UI. Gradio and custom FastAPI/React not used.
**Why:** Multi-page structure matches four panels (chat, documents, graph, benchmarks). Built-in state management. pyvis graph renders natively. Ships as demo without build tooling.
**Pros:** Fast to build — focus stays on AI engineering. Multi-page with shared session state. Plotly charts native. Professional enough for LinkedIn screenshots.
**Cons/Limitations:** Prototype/demo tool — not suitable for production web deployment. Single Python thread per user session. No mobile-responsive layout. Session state clears on F5.
**Impact:** UI is a demonstration layer — correct investment level for Phase 1. Benchmark visualisation with Plotly renders natively.
**What's lacking:** Persistent chat history across page refreshes. Large graph views (>10K nodes) will be slow (pyvis renders client-side).
**Ideal state:** Streamlit for Phase 1. FastAPI + lightweight React for Phase 3 (multi-user, mobile). MCP server is the production integration path regardless of UI.
**Updated:** 2026-06-13 — REVERSED. Migrated off Streamlit to a **FastAPI backend (`src/api/`) + React/TypeScript/Vite/Tailwind SPA (`ui/web/`)**. Reason: Streamlit's full-rerun model couldn't deliver proper token-by-token chat streaming or a non-blocking document-upload UX, and session state cleared on refresh. FastAPI streams tokens over SSE (`/api/chats/{id}/stream`) and runs ingestion as a background task; React owns client state. In prod the Vite build is served from `src/api/static` by the same FastAPI process (single origin, no CORS); in dev Vite (5173) proxies `/api/*` to uvicorn (8000). The legacy Streamlit code (`ui/app.py`, `ui/pages/`) remains in-tree but is dead — `ui/web/` is the live UI. The "FastAPI + React" originally slated for Phase 3 was pulled forward to Phase 1.

---

### D15 — Embedding Model: nomic-embed-text v1 (upgrade path identified)
**Decision:** nomic-embed-text v1 via Ollama for Phase 1. Upgrade candidates researched: Qwen3-Embedding-8B (MTEB 70.58), Jina v5-text-small (MTEB v2 71.7), Nomic Embed v2 (improved multilingual).
**Why:** nomic-embed-text v1 runs on CPU (Ollama), zero setup, 768 dimensions, proven stable in this stack. Embedding runs while the 14B model holds GPU — no VRAM contention. Qwen3-Embedding-8B would require 8GB VRAM and Ollama support, conflicting with the production model.
**Pros:** Zero VRAM impact — embedding on CPU while LLM is on GPU. No model management overhead. Stable, well-documented in LanceDB integrations.
**Cons/Limitations:** nomic-embed-text v1 MTEB score (~62) is significantly below current best-in-class (Qwen3-Embedding-8B at 70.58, ~13% better). Lower embedding quality = worse retrieval recall. This is the primary retrieval quality limitation of Phase 1.
**Impact:** Retrieval quality is bounded by embedding quality before the reranker. The cross-encoder reranker partially compensates — it re-scores (query, chunk) pairs directly and doesn't depend on embedding quality. Net impact: lower recall (some relevant chunks never retrieved), but precision among retrieved chunks is high.
**What's lacking:** The embedding gap is the largest unaddressed retrieval quality issue. Nomic Embed v2 (Ollama-compatible) would be a zero-friction upgrade — same VRAM profile, better MTEB.
**Ideal state:** Nomic Embed v2 for Phase 1 (minimal change, better multilingual). Qwen3-Embedding-8B when a dedicated embedding GPU or CPU inference budget allows (~300ms/chunk acceptable for background ingestion).

---

### D16 — GraphRAG Implementation: NetworkX + LazyGraphRAG Pattern (not MS GraphRAG)
**Decision:** In-process NetworkX graph with LLM-assisted relationship extraction, following the LazyGraphRAG pattern (Microsoft, June 2025). Full MS GraphRAG (Azure-dependent, $500/corpus) not used.
**Why:** MS GraphRAG's original approach pre-computes all community summaries at indexing time — expensive ($500 for a medium corpus) and requires Azure OpenAI. LazyGraphRAG defers community detection to query time, reducing indexing cost to ~$5. We implement the same deferral principle: extract entities at ingestion, build communities lazily on first graph query.
**Pros:** GraphRAG cost drops from $500 → ~$5 equivalent (LLM extraction calls only on ingestion, not full summarisation). Enables graph retrieval on personal document corpora where upfront cost was prohibitive. NetworkX handles graph algorithms; JSON ensures portability.
**Cons/Limitations:** Relationship extraction quality is bounded by spaCy NER + LLM extraction quality. Community detection runs at query time (latency spike on first query to a new community). No native graph vector search — entity lookup is text-match before graph traversal.
**Impact:** GraphRAG is now practically deployable for a personal knowledge base — the indexing cost was the blocker. Multi-hop queries ("what did I write about X in the context of Y?") now answerable without a dedicated graph database.
**What's lacking:** Entity deduplication (same entity extracted with different surface forms). No entity similarity search (would need node embeddings). Community detection is single-pass at query time — no persistence of detected communities.
**Ideal state:** Named entity normalisation (coreference resolution). Node embeddings for entity similarity search. Persist community detection results for faster subsequent queries.

---

### D17 — Model Storage: D:\ollama_models (not default C:\Users\..\.ollama\models)
**Decision:** Ollama model directory moved to `D:\ollama_models` via `OLLAMA_MODELS` env var. Default C: path kept for project data and code only.
**Why:** C: drive filled to 0 GB free during benchmark model pulls (~70 GB of models). D: has 294 GB free. The 9 benchmark model variants + vision + embedding total ~70 GB and cannot coexist with OS and apps on a typical C: partition.
**Pros:** All models accessible. D: has headroom for additional models without impacting system stability. C: retains 69 GB free for OS/project use.
**Cons/Limitations:** Models are on a separate drive — if D: is not present (e.g., different machine), Ollama will not find models. Must set `OLLAMA_MODELS=D:\ollama_models` before starting Ollama.
**Impact:** `start_ollama.ps1` now sets `OLLAMA_MODELS` alongside Vulkan vars. `.env` and `.env.example` updated. Anyone cloning the repo must set this path to match their own machine.
**What's lacking:** No automated check that D: is available before starting Ollama.
**Ideal state:** `verify_gpu.py` checks `OLLAMA_MODELS` exists and warns if the path is missing or empty.
