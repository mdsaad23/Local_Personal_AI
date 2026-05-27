"""
Central configuration — single source of truth for all paths, model names,
and tunable parameters. Everything reads from .env with safe defaults.
"""

import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

# ── Project layout ─────────────────────────────────────────────────────────────
ROOT_DIR = Path(__file__).parent.parent

DOCUMENTS_DIR   = ROOT_DIR / os.getenv("DOCUMENTS_DIR",   "data/documents")
DB_DIR          = ROOT_DIR / os.getenv("DB_DIR",          "data/db")
BENCHMARKS_DIR  = ROOT_DIR / os.getenv("BENCHMARKS_DIR",  "data/benchmarks")
PROCESSED_DIR   = ROOT_DIR / "data/processed"

LANCEDB_DIR    = DB_DIR / "lancedb"
BM25_DIR       = DB_DIR / "bm25"
GRAPH_DIR      = DB_DIR / "graph"
MEMORY_DB_PATH = DB_DIR / "memory.db"
GRAPH_PATH     = GRAPH_DIR / "graph.json"

# Ensure all directories exist at import time
for _d in [DOCUMENTS_DIR, DB_DIR, BENCHMARKS_DIR, PROCESSED_DIR,
           LANCEDB_DIR, BM25_DIR, GRAPH_DIR]:
    _d.mkdir(parents=True, exist_ok=True)

# ── Ollama ─────────────────────────────────────────────────────────────────────
OLLAMA_BASE_URL    = os.getenv("OLLAMA_BASE_URL",    "http://localhost:11434")
PRODUCTION_MODEL   = os.getenv("PRODUCTION_MODEL",   "qwen3:14b-q4_K_M")
VISION_MODEL       = os.getenv("VISION_MODEL",       "minicpm-v")
EMBED_MODEL        = os.getenv("EMBED_MODEL",        "nomic-embed-text")

# ── Inference ──────────────────────────────────────────────────────────────────
CONTEXT_LENGTH    = int(os.getenv("CONTEXT_LENGTH",   "32768"))
KV_CACHE_TYPE     = os.getenv("KV_CACHE_TYPE",        "q8_0")

# When conversation reaches this fraction of CONTEXT_LENGTH, compress history
CONTEXT_COMPRESSION_THRESHOLD = float(
    os.getenv("CONTEXT_COMPRESSION_THRESHOLD", "0.80")
)

# ── Chunking ───────────────────────────────────────────────────────────────────
CHUNK_SIZE        = 512    # tokens — semantic chunker target size
CHUNK_OVERLAP     = 64     # tokens — overlap between consecutive chunks
MIN_CHUNK_SIZE    = 128    # tokens — discard chunks smaller than this

# ── Retrieval ──────────────────────────────────────────────────────────────────
TOP_K_DENSE       = 10     # candidates from LanceDB ANN search
TOP_K_SPARSE      = 10     # candidates from BM25
TOP_K_RERANK      = 5      # final chunks passed to LLM after reranking
RRF_K             = 60     # Reciprocal Rank Fusion constant (standard: 60)
HYDE_ENABLED      = True   # Hypothetical Document Embeddings for hard queries

# ── GraphRAG ───────────────────────────────────────────────────────────────────
GRAPH_MAX_HOPS       = 2   # max relationship hops during graph traversal
GRAPH_TOP_K_ENTITIES = 5   # entities returned per graph query
# spaCy model — install with: python -m spacy download en_core_web_sm
SPACY_MODEL          = "en_core_web_sm"

# ── Reranker ───────────────────────────────────────────────────────────────────
RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L6-v2"

# ── Memory ─────────────────────────────────────────────────────────────────────
# Max conversation turns kept in working context before compression triggers
MAX_WORKING_MEMORY_TURNS = 10
# How many episodic memories to inject into system prompt at session start
EPISODIC_INJECT_COUNT    = 5

# ── Judge / evaluation ─────────────────────────────────────────────────────────
DEEPSEEK_API_KEY  = os.getenv("DEEPSEEK_API_KEY", "")
DEEPSEEK_BASE_URL = "https://api.deepseek.com"
DEEPSEEK_MODEL    = "deepseek-chat"   # DeepSeek-V3

# ── Benchmark ──────────────────────────────────────────────────────────────────
BENCHMARK_QUERY_COUNT   = 50
BENCHMARK_WARMUP_RUNS   = 3    # discarded — let GPU/CPU warm up before timing
BENCHMARK_TIMEOUT_SEC   = 120  # abort single query if it exceeds this

# ── Supported file types ───────────────────────────────────────────────────────
SUPPORTED_EXTENSIONS = {".pdf", ".docx", ".md", ".txt",
                        ".png", ".jpg", ".jpeg", ".webp", ".gif"}
IMAGE_EXTENSIONS     = {".png", ".jpg", ".jpeg", ".webp", ".gif"}
