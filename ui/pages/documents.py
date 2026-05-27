"""
Document library manager — upload, view, and delete ingested documents.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import streamlit as st
import tempfile

st.header("📄 Documents")

# ── Upload ────────────────────────────────────────────────────────────────────
st.subheader("Add Document")
uploaded = st.file_uploader(
    "Drop a file to ingest",
    type=["pdf", "docx", "md", "txt", "png", "jpg", "jpeg", "webp", "gif"],
)

if uploaded:
    with tempfile.NamedTemporaryFile(
        delete=False, suffix=Path(uploaded.name).suffix
    ) as tmp:
        tmp.write(uploaded.getbuffer())
        tmp_path = Path(tmp.name)

    with st.spinner(f"Ingesting {uploaded.name} …"):
        from src.ingestion.parsers import parse_file
        from src.ingestion.chunker import chunk_pages
        from src.ingestion.embedder import embed_chunks
        from src.retrieval.dense import store_chunks
        from src.retrieval.sparse import add_chunks_to_bm25
        from src.ingestion.graph_builder import build_graph_from_chunks

        try:
            sections = parse_file(tmp_path)
            chunks = chunk_pages(sections)
            embedded = embed_chunks(chunks)
            store_chunks(embedded)
            add_chunks_to_bm25(embedded)
            build_graph_from_chunks(embedded)
            st.success(
                f"Ingested **{uploaded.name}** → "
                f"{len(sections)} sections · {len(embedded)} chunks"
            )
        except Exception as e:
            st.error(f"Ingestion failed: {e}")
        finally:
            tmp_path.unlink(missing_ok=True)

st.divider()

# ── Document list ─────────────────────────────────────────────────────────────
st.subheader("Ingested Documents")

from src.retrieval.dense import list_documents
docs = list_documents()

if not docs:
    st.info("No documents yet. Upload one above or drop files into `data/documents/` and run the watcher.")
else:
    for doc in docs:
        col1, col2, col3, col4 = st.columns([3, 1, 1, 1])
        col1.markdown(f"**{doc['source']}**")
        col2.caption(doc.get("file_type", ""))
        col3.caption(f"{doc.get('chunk_count', 0)} chunks")
        if col4.button("🗑️", key=f"del_{doc['doc_id']}"):
            from src.retrieval.dense import delete_doc
            delete_doc(doc["doc_id"])
            st.rerun()
