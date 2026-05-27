"""
Streamlit entry point — multi-page app.

Pages:
  Chat        — conversational AI with document grounding
  Documents   — document library manager
  Graph       — GraphRAG knowledge graph visualisation
  Benchmarks  — model comparison results viewer

Run with: streamlit run ui/app.py
"""
import sys
from pathlib import Path

# Ensure project root is on the path
sys.path.insert(0, str(Path(__file__).parent.parent))

import streamlit as st

st.set_page_config(
    page_title="Local AI Assistant",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Navigation
pages = {
    "💬 Chat": "ui/pages/chat.py",
    "📄 Documents": "ui/pages/documents.py",
    "🕸️ Knowledge Graph": "ui/pages/graph.py",
    "📊 Benchmarks": "ui/pages/benchmarks.py",
}

with st.sidebar:
    st.title("Local AI Assistant")
    st.caption("Fully offline · Private · On-device")
    st.divider()
    page = st.radio("Navigate", list(pages.keys()), label_visibility="collapsed")
    st.divider()

    # System status widget
    from ui.components.system_status import render_system_status
    render_system_status()

# Route to selected page
selected_path = pages[page]
with open(selected_path, encoding="utf-8") as f:
    code = f.read()
exec(compile(code, selected_path, "exec"), {"__name__": "__main__"})
