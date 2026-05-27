"""Sidebar system status — Ollama health, model, context usage."""
from __future__ import annotations

import streamlit as st


def render_system_status() -> None:
    from src.generation.ollama_client import check_ollama_health, list_local_models
    from config.settings import PRODUCTION_MODEL, CONTEXT_LENGTH

    ollama_ok = check_ollama_health()
    status_colour = "🟢" if ollama_ok else "🔴"
    st.markdown(f"**Ollama** {status_colour} {'Online' if ollama_ok else 'Offline'}")

    if ollama_ok:
        models = list_local_models()
        active = PRODUCTION_MODEL
        active_label = active if active in models else f"{active} ⚠️ not pulled"
        st.caption(f"Model: `{active_label}`")
        st.caption(f"Context: {CONTEXT_LENGTH:,} tokens")

    # Session token usage (rough estimate from session_state)
    messages = st.session_state.get("messages", [])
    if messages:
        total_chars = sum(len(m.get("content", "")) for m in messages)
        token_est = total_chars // 4
        pct = min(token_est / CONTEXT_LENGTH, 1.0)
        st.caption(f"Context used: ~{token_est:,} tokens ({pct:.0%})")
        st.progress(pct)
