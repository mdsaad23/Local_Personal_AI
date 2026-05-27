"""Inline source citation display — renders retrieved chunks as expandable cards."""
from __future__ import annotations

from typing import Any

import streamlit as st


def render_sources(chunks: list[dict[str, Any]]) -> None:
    if not chunks:
        return

    with st.expander(f"📎 {len(chunks)} source{'s' if len(chunks) != 1 else ''} retrieved", expanded=False):
        for i, chunk in enumerate(chunks):
            source = chunk.get("source", "unknown")
            page = chunk.get("page", "")
            section = chunk.get("section", "")
            score = chunk.get("rerank_score") or chunk.get("rrf_score") or chunk.get("score")
            retrieval = chunk.get("retrieval", "")

            label_parts = [f"**[{i+1}] {source}**"]
            if page:
                label_parts.append(f"p.{page}")
            if section:
                label_parts.append(f"§ {section}")
            if score is not None:
                label_parts.append(f"score={score:.3f}")
            if retrieval:
                label_parts.append(f"via {retrieval}")

            st.markdown(" · ".join(label_parts))
            st.markdown(f"```\n{chunk.get('text', '')[:500]}\n```")
            if i < len(chunks) - 1:
                st.divider()
