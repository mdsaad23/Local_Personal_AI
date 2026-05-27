"""
Chat page — conversational AI with hybrid RAG retrieval, memory injection,
context compression, and inline source citation.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import streamlit as st

from src.generation.router import retrieve_for_query, build_prompt
from src.generation.ollama_client import stream_response, check_ollama_health
from src.memory.episodic import retrieve_relevant, extract_and_store
from src.memory.compressor import should_compress, compress
from src.memory.session import new_session, add_message, get_messages
from ui.components.source_viewer import render_sources

st.header("💬 Chat")

# ── Session init ────────────────────────────────────────────────────────────
if "session_id" not in st.session_state:
    st.session_state.session_id = new_session()
if "messages" not in st.session_state:
    st.session_state.messages = []
if "last_chunks" not in st.session_state:
    st.session_state.last_chunks = []

# ── Ollama health check ──────────────────────────────────────────────────────
if not check_ollama_health():
    st.error("Ollama is offline. Start it with: `.\\scripts\\start_ollama.ps1`")
    st.stop()

# ── Display history ──────────────────────────────────────────────────────────
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# ── Source display (after last assistant turn) ────────────────────────────────
if st.session_state.last_chunks:
    render_sources(st.session_state.last_chunks)

# ── Input ────────────────────────────────────────────────────────────────────
if prompt := st.chat_input("Ask anything about your documents …"):
    # Add user message
    st.session_state.messages.append({"role": "user", "content": prompt})
    add_message(st.session_state.session_id, "user", prompt)

    with st.chat_message("user"):
        st.markdown(prompt)

    # Retrieve
    with st.spinner("Retrieving …"):
        chunks, route = retrieve_for_query(prompt)
        memories = retrieve_relevant(prompt, limit=3)
        st.session_state.last_chunks = chunks

    # Compress context if approaching limit
    working_msgs = list(st.session_state.messages)
    if should_compress(working_msgs):
        working_msgs = compress(working_msgs)
        st.session_state.messages = [m for m in working_msgs if m["role"] != "system"]

    # Build prompt
    messages_for_llm = build_prompt(prompt, chunks, memories)

    # Stream response
    with st.chat_message("assistant"):
        output_placeholder = st.empty()
        full_response = []
        ollama_metrics = {}

        gen = stream_response(messages_for_llm)
        try:
            while True:
                token = next(gen)
                full_response.append(token)
                output_placeholder.markdown("".join(full_response) + "▌")
        except StopIteration as e:
            ollama_metrics = e.value or {}

        final_text = "".join(full_response)
        output_placeholder.markdown(final_text)

        # Metrics footer
        ttft = ollama_metrics.get("ttft_s")
        tgs = ollama_metrics.get("tgs")
        if ttft and tgs:
            st.caption(f"TTFT: {ttft:.2f}s · {tgs:.0f} tok/s · route: {route}")

    # Persist assistant message
    st.session_state.messages.append({"role": "assistant", "content": final_text})
    add_message(st.session_state.session_id, "assistant", final_text)

    # Show sources
    if chunks:
        render_sources(chunks)

    # Async memory extraction every 10 turns
    turn_count = len([m for m in st.session_state.messages if m["role"] == "user"])
    if turn_count % 10 == 0:
        all_msgs = [{"role": m["role"], "content": m["content"]}
                    for m in st.session_state.messages]
        extract_and_store(all_msgs, st.session_state.session_id)

# ── Sidebar controls ─────────────────────────────────────────────────────────
with st.sidebar:
    if st.button("🗑️ Clear conversation"):
        from src.memory.session import end_session
        extract_and_store(
            [{"role": m["role"], "content": m["content"]}
             for m in st.session_state.messages],
            st.session_state.session_id,
        )
        end_session(st.session_state.session_id)
        st.session_state.messages = []
        st.session_state.session_id = new_session()
        st.session_state.last_chunks = []
        st.rerun()
