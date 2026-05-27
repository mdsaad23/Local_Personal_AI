"""
GraphRAG knowledge graph visualisation using pyvis.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import streamlit as st
import streamlit.components.v1 as components
import tempfile

st.header("🕸️ Knowledge Graph")

from src.ingestion.graph_builder import get_graph
g = get_graph()

st.metric("Nodes", g.number_of_nodes())
st.metric("Edges", g.number_of_edges())

if g.number_of_nodes() == 0:
    st.info("Graph is empty. Ingest documents with entities to populate it.")
    st.stop()

# Filter controls
max_nodes = st.slider("Max nodes to display", 50, min(500, g.number_of_nodes()), 150)

# Node type filter
all_labels = list({data.get("label", "UNKNOWN") for _, data in g.nodes(data=True)})
selected_labels = st.multiselect("Filter by entity type", all_labels, default=all_labels)

# Build subgraph
filtered_nodes = [
    n for n, d in g.nodes(data=True)
    if d.get("label", "UNKNOWN") in selected_labels
][:max_nodes]
subgraph = g.subgraph(filtered_nodes)

# Render with pyvis
from pyvis.network import Network

label_colours = {
    "PERSON": "#4CAF50",
    "ORG": "#2196F3",
    "GPE": "#FF9800",
    "DATE": "#9C27B0",
    "MONEY": "#F44336",
    "EVENT": "#00BCD4",
    "TASK": "#FF5722",
}

net = Network(height="600px", width="100%", bgcolor="#0e1117", font_color="white")
net.barnes_hut(gravity=-5000, central_gravity=0.3)

for node_id, attrs in subgraph.nodes(data=True):
    label = attrs.get("label", "?")
    text = attrs.get("text", node_id)
    colour = label_colours.get(label, "#888888")
    net.add_node(node_id, label=text[:30], color=colour, title=f"{label}: {text}")

for src, dst, attrs in subgraph.edges(data=True):
    relation = attrs.get("relation", "")
    net.add_edge(src, dst, label=relation, title=relation)

with tempfile.NamedTemporaryFile(delete=False, suffix=".html", mode="w", encoding="utf-8") as f:
    net.save_graph(f.name)
    html_path = f.name

with open(html_path, encoding="utf-8") as f:
    html_content = f.read()

components.html(html_content, height=620, scrolling=False)
Path(html_path).unlink(missing_ok=True)

# Entity table
st.subheader("Entity Table")
import pandas as pd
rows = [
    {"Entity": d.get("text", n), "Type": d.get("label", "?"),
     "Documents": len(d.get("doc_ids", [])),
     "Connections": subgraph.degree(n) if n in subgraph else 0}
    for n, d in g.nodes(data=True)
    if d.get("label", "?") in selected_labels
]
if rows:
    df = pd.DataFrame(rows).sort_values("Connections", ascending=False)
    st.dataframe(df, use_container_width=True)
