"""
Benchmark results viewer — loads CSV output from benchmark.py and renders
interactive comparison charts.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from config.settings import BENCHMARKS_DIR

st.header("📊 Benchmark Results")

# ── Load data ─────────────────────────────────────────────────────────────────
csv_files = sorted(BENCHMARKS_DIR.glob("benchmark_*.csv"), reverse=True)
if not csv_files:
    st.info("No benchmark results yet. Run: `python src/evaluation/benchmark.py`")
    st.stop()

selected_file = st.selectbox(
    "Select benchmark run",
    csv_files,
    format_func=lambda p: p.stem,
)

df = pd.read_csv(selected_file)
df = df[df["error"].isna() | (df["error"] == "")]

# ── Summary table ─────────────────────────────────────────────────────────────
st.subheader("Model Summary")
summary = (
    df.groupby("model_name")
    .agg(
        avg_tgs=("tgs", "mean"),
        avg_ttft=("ttft_s", "mean"),
        avg_faith=("faithfulness", "mean"),
        avg_relevancy=("answer_relevancy", "mean"),
        query_count=("query", "count"),
    )
    .reset_index()
    .sort_values("avg_tgs", ascending=False)
)
summary.columns = ["Model", "Avg TGS (tok/s)", "Avg TTFT (s)", "Faithfulness", "Relevancy", "Queries"]
st.dataframe(summary.style.format({
    "Avg TGS (tok/s)": "{:.1f}",
    "Avg TTFT (s)": "{:.2f}",
    "Faithfulness": "{:.2f}",
    "Relevancy": "{:.2f}",
}), use_container_width=True)

# ── TGS comparison bar chart ──────────────────────────────────────────────────
st.subheader("Token Generation Speed")
fig_tgs = px.bar(
    summary.sort_values("Avg TGS (tok/s)"),
    x="Avg TGS (tok/s)", y="Model", orientation="h",
    color="Avg TGS (tok/s)", color_continuous_scale="Blues",
    title="Average Token Generation Speed (higher is better)",
)
fig_tgs.update_layout(showlegend=False, height=400)
st.plotly_chart(fig_tgs, use_container_width=True)

# ── Quality vs Speed scatter ───────────────────────────────────────────────────
st.subheader("Quality vs Speed Tradeoff")
fig_scatter = px.scatter(
    summary,
    x="Avg TGS (tok/s)", y="Faithfulness",
    size="Avg TTFT (s)",
    text="Model",
    title="Faithfulness vs Generation Speed (bubble size = TTFT)",
    color="Relevancy",
    color_continuous_scale="RdYlGn",
)
fig_scatter.update_traces(textposition="top center")
fig_scatter.update_layout(height=500)
st.plotly_chart(fig_scatter, use_container_width=True)

# ── TTFT distribution ─────────────────────────────────────────────────────────
st.subheader("Time to First Token Distribution")
fig_ttft = px.box(
    df[df["ttft_s"].notna()],
    x="model_name", y="ttft_s",
    title="TTFT Distribution per Model (lower is better)",
    color="model_name",
)
fig_ttft.update_layout(showlegend=False, xaxis_tickangle=-45, height=400)
st.plotly_chart(fig_ttft, use_container_width=True)

# ── Raw data ──────────────────────────────────────────────────────────────────
with st.expander("Raw data"):
    st.dataframe(df, use_container_width=True)
    csv_bytes = df.to_csv(index=False).encode("utf-8")
    st.download_button("Download CSV", csv_bytes, file_name=selected_file.name)
