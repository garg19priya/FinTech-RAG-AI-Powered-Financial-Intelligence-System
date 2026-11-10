import json
import time
from pathlib import Path

import streamlit as st

from core.clients import collection_for
from core.rag import answer_question
from ingestion.text_utils import token_length
from ui.markdown_table import render_table
from ui.tabs.qa import SAMPLE_QUESTIONS

PIPELINES = {
    "Basic (PyPDFLoader + regex)": {
        "collection": collection_for("sec_filings_basic"),
        "chunks_path": Path("data/chunks_basic/AAPL.json"),
    },
    "Advanced (docling)": {
        "collection": collection_for("sec_filings_advanced"),
        "chunks_path": Path("data/chunks_advanced/AAPL.json"),
    },
}

STRATEGY_LABELS = {
    "Baseline": "baseline",
    "Query rewrite": "query_rewrite",
    "HyDE": "hyde",
    "Multi-query fusion": "multi_query",
    "Re-rank": "rerank",
}


@st.cache_data
def load_chunks(path_str: str) -> list[dict]:
    path = Path(path_str)
    if not path.exists():
        return []
    return json.loads(path.read_text(encoding="utf-8"))


def is_table_chunk(text: str) -> bool:
    return text.count("|") > 5


def chunk_stats(chunks: list[dict]) -> dict:
    if not chunks:
        return {"count": 0, "avg_tokens": 0, "table_chunks": 0}
    token_counts = [token_length(c["text"]) for c in chunks]
    return {
        "count": len(chunks),
        "avg_tokens": round(sum(token_counts) / len(token_counts)),
        "table_chunks": sum(1 for c in chunks if is_table_chunk(c["text"])),
    }


@st.cache_data(show_spinner=False)
def run_query(question: str, collection: str, strategy: str) -> dict:
    start = time.perf_counter()
    result = answer_question(question, collection_name=collection, strategy=strategy)
    elapsed = time.perf_counter() - start
    return {
        "answer": result.answer,
        "citations": [
            {"section": c.section, "text_snippet": c.text_snippet} for c in result.citations
        ],
        "elapsed": elapsed,
    }


def render_stats() -> None:
    st.subheader("Chunking stats")
    cols = st.columns(len(PIPELINES))
    for col, (label, cfg) in zip(cols, PIPELINES.items()):
        chunks = load_chunks(str(cfg["chunks_path"]))
        stats = chunk_stats(chunks)
        with col:
            st.markdown(f"**{label}**")
            st.metric("Chunks", stats["count"])
            st.metric("Avg tokens/chunk", stats["avg_tokens"])
            st.metric("Chunks with tables", stats["table_chunks"])


def render_sample_chunk_viewer() -> None:
    st.subheader("Sample table chunk: basic vs. advanced")
    basic_chunks = load_chunks(str(PIPELINES["Basic (PyPDFLoader + regex)"]["chunks_path"]))
    advanced_chunks = load_chunks(str(PIPELINES["Advanced (docling)"]["chunks_path"]))

    basic_table = next((c for c in basic_chunks if is_table_chunk(c["text"])), None)
    advanced_table = next((c for c in advanced_chunks if is_table_chunk(c["text"])), None)

    left, right = st.columns(2)
    with left:
        st.caption("Basic")
        if basic_table:
            st.code(basic_table["text"][:800], language=None)
        else:
            st.write("No table-like chunk found.")
    with right:
        st.caption("Advanced")
        if advanced_table:
            st.markdown(advanced_table["text"][:800])
        else:
            st.write("No table-like chunk found.")


def render_sample_question_chips(target_key: str, key_prefix: str) -> None:
    cols = st.columns(len(SAMPLE_QUESTIONS))
    for col, (label, sample) in zip(cols, SAMPLE_QUESTIONS):
        if col.button(label, help=sample, width="stretch", key=f"{key_prefix}_{label}"):
            st.session_state[target_key] = sample


def render_single_query() -> None:
    st.subheader("Ask a question")
    render_sample_question_chips("single_question", "single")
    question = st.text_input("Question", key="single_question")
    pipeline_label = st.selectbox("Chunking pipeline", list(PIPELINES.keys()))
    strategy_label = st.selectbox("Retrieval strategy", list(STRATEGY_LABELS.keys()))

    if st.button("Run") and question:
        collection = PIPELINES[pipeline_label]["collection"]
        strategy = STRATEGY_LABELS[strategy_label]
        with st.spinner("Retrieving and generating..."):
            result = run_query(question, collection, strategy)
        st.write(result["answer"])
        st.caption(f"{result['elapsed']:.1f}s")
        if result["citations"]:
            with st.expander(f"Sources ({len(result['citations'])})"):
                for citation in result["citations"]:
                    st.markdown(f"**{citation['section']}**")
                    st.caption(citation["text_snippet"])


def render_compare_all() -> None:
    with st.expander("Compare all (2 pipelines x 5 strategies -- slower, several extra LLM calls)"):
        render_sample_question_chips("compare_question", "compare")
        question = st.text_input("Question", key="compare_question")
        if st.button("Run full matrix") and question:
            rows = []
            progress = st.progress(0.0)
            combos = [
                (p_label, s_label)
                for p_label in PIPELINES
                for s_label in STRATEGY_LABELS
            ]
            for i, (p_label, s_label) in enumerate(combos):
                collection = PIPELINES[p_label]["collection"]
                strategy = STRATEGY_LABELS[s_label]
                result = run_query(question, collection, strategy)
                rows.append(
                    {
                        "Pipeline": p_label,
                        "Strategy": s_label,
                        "Latency (s)": round(result["elapsed"], 1),
                        "Answer": result["answer"],
                    }
                )
                progress.progress((i + 1) / len(combos))
            render_table(rows)


def render() -> None:
    st.caption(
        "Compares two PDF chunking pipelines (basic PyPDFLoader+regex vs. advanced docling) "
        "and five retrieval strategies over Apple's 10-K."
    )

    tab_overview, tab_try_it = st.tabs(["Overview", "Try It"])
    with tab_overview:
        render_stats()
        st.divider()
        render_sample_chunk_viewer()
    with tab_try_it:
        render_single_query()
        st.divider()
        render_compare_all()
