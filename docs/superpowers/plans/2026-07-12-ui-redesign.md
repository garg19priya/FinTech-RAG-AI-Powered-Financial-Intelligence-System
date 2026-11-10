# UI Redesign: Tabbed App + Architecture Tab Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Merge the two Streamlit pages (Q&A, Pipeline Comparison) into a single tabbed app (`ui/app.py`) with a new Architecture tab, and apply consistent, projector-friendly visual styling.

**Architecture:** `ui/app.py` becomes a thin entrypoint (page config, CSS, sidebar, `st.tabs(...)` dispatch). Each tab's logic lives in its own module under `ui/tabs/`, each exposing a single `render()` function. `ui/pages/` (Streamlit's file-based multipage mechanism) is removed.

**Tech Stack:** Streamlit 1.59 (`st.tabs`, `st.graphviz_chart` with a raw DOT string — no `graphviz` pip package or system `dot` binary required, confirmed by reading `streamlit.elements.graphviz_chart.marshall`, which takes a `str` on the `isinstance(figure_or_dot, str)` branch).

## Global Constraints

- Spec: `docs/superpowers/specs/2026-07-12-ui-redesign-design.md` — UI-only change, no `core`/`ingestion` behavior change, no live source-reading in Architecture tab.
- No automated tests apply (per spec) — verification is `python -c "import ..."` per task (catches syntax/import errors) plus one final manual click-through.
- Streamlit's `_fix_sys_path` (in `streamlit.web.bootstrap`) inserts only the running script's own directory (`ui/`) onto `sys.path`, not the repo root. `core.*` / `ingestion.*` imports work today only because those packages are separately installed editable (`pyproject.toml`'s `[tool.setuptools.packages.find] include = ["core*", "ingestion*"]`). `ui` is **not** in that include list. Therefore new tab modules must be imported as flat top-level modules (`from tabs import qa`), not as `ui.tabs.qa` — since `ui/` itself is what lands on `sys.path[0]`, `tabs/` inside it resolves as a top-level package with zero pyproject changes.

---

### Task 1: Streamlit theme config

**Files:**
- Create: `.streamlit/config.toml`

**Interfaces:**
- Produces: a `[theme]` section Streamlit reads automatically on startup — no code depends on this directly.

- [ ] **Step 1: Write the theme config**

```toml
[theme]
base = "dark"
primaryColor = "#2dd4bf"
backgroundColor = "#0e1117"
secondaryBackgroundColor = "#161b22"
textColor = "#e6edf3"
font = "sans serif"
```

- [ ] **Step 2: Verify it parses as valid TOML**

Run: `.venv/bin/python -c "import tomllib; print(tomllib.load(open('.streamlit/config.toml', 'rb')))"`
Expected: prints the parsed dict, e.g. `{'theme': {'base': 'dark', 'primaryColor': '#2dd4bf', ...}}` — no exception.

- [ ] **Step 3: Commit**

```bash
git add .streamlit/config.toml
git commit -m "feat: add dark fintech theme for Streamlit UI"
```

---

### Task 2: Extract the Q&A tab

**Files:**
- Create: `ui/tabs/__init__.py`
- Create: `ui/tabs/qa.py`
- Test: none (manual import check below)

**Interfaces:**
- Produces: `qa.COMPANIES: list[str]`, `qa.render() -> None` — called by `ui/app.py` (Task 5) inside `with tab_qa:`.
- Consumes: `core.models.RAGAnswer`, `core.rag.answer_question` (unchanged).

- [ ] **Step 1: Create the package marker**

```bash
mkdir -p ui/tabs
touch ui/tabs/__init__.py
```

- [ ] **Step 2: Write `ui/tabs/qa.py`**

This is today's `ui/app.py` body, with `st.set_page_config`, `st.title`, and the sidebar rendering removed (page config and the shared header/sidebar move to `ui/app.py` in Task 5, since they're shared chrome, not Q&A-specific):

```python
import streamlit as st

from core.models import RAGAnswer
from core.rag import answer_question

COMPANIES = [
    "Apple (AAPL)",
]


def render_answer(result: RAGAnswer) -> None:
    st.write(result.answer)
    if result.citations:
        with st.expander(f"Sources ({len(result.citations)})"):
            for citation in result.citations:
                st.markdown(
                    f"**{citation.company_name}** — {citation.section} "
                    f"(FY{citation.fiscal_year})"
                )
                st.caption(citation.text_snippet)


def render() -> None:
    question = st.text_input("Ask a question about these filings")
    if st.button("Ask") and question:
        with st.spinner("Retrieving and generating answer..."):
            result = answer_question(question)
        render_answer(result)
```

- [ ] **Step 3: Verify it imports cleanly**

Run: `.venv/bin/python -c "import sys; sys.path.insert(0, 'ui'); import tabs.qa as qa; print(qa.COMPANIES, qa.render)"`
Expected: `['Apple (AAPL)'] <function render at ...>` — no exception. (This simulates Streamlit's `sys.path` insertion of the `ui/` directory, per the Global Constraints note.)

- [ ] **Step 4: Commit**

```bash
git add ui/tabs/__init__.py ui/tabs/qa.py
git commit -m "refactor: extract Q&A tab into ui/tabs/qa.py"
```

---

### Task 3: Extract the Pipeline Comparison tab

**Files:**
- Create: `ui/tabs/pipeline_comparison.py`
- Delete: `ui/pages/1_Pipeline_Comparison.py`
- Delete: `ui/pages/` (now empty)

**Interfaces:**
- Produces: `pipeline_comparison.render() -> None` — called by `ui/app.py` (Task 5) inside `with tab_compare:`.
- Consumes: `core.rag.answer_question`, `ingestion.chunk_filings._token_length` (unchanged).

- [ ] **Step 1: Write `ui/tabs/pipeline_comparison.py`**

Identical logic to today's `ui/pages/1_Pipeline_Comparison.py`, with `main()` renamed to `render()`, the `st.set_page_config` call removed (now owned by `ui/app.py`), and `st.title` replaced by the existing `st.caption` (the page-level title is now the tab label itself):

```python
import json
import time
from pathlib import Path

import streamlit as st

from core.rag import answer_question
from ingestion.chunk_filings import _token_length

PIPELINES = {
    "Basic (PyPDFLoader + regex)": {
        "collection": "sec_filings_basic",
        "chunks_path": Path("data/chunks_basic/AAPL.json"),
    },
    "Advanced (docling)": {
        "collection": "sec_filings_advanced",
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
    token_counts = [_token_length(c["text"]) for c in chunks]
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


def render_single_query() -> None:
    st.subheader("Ask a question")
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
            st.dataframe(rows, use_container_width=True)


def render() -> None:
    st.caption(
        "Compares two PDF chunking pipelines (basic PyPDFLoader+regex vs. advanced docling) "
        "and five retrieval strategies over Apple's 10-K."
    )
    render_stats()
    st.divider()
    render_sample_chunk_viewer()
    st.divider()
    render_single_query()
    st.divider()
    render_compare_all()
```

- [ ] **Step 2: Verify it imports cleanly**

Run: `.venv/bin/python -c "import sys; sys.path.insert(0, 'ui'); import tabs.pipeline_comparison as pc; print(pc.PIPELINES.keys(), pc.render)"`
Expected: `dict_keys(['Basic (PyPDFLoader + regex)', 'Advanced (docling)']) <function render at ...>` — no exception.

- [ ] **Step 3: Delete the old multipage file and directory**

```bash
git rm ui/pages/1_Pipeline_Comparison.py
rmdir ui/pages
```

- [ ] **Step 4: Commit**

```bash
git add ui/tabs/pipeline_comparison.py
git commit -m "refactor: move Pipeline Comparison into ui/tabs, drop file-based multipage"
```

---

### Task 4: Build the Architecture tab

**Files:**
- Create: `ui/tabs/architecture.py`

**Interfaces:**
- Produces: `architecture.render() -> None` — called by `ui/app.py` (Task 5) inside `with tab_arch:`.
- Consumes: nothing outside `streamlit` — all content is static narrative text authored in this task, matching the current state of `core/` and `ingestion/` (per spec: no live source-reading).

- [ ] **Step 1: Write `ui/tabs/architecture.py`**

```python
import streamlit as st

FLOW_DIAGRAM = """
digraph {
    rankdir=LR;
    bgcolor="transparent";
    node [shape=box, style="rounded,filled", fillcolor="#161b22", fontcolor="#e6edf3", color="#2dd4bf", fontname="Helvetica"];
    edge [color="#2dd4bf"];
    fetch [label="Fetch (PDF)"];
    chunk [label="Chunk (basic | advanced)"];
    embed [label="Embed (dense + sparse)"];
    index [label="Index (Qdrant)"];
    retrieve [label="Retrieve (hybrid + RRF)"];
    generate [label="Generate (LLM, cited)"];
    fetch -> chunk -> embed -> index -> retrieve -> generate;
}
"""

STAGES = [
    {
        "title": "1. Fetch",
        "high_level": (
            "Pull the filing PDF from SEC EDGAR and cache it locally so ingestion is "
            "repeatable and offline."
        ),
        "low_level": (
            "`ingestion/fetch_filing_pdf.py: fetch_filing_pdf()` downloads Apple's 10-K PDF "
            "into `data/raw_pdf/AAPL.pdf`. `ingestion/fetch_filings.py` holds the `COMPANIES` "
            "metadata (ticker, CIK, name) used across ingestion."
        ),
    },
    {
        "title": "2. Chunk",
        "high_level": (
            "Split the filing into retrievable passages. Two competing strategies exist so "
            "the demo can compare naive vs. structure-aware chunking."
        ),
        "low_level": (
            "`ingestion/chunk_filings_basic.py`: PyPDFLoader extracts raw text, a regex finds "
            "Item boundaries, then `RecursiveCharacterTextSplitter` splits with "
            "`chunk_size=800`, `chunk_overlap=150` tokens.\n\n"
            "`ingestion/chunk_filings_advanced.py`: docling's `HybridChunker` (with an "
            "`OpenAITokenizer`) parses the PDF's real structure (tables, headings) and "
            "produces table-aware chunks.\n\n"
            "Both tag each chunk with a `pipeline` field (`FilingChunk.pipeline` in "
            "`core/models.py`) so the two runs never mix in Qdrant."
        ),
    },
    {
        "title": "3. Embed",
        "high_level": (
            "Turn each chunk into vectors for both semantic (dense) and keyword (sparse) "
            "search, so hybrid retrieval can use either signal."
        ),
        "low_level": (
            "`core/embeddings.py: embed_dense()` calls OpenAI's `text-embedding-3-small` "
            "(`EMBEDDING_DIM = 1536`). The sparse side is Qdrant's built-in BM25 FastEmbed "
            "model, invoked inline during indexing."
        ),
    },
    {
        "title": "4. Index",
        "high_level": (
            "Store every chunk's dense vector, sparse vector, and metadata (ticker, section, "
            "fiscal year) in Qdrant, ready for retrieval."
        ),
        "low_level": (
            "`ingestion/index_to_qdrant.py: index_all_filings()` upserts into one Qdrant "
            "collection per pipeline -- `sec_filings_basic` and `sec_filings_advanced` -- so "
            "the Pipeline Comparison tab can query either independently."
        ),
    },
    {
        "title": "5. Retrieve",
        "high_level": (
            "Given a question, fetch the most relevant chunks by combining dense and sparse "
            "search. Five interchangeable strategies let the demo show how each affects "
            "answer quality."
        ),
        "low_level": (
            "`core/retrieval.py: hybrid_search()` runs dense + sparse search in one Qdrant "
            "query, fused with Reciprocal Rank Fusion (`Fusion.RRF`), `top_k=8`.\n\n"
            "`core/retrieval_strategies.py` builds on top of it: `baseline` (hybrid_search "
            "directly), `query_rewrite` (LLM rewrites the question first), `hyde` (LLM drafts "
            "a hypothetical answer, embeds that instead), `multi_query` (LLM generates "
            "variants, RRF-fuses their results), `rerank` (over-fetches `top_k*3` candidates, "
            "re-scores with cross-encoder `Xenova/ms-marco-MiniLM-L-6-v2`)."
        ),
    },
    {
        "title": "6. Generate",
        "high_level": (
            "Produce a grounded, cited answer -- the model is instructed to answer only from "
            "the retrieved chunks and cite which ones it used."
        ),
        "low_level": (
            "`core/rag.py: answer_question()` builds a prompt constrained to the retrieved "
            "chunks, calls `core/llm.py: complete()` (OpenAI `gpt-4o`), then parses the "
            "model's cited chunk indices back into `Citation` objects. If no chunk is "
            "retrieved, it returns a fixed \"not found\" answer instead of letting the model "
            "guess."
        ),
    },
]

FILE_MAP = [
    ("core/__init__.py", "Loads .env on import so API keys are available everywhere."),
    ("core/models.py", "Shared dataclasses: FilingChunk, RetrievedChunk, Citation, RAGAnswer."),
    ("core/embeddings.py", "Dense embedding calls (OpenAI text-embedding-3-small)."),
    ("core/llm.py", "LLM completion calls (OpenAI gpt-4o)."),
    ("core/retrieval.py", "hybrid_search() -- dense + sparse fusion (RRF) over Qdrant."),
    ("core/retrieval_strategies.py", "Five retrieval strategies built on hybrid_search()."),
    ("core/rag.py", "answer_question() -- the single public entrypoint: retrieve, prompt, cite."),
    ("ingestion/fetch_filing_pdf.py", "Downloads and caches the source PDF."),
    ("ingestion/fetch_filings.py", "Company/ticker/CIK metadata shared across ingestion."),
    ("ingestion/chunk_filings_basic.py", "PyPDFLoader + regex + recursive splitter chunking."),
    ("ingestion/chunk_filings_advanced.py", "docling HybridChunker, table-aware chunking."),
    ("ingestion/index_to_qdrant.py", "Embeds and upserts chunks into a Qdrant collection."),
    ("ingestion/index_comparison_collections.py", "Runs both pipelines into their own collections."),
]


def render() -> None:
    st.caption(
        "How a question turns into a cited answer -- from PDF to response. "
        "Expand each stage for the underlying implementation."
    )
    st.graphviz_chart(FLOW_DIAGRAM, use_container_width=True)

    for stage in STAGES:
        with st.expander(stage["title"]):
            st.markdown(stage["high_level"])
            st.divider()
            st.markdown(stage["low_level"])

    st.divider()
    st.subheader("File map")
    st.dataframe(
        [{"File": f, "Responsibility": r} for f, r in FILE_MAP],
        use_container_width=True,
        hide_index=True,
    )
```

- [ ] **Step 2: Verify it imports cleanly**

Run: `.venv/bin/python -c "import sys; sys.path.insert(0, 'ui'); import tabs.architecture as arch; assert len(arch.STAGES) == 6; assert len(arch.FILE_MAP) == 13; print('ok')"`
Expected: `ok` — no exception.

- [ ] **Step 3: Commit**

```bash
git add ui/tabs/architecture.py
git commit -m "feat: add Architecture tab with flow diagram and per-stage narrative"
```

---

### Task 5: Rewrite `ui/app.py` as the tabbed entrypoint

**Files:**
- Modify: `ui/app.py` (full rewrite)

**Interfaces:**
- Consumes: `tabs.qa.render`, `tabs.qa.COMPANIES`, `tabs.pipeline_comparison.render`, `tabs.architecture.render` (all from Tasks 2-4).

- [ ] **Step 1: Rewrite `ui/app.py`**

```python
import streamlit as st

from tabs import architecture, pipeline_comparison, qa

CUSTOM_CSS = """
<style>
.block-container {
    padding-top: 2rem;
    padding-bottom: 3rem;
    max-width: 1100px;
}
h1 {
    font-weight: 700;
    letter-spacing: -0.02em;
}
.stTabs [data-baseweb="tab-list"] {
    gap: 4px;
}
.stTabs [data-baseweb="tab"] {
    padding: 8px 18px;
    border-radius: 8px 8px 0 0;
}
div[data-testid="stExpander"] {
    border: 1px solid rgba(45, 212, 191, 0.25);
    border-radius: 10px;
}
div[data-testid="stMetric"] {
    background: rgba(45, 212, 191, 0.06);
    border: 1px solid rgba(45, 212, 191, 0.18);
    border-radius: 10px;
    padding: 12px 14px;
}
</style>
"""


def render_sidebar() -> None:
    with st.sidebar:
        st.subheader("Companies in this demo")
        for company in qa.COMPANIES:
            st.write(f"- {company}")


def main() -> None:
    st.set_page_config(page_title="FinTech RAG Demo", layout="wide")
    st.markdown(CUSTOM_CSS, unsafe_allow_html=True)
    st.title("SEC Filing Q&A")
    st.caption(
        "A retrieval-augmented generation demo over Apple's FY2024 10-K, "
        "with citations grounded in the source text."
    )
    render_sidebar()

    tab_qa, tab_compare, tab_arch = st.tabs(["Q&A", "Pipeline Comparison", "Architecture"])
    with tab_qa:
        qa.render()
    with tab_compare:
        pipeline_comparison.render()
    with tab_arch:
        architecture.render()


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Verify the full module graph imports cleanly**

Run: `.venv/bin/python -c "import sys; sys.path.insert(0, 'ui'); import app; print(app.main)"`
Expected: `<function main at ...>` — no exception. This confirms `tabs.qa`, `tabs.pipeline_comparison`, and `tabs.architecture` all resolve together the same way Streamlit will resolve them at runtime.

- [ ] **Step 3: Commit**

```bash
git add ui/app.py
git commit -m "feat: merge Q&A and Pipeline Comparison into a single tabbed app"
```

---

### Task 6: Manual verification pass

**Files:** none (verification only)

- [ ] **Step 1: Start the app**

Run: `.venv/bin/streamlit run ui/app.py`
Expected: server starts, browser opens to `http://localhost:8501`, no traceback in the terminal.

- [ ] **Step 2: Check the Q&A tab**

Ask a question (e.g. "What was Apple's total net sales for fiscal year 2024?"). Expected: same behavior as before the refactor — spinner, answer text, expandable "Sources" section with citations. Sidebar still lists "Apple (AAPL)".

- [ ] **Step 3: Check the Pipeline Comparison tab**

Expected: chunking stats, sample chunk viewer, single-query form, and "Compare all" expander all render and function exactly as before (this tab's logic is unchanged, only relocated).

- [ ] **Step 4: Check the Architecture tab**

Expected: the six-stage flow diagram renders (Fetch → Chunk → Embed → Index → Retrieve → Generate), each stage expands to show high-level + low-level text, and the file map table lists all 13 files.

- [ ] **Step 5: Confirm no `ui/pages` sidebar navigation remains**

Expected: Streamlit's default multipage sidebar nav (page links) is gone — only the tab bar and the "Companies in this demo" sidebar content remain.

- [ ] **Step 6: Stop the server**

Press `Ctrl+C` in the terminal running Streamlit.

No commit for this task — it's verification only, not a code change.
