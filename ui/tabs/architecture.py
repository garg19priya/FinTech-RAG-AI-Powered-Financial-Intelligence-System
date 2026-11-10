import streamlit as st

from ui.markdown_table import render_table

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
            "Download the company-published 10-K PDF and cache it locally so ingestion is "
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
            "produces table-aware chunks of at most 800 tokens, tables kept as markdown.\n\n"
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
            "`core/embeddings.py: embed_dense()` calls OpenAI `text-embedding-3-small` or Gemini "
            "`gemini-embedding-001` (set by `EMBEDDING_PROVIDER`), both at `EMBEDDING_DIM = 1536`. "
            "The sparse side is Qdrant's BM25 FastEmbed model (`Qdrant/bm25`), invoked inline "
            "during indexing."
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
            "the Pipeline Comparison tab can query either independently. Qdrant runs as a Docker "
            "server (`QDRANT_URL`) or embedded in-process (`QDRANT_PATH`); see `core/clients.py`. "
            "`ingestion/prepare.py: prepare_data()` runs fetch, chunk and index, skipping what exists."
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
            "chunks, calls `core/llm.py: complete_with_usage()` (OpenAI `gpt-4o` or Gemini "
            "`gemini-3.5-flash`, set by `LLM_PROVIDER`), then parses the model's cited chunk "
            "indices back into `Citation` objects. If nothing is retrieved, or the model cites "
            "nothing, it returns a fixed \"not found\" answer instead of letting the model guess."
        ),
    },
]

FILE_MAP = [
    ("core/__init__.py", "Loads .env on import so API keys are available everywhere."),
    ("core/models.py", "Shared dataclasses: FilingChunk, RetrievedChunk, Citation, RAGAnswer."),
    ("core/clients.py", "Qdrant client (Docker server or embedded) and BM25 sparse model."),
    ("core/embeddings.py", "Dense embedding calls (OpenAI or Gemini, set by EMBEDDING_PROVIDER)."),
    ("core/llm.py", "LLM completion calls (OpenAI or Gemini, set by LLM_PROVIDER), token/cost tracking."),
    ("core/retry.py", "Shared exponential-backoff retry wrapper for OpenAI and Gemini calls."),
    ("core/retrieval.py", "hybrid_search() -- dense + sparse fusion (RRF) over Qdrant."),
    ("core/retrieval_strategies.py", "Five retrieval strategies built on hybrid_search()."),
    ("core/rag.py", "answer_question() -- the single public entrypoint: retrieve, prompt, cite."),
    ("eval/golden_dataset.py", "Curated question set with expected sections/reference answers."),
    ("eval/retrieval_metrics.py", "Deterministic hit rate / recall@k / MRR against expected chunks."),
    ("eval/judge.py", "LLM-as-judge faithfulness and correctness scoring."),
    ("eval/run_eval.py", "CLI: runs the golden set against a chosen pipeline/strategy."),
    ("ingestion/fetch_filing_pdf.py", "Downloads and caches the source PDF."),
    ("ingestion/fetch_filings.py", "Company/ticker/CIK metadata shared across ingestion."),
    ("ingestion/chunk_filings_basic.py", "PyPDFLoader + regex + recursive splitter chunking."),
    ("ingestion/chunk_filings_advanced.py", "docling HybridChunker, table-aware chunking."),
    ("ingestion/index_to_qdrant.py", "Embeds and upserts chunks into a Qdrant collection."),
    ("ingestion/index_comparison_collections.py", "Runs both pipelines into their own collections."),
    ("ingestion/prepare.py", "One command for the whole ingestion; skips what already exists."),
]

PRODUCTION_CONSIDERATIONS = [
    {
        "title": "Cost, latency & token tracking",
        "status": "Implemented",
        "detail": (
            "`core/llm.py: complete_with_usage()` returns an `LLMResponse` with input/output "
            "token counts and an estimated USD cost per call. `RAGAnswer` carries these through "
            "so the Evaluation tab can show cost-per-strategy, not just latency."
        ),
    },
    {
        "title": "Retry / backoff on API calls",
        "status": "Implemented",
        "detail": (
            "`core/retry.py: with_retry()` wraps the OpenAI and Gemini completion and embedding calls "
            "with exponential backoff on transient errors (connection, rate limit, timeout, "
            "5xx) -- previously a single flaky API call would kill an entire eval run or demo "
            "question."
        ),
    },
    {
        "title": "Observability & monitoring",
        "status": "Documented (not built)",
        "detail": (
            "No structured tracing (e.g. OpenTelemetry/LangSmith spans per retrieve/generate "
            "step), no dashboards on error rate, latency, or refusal rate. Today's only signal "
            "is stdout or the Streamlit UI itself."
        ),
    },
    {
        "title": "Guardrails / prompt-injection defense",
        "status": "Documented (not built)",
        "detail": (
            "`core/rag.py: _build_prompt()` embeds raw retrieved filing text verbatim into the "
            "prompt with no sanitization. An adversarially-crafted excerpt in the corpus could "
            "carry instructions the model follows."
        ),
    },
    {
        "title": "Secrets management & Qdrant auth",
        "status": "Documented (not built)",
        "detail": (
            "API keys via `.env` are fine for a demo but not for shared deployment. "
            "`core/clients.py: get_qdrant_client()` talks to `QDRANT_URL` with no API key and "
            "no TLS enforcement (or opens a local folder via `QDRANT_PATH`)."
        ),
    },
    {
        "title": "Rate limiting & cost controls",
        "status": "Documented (not built)",
        "detail": (
            "Nothing throttles concurrent `answer_question()` calls against a shared, paid "
            "LLM provider account -- a burst of traffic on a public demo instance has no ceiling."
        ),
    },
    {
        "title": "CI-gated eval regression thresholds",
        "status": "Documented (not built)",
        "detail": (
            "`eval/run_eval.py` reports hit rate, MRR, and judge scores but nothing fails a "
            "build if they regress. Wiring a minimum-score gate into CI is the natural next step."
        ),
    },
    {
        "title": "Data freshness & re-ingestion cadence",
        "status": "Documented (not built)",
        "detail": (
            "The corpus is a single, hand-fetched filing with no scheduled re-fetch from SEC "
            "EDGAR and no versioning of which filing vintage produced a given answer."
        ),
    },
    {
        "title": "Semantic caching",
        "status": "Documented (not built)",
        "detail": (
            "Repeated or near-duplicate questions re-pay full retrieval + generation cost. "
            "`st.cache_data` only caches within one running process, not across users/restarts."
        ),
    },
]


def render() -> None:
    st.caption(
        "How a question turns into a cited answer -- from PDF to response. "
        "Expand each stage for the underlying implementation."
    )
    st.graphviz_chart(FLOW_DIAGRAM, width="stretch")

    for stage in STAGES:
        with st.expander(stage["title"]):
            st.markdown(stage["high_level"])
            st.divider()
            st.markdown(stage["low_level"])

    st.divider()
    st.subheader("File map")
    render_table([{"File": f, "Responsibility": r} for f, r in FILE_MAP])

    st.divider()
    st.subheader("Production considerations")
    st.caption(
        "What this demo skips versus a real deployment. Two items are actually implemented; "
        "the rest are documented gaps, not built."
    )
    for item in PRODUCTION_CONSIDERATIONS:
        with st.expander(f"{item['title']} -- {item['status']}"):
            st.markdown(item["detail"])
