# UI Redesign: Tabbed App + Architecture Tab — Design

## Purpose

The current UI is split across two Streamlit sidebar pages (`ui/app.py` for Q&A, `ui/pages/1_Pipeline_Comparison.py` for chunking/retrieval comparison) with default Streamlit styling. For presenting this project as a talk demo, the presenter needs:

1. A single app they can drive from one URL without sidebar navigation.
2. A visually polished, projector-friendly look.
3. An "Architecture" tab that lets them narrate the system end-to-end — both high-level (what each stage does and why) and low-level (actual functions, parameters, files) — without switching to an editor.

## Structure

- `ui/app.py` becomes the single entrypoint: page config, shared header, injected CSS, and `st.tabs(["Q&A", "Pipeline Comparison", "Architecture"])` dispatching to per-tab `render()` functions.
- `ui/tabs/qa.py` — today's `ui/app.py` body (question box, answer, citations).
- `ui/tabs/pipeline_comparison.py` — today's `ui/pages/1_Pipeline_Comparison.py` body, relocated, with a `render()` entrypoint. Logic unchanged.
- `ui/tabs/architecture.py` — new. Flow diagram + per-stage narrative + file map (see below).
- `ui/pages/` is deleted — file-based multipage nav is replaced by the tab bar.

## Architecture tab content

**Diagram:** one horizontal Graphviz diagram (`st.graphviz_chart`) at the top of the tab:

```
fetch (PDF) -> chunk (basic|advanced) -> embed (dense+sparse) -> index (Qdrant) -> retrieve (hybrid+RRF) -> generate (LLM, cited)
```

**Per-stage sections** (one `st.expander` per stage), each with a high-level line and a low-level sub-block naming the actual current implementation:

| Stage | High-level | Low-level (as of this spec) |
|---|---|---|
| Fetch | Pull the filing PDF | `ingestion/fetch_filing_pdf.py: fetch_filing_pdf()` — downloads to `data/raw_pdf/` |
| Chunk | Split into retrievable passages, two competing strategies | `ingestion/chunk_filings_basic.py` (PyPDFLoader + regex, `chunk_size=800`, `chunk_overlap=150` tokens) vs. `ingestion/chunk_filings_advanced.py` (docling `HybridChunker`, table-aware) |
| Embed | Turn chunks into vectors for both semantic and keyword search | `core/embeddings.py: embed_dense()` — OpenAI `text-embedding-3-small`, 1536-dim; sparse side via Qdrant's built-in BM25 |
| Index | Store vectors + metadata for retrieval | `ingestion/index_to_qdrant.py: index_all_filings()` — one Qdrant collection per pipeline (`sec_filings_basic`, `sec_filings_advanced`) |
| Retrieve | Combine dense + sparse search, five interchangeable strategies | `core/retrieval.py: hybrid_search()` (RRF fusion, `top_k=8`) plus `core/retrieval_strategies.py`: baseline, query_rewrite, hyde, multi_query (RRF over variants), rerank (cross-encoder `Xenova/ms-marco-MiniLM-L-6-v2`) |
| Generate | Produce a grounded, cited answer | `core/rag.py: answer_question()` — builds a prompt constrained to retrieved chunks, calls `core/llm.py: complete()` (OpenAI `gpt-4o`), parses citations |

This table is static narrative text written now to match current code — not read live from source at render time (keeps the tab fast and immune to mid-edit source state during a talk).

**File map:** a small closing table, file -> one-line responsibility, as a quick-reference the presenter can point to (covers `core/*.py` and `ingestion/*.py`).

## Visual style

- `.streamlit/config.toml`: defined theme — dark-friendly base, single teal/blue accent color, consistent font, tuned for projector contrast.
- One block of custom CSS injected via `st.markdown(..., unsafe_allow_html=True)` in `app.py`: card-style bordered section containers, consistent spacing, styled header + subtitle.
- All three tabs keep their existing `st.divider()`-separated internal sections, now wrapped in the shared card styling so the whole app reads as one visual system.

## Non-goals

- No change to retrieval/generation logic, ingestion pipelines, or test suite — this is UI-only.
- No live source-reading in the Architecture tab (explicitly declined in favor of static, always-fast narrative).
- No new Streamlit pages/routing beyond the three tabs.

## Testing / verification

No automated tests apply (no `core`/`ingestion` behavior change). Verification is manual: run `streamlit run ui/app.py`, click through all three tabs, confirm Q&A and Pipeline Comparison behave exactly as before, confirm the Architecture diagram renders and expanders open.
