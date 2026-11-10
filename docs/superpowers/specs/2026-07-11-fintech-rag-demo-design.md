# FinTech RAG Demo — Design

## Purpose

A code-based demo for a fintech startup audience, showing a progression of RAG techniques from basic to advanced, built over real SEC filing data:

1. **RAG** (basic hybrid-search RAG)
2. **Agentic RAG** (multi-step, tool-using retrieval agent)
3. **Harness** (planner, tracing, evaluation — wraps any agent)
4. **Multi-agent RAG** (specialist agents orchestrated together, running inside the harness)

Each phase gets its own spec + implementation plan, landing incrementally in one growing repo (not four throwaway projects). This document specifies **Phase 1** in full and records the **agreed roadmap** for Phases 2-4 so later specs stay consistent with this foundation.

## Stack (applies to all phases)

- **Language:** Python
- **LLM:** Anthropic Claude (Sonnet 5 for reasoning/agents, Haiku 4.5 for cheap/fast calls where useful)
- **Embeddings:** Voyage AI, `voyage-finance-2` (finance-domain-specific embedding model)
- **Agent framework:** LangGraph (from Phase 2 onward)
- **Vector store:** Qdrant, self-hosted via Docker Compose (native hybrid dense+sparse search, free/open-source)
- **Dataset:** SEC EDGAR filings, fixed local corpus (not live-fetch) for demo reliability

## Roadmap (Phases 2-4, for context — specced separately later)

- **Phase 2 (Agentic RAG):** New `agentic/` module. A LangGraph agent uses retrieval as a *tool*, decomposing questions into sub-questions and re-querying when results are weak, deciding when it has enough to answer. Reuses `core/retrieval.py` and `core/models.py`. New entrypoint: `agentic/agent.py: run_agentic_query()`.
- **Phase 3 (Harness):** New `harness/` module providing planner, tracing, and evaluation, generic enough to wrap either Phase 2's agent or Phase 4's multi-agent system. This is where "prod-ready" observability and quality scoring live.
- **Phase 4 (Multi-agent RAG):** New `multi_agent/` module — a router agent plus specialist sub-agents (e.g. per-company or per-filing-section), orchestrated via LangGraph, running inside the Phase 3 harness.
- **UI:** Extended over time with a mode selector so one Streamlit app can demo all four phases side by side.

## Phase 1: Basic Hybrid RAG

### Scope

A working, hybrid-search RAG system over real SEC 10-K filings, demoable via a minimal web UI, with core logic fully decoupled from the UI behind a single public function.

**In scope:** hybrid (dense + sparse) retrieval, grounded answer generation with citations, a minimal web UI.

**Out of scope (deferred to later phases):** agentic multi-step reasoning, tool use, planning/tracing/evaluation harness, multi-agent orchestration, memory, MCP, skills.

### Corpus

Latest 10-K filing for 5 companies, fetched once from SEC EDGAR and cached locally:

- Apple (AAPL)
- JPMorgan Chase (JPM)
- Visa (V)
- Bank of America (BAC)
- Wells Fargo (WFC)

Mix of tech and banking so both semantic questions ("how does the company think about AI risk") and exact-term questions ("net interest margin", "Basel III") exercise hybrid search meaningfully.

### Ingestion Pipeline (offline, run once)

1. **Fetch** (`ingestion/fetch_filings.py`): For each ticker, look up CIK, call SEC EDGAR's `submissions` JSON API (`https://data.sec.gov/submissions/CIK##########.json`) to find the latest 10-K filing, download the filing HTML. Requires a polite `User-Agent` header with contact info per SEC's access rules; EDGAR rate-limits to 10 req/sec (a non-issue at this volume). Cache raw HTML to `data/raw/<ticker>.html`.
2. **Chunk** (`ingestion/chunk_filings.py`): Strip HTML to clean text with BeautifulSoup, preserving section headers (Item 1, Item 1A Risk Factors, Item 7 MD&A, Item 8 Financial Statements, etc.). Split on Item boundaries first, then recursively split long sections into ~800-token chunks with ~150-token overlap (LangChain's `RecursiveCharacterTextSplitter`). Tag each chunk with metadata: `{ticker, company_name, filing_type, section, fiscal_year}`. Cache to `data/chunks/<ticker>.json`.
3. **Index** (`ingestion/index_to_qdrant.py`): Embed each chunk's text with Voyage `voyage-finance-2` (dense vector) and Qdrant's built-in FastEmbed sparse model `Qdrant/bm25` (sparse vector). Upsert into a single Qdrant collection `sec_filings`, storing both vectors plus the metadata payload.

Ingestion is a one-time/offline process, not part of the live query path. Re-running it is idempotent (cached raw/chunk files are reused unless deleted).

### Storage & Hybrid Retrieval

- Qdrant runs locally via Docker Compose (single service, persistent volume, no external account needed).
- Collection `sec_filings` holds, per point: dense vector (`voyage-finance-2`), sparse vector (`Qdrant/bm25`), and metadata payload from ingestion.
- Query time: embed the question with `voyage-finance-2`, run Qdrant's native hybrid query (dense + sparse in one request), fused via **Reciprocal Rank Fusion (RRF)** — Qdrant's built-in fusion method. Return top-8 chunks.
- This is implemented in `core/retrieval.py` as a single function, e.g. `hybrid_search(query: str, top_k: int = 8) -> list[RetrievedChunk]`.

### Core RAG Library

Single public entrypoint — the only thing any caller (UI or otherwise) depends on:

```python
# core/rag.py
def answer_question(question: str) -> RAGAnswer:
    """Retrieve relevant filing chunks via hybrid search and generate a grounded, cited answer."""
```

```python
# core/models.py
@dataclass
class Citation:
    ticker: str
    company_name: str
    section: str
    fiscal_year: str
    text_snippet: str  # the actual chunk text used

@dataclass
class RAGAnswer:
    question: str
    answer: str
    citations: list[Citation]
```

Internals of `answer_question`:

1. Call `core.retrieval.hybrid_search(question)` → top-8 chunks.
2. Build a prompt for Claude (Sonnet 5) instructing it to answer **only from the provided chunks**, and to reference which chunk(s) it used (by index) per claim.
3. Parse the model's cited chunk indices back into `Citation` objects (ticker, section, snippet).
4. If no retrieved chunk clears a minimum relevance score, return an `RAGAnswer` whose `answer` explicitly states the information wasn't found in the available filings, rather than letting the model guess ungrounded.

No module outside `core/` imports Qdrant, Voyage, or Anthropic clients directly — everything funnels through `answer_question`.

### Web UI

- Minimal Streamlit app (`ui/app.py`): text input for the question, submit button, answer display, and an expandable "Sources" section listing each `Citation` (company, section, fiscal year, snippet).
- Imports only `core.rag.answer_question` and `core.models` — no direct dependency on Qdrant, Voyage, or LangGraph.
- Sidebar lists the 5 companies in the demo corpus so the presenter/audience knows what's queryable.

### Testing Strategy

- **`tests/test_chunking.py`** — section-aware splitter produces correctly-tagged chunks from a small synthetic fixture HTML file (2 fake "Item" sections), verifying section boundaries and overlap.
- **`tests/test_retrieval.py`** — against a small seeded Qdrant collection (~10 known fixture chunks), verify hybrid query returns the expected chunk for both a keyword-heavy query and a semantic-only query, demonstrating hybrid search outperforms either alone.
- **`tests/test_rag.py`** — `answer_question` with a mocked Qdrant client (fixed chunks) and a mocked Claude call, verifying citation-parsing logic and the "not found" fallback path. No real API calls — fast and free to run in CI.
- **`scripts/smoke_test.py`** — manual script running a few real questions against the real pipeline (real Qdrant, real Claude/Voyage calls) for pre-demo sanity checking. Not part of automated test suite.

### Project Structure

```
fintech-rag/
├── docker-compose.yml          # Qdrant service
├── .env.example                 # ANTHROPIC_API_KEY, VOYAGE_API_KEY
├── pyproject.toml
├── data/
│   ├── raw/                     # cached raw filing text per ticker
│   └── chunks/                  # cached chunked+tagged output (JSON)
├── ingestion/
│   ├── fetch_filings.py         # EDGAR fetch -> data/raw/
│   ├── chunk_filings.py         # section-aware split -> data/chunks/
│   └── index_to_qdrant.py       # embed + upsert dense+sparse vectors
├── core/
│   ├── models.py                # Citation, RAGAnswer dataclasses
│   ├── retrieval.py              # Qdrant hybrid query wrapper
│   └── rag.py                    # answer_question() -- the one public entrypoint
├── ui/
│   └── app.py                    # Streamlit app, imports only core.rag/core.models
├── scripts/
│   └── smoke_test.py             # manual pre-demo sanity check
└── tests/
    ├── test_chunking.py
    ├── test_retrieval.py
    └── test_rag.py
```

## Open Questions / Risks

- SEC EDGAR HTML structure varies slightly across filers; the section-aware splitter must tolerate missing/renamed Item headers gracefully (fall back to plain recursive splitting if Item boundaries aren't found) rather than failing ingestion.
- Voyage AI and Anthropic API keys are required (both have free tiers sufficient for a 5-document demo corpus); no other paid services are used in Phase 1.
