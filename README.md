# FinTech RAG

Ask questions about a company's annual report (SEC Form 10-K) and get answers that **cite the exact passages they came from**. If the filing doesn't contain the answer, the app says so instead of guessing.

**The problem it solves:** a 10-K is long (Apple's is 121 pages) and dense with tables. Asking a chatbot directly can return made-up numbers with no source. This project is a retrieval-augmented generation (RAG) system: it finds the few relevant passages in the filing, has an LLM answer **only** from them, and shows which passages it used.

- Corpus: Apple's 10-K (`ingestion/fetch_filings.py: COMPANIES`).
- Models: **OpenAI** or **Google Gemini**, chosen in `.env`, no code changes.
- Vector database: **Qdrant**, either embedded in Python (no Docker) or as a Docker server.
- Ways to use it: a Streamlit app, five step-by-step Jupyter notebooks (from scratch to agentic RAG), a CLI evaluation, or plain Python.

This README is the single reference for setup, configuration, commands and code layout.

## Contents

1. [Quickstart](#quickstart)
2. [The app](#the-app)
3. [The notebooks](#the-notebooks)
4. [Configuration](#configuration)
5. [How it works](#how-it-works)
   - [Agentic RAG](#agentic-rag)
6. [Command reference](#command-reference)
7. [Evaluation](#evaluation)
8. [Project structure](#project-structure)
9. [Using it from Python](#using-it-from-python)
10. [Adding a company](#adding-a-company)
11. [Troubleshooting](#troubleshooting)
12. [Sample questions](#sample-questions)
13. [Tests](#tests)

---

## Quickstart

About 15 minutes. Run every command from a terminal inside the project folder.

### Step 0: Have these ready

| You need | Check | Get it |
|---|---|---|
| **Python 3.11+** | `python3 --version` | [python.org/downloads](https://www.python.org/downloads/) |
| **uv** (recommended; plain `pip` also works) | `uv --version` | `curl -LsSf https://astral.sh/uv/install.sh \| sh` ([docs](https://docs.astral.sh/uv/)) |
| **Git** | `git --version` | [git-scm.com](https://git-scm.com/downloads) |
| **An API key**: OpenAI or Gemini | | OpenAI: [platform.openai.com/api-keys](https://platform.openai.com/api-keys) · Gemini: [aistudio.google.com/apikey](https://aistudio.google.com/apikey) |
| Docker Desktop (**optional**) | `docker --version` | [docker.com/products/docker-desktop](https://www.docker.com/products/docker-desktop/), only for the [Docker database mode](#database-embedded-or-docker) |

### Step 1: Get the code

```bash
git clone https://github.com/iam-suraj-kumar/fintech-rag.git
cd fintech-rag
```

### Step 2: Install the Python packages

With **uv** (creates `.venv/` from `uv.lock`; `--extra dev` adds pytest, Jupyter and matplotlib):

```bash
uv sync --extra dev
source .venv/bin/activate        # Windows: .venv\Scripts\activate
```

Or with **pip**:

```bash
python3 -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -e ".[dev]"
```

Every later command assumes the virtual environment is active (your prompt starts with `(.venv)`). In a new terminal, run the `source` line again.

### Step 3: Create `.env`

```bash
cp .env.example .env
```

Open `.env` and fill in **one** model setup:

**Option A: OpenAI (default)**
```
OPENAI_API_KEY=sk-...
```

**Option B: Google Gemini**
```
GEMINI_API_KEY=your-ai-studio-key
LLM_PROVIDER=gemini
EMBEDDING_PROVIDER=gemini
```

You can mix providers (e.g. Gemini answers, OpenAI embeddings); see [Configuration](#configuration). `.env` is git-ignored; never commit it.

### Step 4: Choose where the database runs

**Embedded (recommended to start, no Docker):** add this line to `.env`:

```
QDRANT_PATH=data/qdrant_local
```

**Or Docker:** leave `QDRANT_PATH` empty, open Docker Desktop, then:

```bash
docker compose up -d
```

The trade-offs are in [Database: embedded or Docker](#database-embedded-or-docker).

### Step 5: Build the database (once)

```bash
python -m ingestion.prepare
```

This downloads the 10-K PDF, chunks it two ways, embeds every chunk and indexes everything in Qdrant. Each step is skipped if its output already exists. The first run takes a few minutes, mostly docling downloading its layout models; later runs take seconds. [Command reference](#command-reference) lists the individual steps.

### Step 6: Ask questions

```bash
streamlit run ui/app.py
```

Your browser opens at [http://localhost:8501](http://localhost:8501). In the **Q&A** tab, click a sample question or type your own. Try *What was Apple's revenue in fiscal year 2019?*: it isn't in the filing, so the app should say "not found".

Prefer to learn step by step? Use [the notebooks](#the-notebooks) instead of Steps 5 and 6.

**Next time:** activate the venv → (Docker mode only: `docker compose up -d`) → `streamlit run ui/app.py`.

---

## The app

`streamlit run ui/app.py` opens four tabs (`ui/app.py`):

| Tab | What it does | Code |
|---|---|---|
| **Q&A** | Ask a question; get the answer with expandable sources. Five sample-question buttons. Uses the advanced collection and the baseline strategy. | `ui/tabs/qa.py` |
| **Pipeline Comparison** | *Overview*: chunk counts and sizes, and the same table chunk from both chunkers. *Try It*: one question against a chosen chunker + retrieval strategy, or across all combinations. | `ui/tabs/pipeline_comparison.py` |
| **Evaluation** | Runs the golden question set for a chosen chunker, strategy and category, optionally with the LLM judge, or the full chunker × strategy matrix. Shows metrics and cost. | `ui/tabs/evaluation.py` |
| **Architecture** | The pipeline diagram, each stage explained, a file map, and production considerations. | `ui/tabs/architecture.py` |

The theme is set in `.streamlit/config.toml`. Use another port with `streamlit run ui/app.py --server.port 8502`.

---

## The notebooks

`notebooks/` teaches the project from scratch, one cell at a time, using the project's real code (`core/`, `ingestion/`), not copies of it. Each notebook opens with what you'll learn, time and cost, a flow diagram, and a table of its Parts; each Part explains why the step exists and what to look for in the output, and ends with things to try.

```bash
jupyter lab notebooks/           # run cells with Shift + Enter
```

| Notebook | What you'll learn |
|---|---|
| `00_start_here.ipynb` | The problem and the idea of RAG, a setup check, and each building block on toy examples: tokens, embeddings and cosine similarity, BM25 keyword scoring, RRF by hand, grounded prompts and citations. Needs no database |
| `01_ingest.ipynb` | Phase 1: the 10-K's Items, why and how to chunk, basic vs advanced chunks (size chart, the same table from each chunker), dense and sparse vectors, Qdrant collections and points |
| `02_answer.ipynb` | Phase 2, steps 1–8: dense search, sparse search, how much they overlap, RRF by hand vs Qdrant, choosing top-k, the grounded prompt, the LLM reply, parsed citations, and the "not found" refusal |
| `03_models_and_experiments.ipynb` | A: OpenAI vs Gemini · B: basic vs advanced chunks · C: the five retrieval strategies · D: a mini evaluation |
| `04_agentic_rag.ipynb` | [Agentic RAG](#agentic-rag): classic vs agentic, the tool / decision format / loop / guardrails, a live trace, a side-by-side comparison, and a checklist for converting a RAG pipeline into an agent |

How they behave:

- **Every notebook runs on its own, in any order.** In 02–04 the second cell calls `prepare_data()`, which builds whatever is missing; 01 builds the data step by step; 00 needs no database. Each section of notebook 03 (A–D) also runs alone after the two setup cells.
- **Diagrams** are Mermaid blocks, drawn by JupyterLab 4.1+ (installed by the dev extra).
- **Database:** cell 1 has `USE_EMBEDDED_QDRANT = False`. Set it to `True` to use `data/qdrant_local/` without Docker, then **Kernel → Restart**. Setting `QDRANT_PATH` in `.env` works too.
- **Slow steps are opt-in:** in notebook 01, `RECHUNK = True` re-runs docling and `REINDEX = True` re-embeds everything.
- **Gemini column in 03A** is skipped when `.env` has no `GEMINI_API_KEY`.
- **Cost** of running all five: roughly 20¢ with OpenAI (most of it in 03 and 04).
- The notebooks are generated by `scripts/make_notebooks.py`. Edit that script to change them; running it (`python scripts/make_notebooks.py notebooks`) overwrites the notebooks and clears their saved outputs.

---

## Configuration

### All settings

Every setting the code reads, all set in `.env` (loaded automatically by `core/__init__.py`):

| Setting | Default | Used for | Read in |
|---|---|---|---|
| `OPENAI_API_KEY` | (none) | OpenAI LLM and/or embeddings | `core/llm.py`, `core/embeddings.py` |
| `GEMINI_API_KEY` | (none) | Gemini LLM and/or embeddings | `core/llm.py`, `core/embeddings.py` |
| `LLM_PROVIDER` | `openai` | `openai` or `gemini` | `core/llm.py` |
| `LLM_MODEL` | `gpt-4o` / `gemini-3.5-flash` | override the answer model | `core/llm.py` |
| `EMBEDDING_PROVIDER` | `openai` | `openai` or `gemini` | `core/embeddings.py` |
| `EMBEDDING_MODEL` | `text-embedding-3-small` / `gemini-embedding-001` | override the embedding model | `core/embeddings.py` |
| `QDRANT_PATH` | (empty) | set a folder to use embedded Qdrant; overrides `QDRANT_URL` | `core/clients.py` |
| `QDRANT_URL` | `http://localhost:6333` | the Docker Qdrant server | `core/clients.py` |

Only the key for the provider(s) you chose is needed. `.env.example` lists all of them.

### Choosing a model

- **Changing the embedding provider or model means re-indexing:** `python -m ingestion.prepare --reindex` (or `REINDEX = True` in notebook 01). Vectors from different models can't be mixed.
- Both embedding providers produce **1536-dimension** vectors, so the Qdrant schema is the same either way.
- Changing the LLM never needs re-indexing.
- Gemini model names change often. On `404 NOT_FOUND` ("no longer available") or `503 UNAVAILABLE` ("high demand"), set `LLM_MODEL` to another model your key lists in [AI Studio](https://aistudio.google.com/). Gemini calls time out after 60 seconds instead of hanging, and run with thinking off so short replies aren't empty.
- Cost figures are **estimates** from the price table in `core/llm.py`. Unlisted models use a fallback price (gpt-4o for OpenAI, gemini-2.5-flash for Gemini).
- Rate-limit and server errors are retried with backoff for both providers (`core/retry.py`).

### Database: embedded or Docker

| | Embedded (`QDRANT_PATH`) | Docker server (`QDRANT_URL`) |
|---|---|---|
| Setup | one line in `.env` | Docker Desktop + `docker compose up -d` (`docker-compose.yml`) |
| Data lives in | `data/qdrant_local/` | Docker volume `qdrant_data` |
| Web dashboard | none | [localhost:6333/dashboard](http://localhost:6333/dashboard) |
| App and a notebook at the same time | ❌ one program at a time (the folder is locked) | ✅ |
| Docker-only tests | skipped | run |

- `QDRANT_PATH` wins over `QDRANT_URL`. Empty it to go back to Docker.
- The two modes keep **separate data**. After switching, run `python -m ingestion.prepare` once.
- To reset embedded data, delete `data/qdrant_local/` and run `python -m ingestion.prepare`. For Docker: `docker compose down -v && docker compose up -d`, then the same command.
- Search, RRF fusion and everything else behave the same in both modes (`core/clients.py: get_qdrant_client()`).

---

## How it works

### Phase 1: Ingest (once)

1. **Fetch** the company-published 10-K PDF to `data/raw_pdf/AAPL.pdf` (`ingestion/fetch_filing_pdf.py`).
2. **Chunk** it two ways, so the app can compare them:
   - **basic**: PyPDF text, split at "Item 1A", "Item 7"… headers, then into pieces of up to 800 tokens with 150 tokens of overlap (`ingestion/chunk_filings_basic.py`) → `data/chunks_basic/AAPL.json` (168 chunks).
   - **advanced**: docling reads the page layout, keeps tables whole as markdown, and caps chunks at 800 tokens of the gpt-4o tokenizer; section headings added on top can push a few slightly past that (`ingestion/chunk_filings_advanced.py`) → `data/chunks_advanced/AAPL.json` (310 chunks).
3. **Embed + index**: every chunk gets a **dense** vector (meaning; OpenAI or Gemini) and a **sparse** BM25 vector (exact words; `Qdrant/bm25`), stored with its ticker, section and fiscal year in Qdrant. Basic chunks go to `sec_filings_basic`, advanced chunks to `sec_filings_advanced` (`ingestion/index_to_qdrant.py`, `ingestion/index_comparison_collections.py`).

`ingestion/prepare.py: prepare_data()` runs all three steps, skipping what already exists.

### Phase 2: Answer (every question)

1. The question is searched both ways at once: dense and sparse, 16 candidates each.
2. **Reciprocal Rank Fusion** (RRF) merges the two ranked lists; the **top k = 8** chunks are kept (`core/retrieval.py: hybrid_search()`).
3. The chunks are numbered in a prompt: *answer using ONLY these excerpts, then write `CITED: 0,2`* (`core/rag.py: _build_prompt()`).
4. The LLM answers (`core/llm.py: complete_with_usage()`); the `CITED:` line is parsed back into citations (`core/rag.py: _parse_response()`).
5. Nothing retrieved, or nothing cited → the answer is "I couldn't find information about this in the available filings."

`core/rag.py: answer_question()` runs steps 1–5 and returns a `RAGAnswer` with the answer, citations, token counts and cost. The Q&A tab queries `sec_filings_advanced` (`core/clients.py: COLLECTION_NAME`).

### Retrieval strategies

`answer_question(..., strategy=...)` accepts five strategies (`core/retrieval_strategies.py`). They're available in the Pipeline Comparison and Evaluation tabs, `eval/run_eval.py --strategy`, and notebook 03C:

| Strategy | What it does |
|---|---|
| `baseline` | `hybrid_search()` as above (the default) |
| `query_rewrite` | the LLM rewrites the question into a search query first |
| `hyde` | the LLM writes a hypothetical 10-K passage, which is searched for instead |
| `multi_query` | the LLM writes question variants; their results are fused with RRF |
| `rerank` | fetch 3 × k candidates, re-order them with the cross-encoder `Xenova/ms-marco-MiniLM-L-6-v2` (downloaded on first use) |

### Agentic RAG

`core/agentic_rag.py: answer_question_agentic()` is an agent version of `answer_question()`. Instead of one search with the user's words, the LLM decides what to search for and how often:

1. Each turn the LLM gets its instructions, the transcript so far, and how many searches are left, and replies with one JSON object: `{"action": "search", "query": ...}` or `{"action": "answer", "answer": ..., "cited": [ids]}`.
2. A search runs `hybrid_search(query)` (5 chunks) and appends the results to the transcript.
3. An answer keeps only cited ids the agent actually retrieved; no valid citation means the same "not found" message as classic RAG.

Guardrails: at most 4 searches (`max_searches`), repeated searches are refused, invalid JSON is retried within a hard turn limit. Plain JSON instead of native tool calling keeps it working with both OpenAI and Gemini. It returns an `AgentRun` with the `RAGAnswer` and the list of searches (`steps`).

When to use it: it helps on multi-part or loosely-worded questions. Each search adds an LLM call, so cost and latency grow with the number of searches (about 2× classic RAG on a two-part question). It's available from Python and in `notebooks/04_agentic_rag.ipynb`, which also has a checklist for converting a RAG pipeline into an agent. The app and the evaluation use classic RAG.

---

## Command reference

Run from the project folder with the venv active.

| Command | What it does |
|---|---|
| `python -m ingestion.prepare` | Fetch, chunk and index, skipping what exists |
| `python -m ingestion.prepare --reindex` | Rebuild both collections (after changing the embedding provider or model) |
| `python -m ingestion.prepare --rechunk` | Re-run both chunkers (docling is slow), then rebuild both collections |
| `python -m ingestion.fetch_filing_pdf` | Step 1 only: download the PDF to `data/raw_pdf/` |
| `python -m ingestion.chunk_filings_basic` | Step 2 only: basic chunks to `data/chunks_basic/` |
| `python -m ingestion.chunk_filings_advanced` | Step 2 only: advanced chunks to `data/chunks_advanced/` |
| `python -m ingestion.index_comparison_collections` | Step 3 only: embed and index both chunk sets (adds to existing collections; use `prepare --reindex` for a clean rebuild) |
| `streamlit run ui/app.py` | [The app](#the-app) |
| `jupyter lab notebooks/` | [The notebooks](#the-notebooks) |
| `python scripts/smoke_test.py` | Three questions end to end, including one that should be refused |
| `python -m eval.run_eval` | [Evaluation](#evaluation) against the golden set |
| `python scripts/make_notebooks.py notebooks` | Regenerate the notebooks (clears their outputs) |
| `pytest` | [Tests](#tests) |
| `docker compose up -d` / `docker compose down` | Start / stop the Docker Qdrant server |

---

## Evaluation

`eval/golden_dataset.py` holds 12 hand-labeled questions: exact facts (3), segment numbers (3), conceptual (3), cross-section (1), and "not found" traps (2).

| Metric | Meaning | Needs an LLM? |
|---|---|---|
| Hit rate | Did an expected section appear in the top-k? | no |
| MRR | `1 / rank` of the first relevant chunk (1.0 = ranked first) | no |
| Recall | Share of a question's expected sections found in the top-k | no |
| Faithfulness | Judge score 1–5: is the answer fully supported by the retrieved chunks? | yes (`eval/judge.py`) |
| Correctness | Judge score 1–5: does it match the reference answer? | yes |
| Correct refusals | For "not found" questions: did the app decline? | no |

Run it from the command line:

```bash
python -m eval.run_eval --pipeline advanced --no-judge
```

| Flag | Default | Meaning |
|---|---|---|
| `--pipeline basic\|advanced` | **`basic`** | which chunk collection to score |
| `--strategy` | `baseline` | one of the [five strategies](#retrieval-strategies) |
| `--top-k` | `8` | chunks retrieved per question |
| `--category` | all | e.g. `exact_term`, `segment_numerical`, `semantic`, `cross_section`, `not_found` |
| `--no-judge` | judge on | skip the LLM judge (cheaper and faster) |
| `--collection` | (none) | raw Qdrant collection name, overrides `--pipeline` |
| `--output FILE` | (none) | write per-question results as JSON |

The Evaluation tab runs the same harness with menus, plus a full chunker × strategy matrix. Retrieval metrics come from `eval/retrieval_metrics.py`. Reported cost under-counts `query_rewrite`, `hyde`, `multi_query` and `rerank`: only their final answer call is tracked, not their extra LLM step.

---

## Project structure

```
core/                         shared pipeline code
  __init__.py                 loads .env on import
  clients.py                  Qdrant client (embedded via QDRANT_PATH, or server via QDRANT_URL),
                              BM25 sparse model; COLLECTION_NAME = "sec_filings_advanced"
  embeddings.py               embed_dense(): OpenAI or Gemini dense vectors, 1536 dims
  llm.py                      complete_with_usage() / complete(): OpenAI or Gemini, tokens + cost
  retry.py                    with_retry(): backoff on rate limits / server errors, both providers
  retrieval.py                hybrid_search(): dense + sparse, fused with RRF
  retrieval_strategies.py     STRATEGIES: baseline, query_rewrite, hyde, multi_query, rerank
  rag.py                      answer_question(): retrieve → grounded prompt → LLM → citations
  agentic_rag.py              answer_question_agentic(): the LLM searches as needed, then answers
  models.py                   FilingChunk, RetrievedChunk, Citation, RAGAnswer
ingestion/                    builds the database
  fetch_filings.py            COMPANIES (ticker, CIK, name) and the HTTP user agent
  fetch_filing_pdf.py         PDF_URLS; downloads → data/raw_pdf/{TICKER}.pdf
  chunk_filings_basic.py      PyPDF + Item regex + 800-token splitter → data/chunks_basic/
  chunk_filings_advanced.py   docling HybridChunker, tables as markdown → data/chunks_advanced/
  text_utils.py               token_length(), Item-header section splitting
  index_to_qdrant.py          creates a collection, embeds and upserts chunks
  index_comparison_collections.py   indexes both chunk sets into their collections
  prepare.py                  prepare_data(): all of the above, skipping what exists
ui/                           Streamlit app (see "The app")
  app.py                      entry point and tabs
  tabs/qa.py, tabs/pipeline_comparison.py, tabs/evaluation.py, tabs/architecture.py
  markdown_table.py           table rendering helper
eval/                         evaluation harness (see "Evaluation")
  golden_dataset.py, retrieval_metrics.py, judge.py, run_eval.py
notebooks/                    00_start_here, 01_ingest, 02_answer, 03_models_and_experiments, 04_agentic_rag
scripts/
  smoke_test.py               quick end-to-end check
  make_notebooks.py           generates notebooks/
tests/                        pytest suite (see "Tests")
docs/                         talk materials: FinTech_RAG_talk.pptx (slides with click animations),
                              FinTech_RAG_onepager.pdf (student handout)
data/                         generated, git-ignored: raw_pdf/, chunks_basic/, chunks_advanced/,
                              qdrant_local/ (embedded mode only)
.env.example                  every setting, copied to .env in Step 3
docker-compose.yml            Qdrant server for Docker mode
.streamlit/config.toml        app theme
pyproject.toml, uv.lock       dependencies (extra "dev": pytest, jupyter, matplotlib)
```

Logging: ingestion scripts, `core/retrieval.py` and `core/rag.py` log chunk counts, upserts, retrieval hits and token usage at INFO level, visible in the terminal running Streamlit or a script.

---

## Using it from Python

```python
from core.rag import answer_question

result = answer_question("How does Apple describe supply chain risk?")
print(result.answer)
for citation in result.citations:
    print(citation.company_name, citation.section, citation.fiscal_year)
print(result.input_tokens, result.output_tokens, result.cost_usd)

# Other collection or strategy:
answer_question("...", collection_name="sec_filings_basic", strategy="rerank")

# Agentic RAG: the LLM decides what to search for
from core.agentic_rag import answer_question_agentic

run = answer_question_agentic("Compare Products and Services gross margin in 2024")
print([step.query for step in run.steps])   # the searches it chose
print(run.result.answer, run.result.citations)
```

For the individual steps (`hybrid_search`, `_build_prompt`, `complete_with_usage`, `_parse_response`), see `notebooks/02_answer.ipynb`.

---

## Adding a company

1. Add it to `COMPANIES` in `ingestion/fetch_filings.py` and its 10-K PDF link to `PDF_URLS` in `ingestion/fetch_filing_pdf.py`.
2. Run `python -m ingestion.prepare --rechunk`.

Things that stay Apple-specific until you change them:
- Chunks are tagged `fiscal_year="2024"` (the default of `chunk_all_filings_basic()` / `chunk_all_filings_advanced()`).
- The sidebar list and sample questions are in `ui/tabs/qa.py`.
- The Pipeline Comparison overview reads `AAPL.json` (`ui/tabs/pipeline_comparison.py`).
- The golden set (`eval/golden_dataset.py`) only has Apple questions.

`PDF_URLS` points at Apple's *latest* 10-K PDF, which changes when Apple files a new one. The `fiscal_year` tag doesn't change with it, so check it against the retrieved text before trusting a "fiscal 2024" answer.

---

## Troubleshooting

| You see | Fix |
|---|---|
| `Connection refused` / `localhost:6333` | Docker mode, but Qdrant isn't running: open Docker Desktop, then `docker compose up -d`. Or switch to embedded mode (`QDRANT_PATH`) |
| `Cannot connect to the Docker daemon` | Docker Desktop isn't open yet: start it and wait until it's running |
| `Storage folder ... is already accessed by another instance` | Embedded mode allows one program at a time: stop the app, or **Kernel → Shut Down** the other notebook |
| `KeyError: 'OPENAI_API_KEY'` or `'GEMINI_API_KEY'` | The key for the provider you chose is missing from `.env` |
| `ModuleNotFoundError` | The venv isn't active: `source .venv/bin/activate` |
| `Collection ... not found` | The database isn't built yet: `python -m ingestion.prepare` |
| `Vector dimension error`, or bad answers after switching embeddings | Re-index: `python -m ingestion.prepare --reindex` |
| Gemini `404 NOT_FOUND` / `503 UNAVAILABLE` | Set another `LLM_MODEL` (see [Choosing a model](#choosing-a-model)) |
| `Qdrant client version ... is incompatible with server version` warning | Harmless: the Python client is newer than the Docker image in `docker-compose.yml` |
| Port 8501 already in use | `streamlit run ui/app.py --server.port 8502` |

---

## Sample questions

The Q&A tab's buttons (`ui/tabs/qa.py: SAMPLE_QUESTIONS`) are taken from this list.

**Exact facts**
- What was Apple's total net sales for fiscal year 2024?
- What was Apple's gross margin percentage in 2024 vs. 2023?
- How much did Apple spend on R&D in fiscal 2024, and what percent of net sales is that?
- What are the interest rates and maturity dates on Apple's outstanding notes?

**Segment numbers**
- Which geographic segment had the largest revenue decline in 2024, and why?
- How did Services net sales compare to iPhone net sales in 2024?
- Why did Greater China net sales decrease in fiscal 2024?
- Compare Products gross margin to Services gross margin: which is higher and why?

**Conceptual**
- How does Apple describe the risks of "Apple Intelligence" and generative AI in its filing?
- What cybersecurity risks does Apple disclose in Item 1C?
- What does Apple say about risks from foreign currency fluctuation?
- What legal proceedings is Apple currently involved in?

**Cross-section**
- What new products did Apple launch in fiscal 2024, and how did they affect segment performance?
- What macroeconomic or geopolitical risks does Apple flag that could affect its Greater China revenue?

**"Not found" traps** (should be refused)
- What was Apple's revenue in fiscal year 2019?
- What is Apple's stock buyback plan for fiscal 2026?

---

## Tests

```bash
pytest
```

Tests use fakes and need no API key. The two tests in `tests/test_index_to_qdrant.py` need the Docker Qdrant server and are skipped when it isn't running.
