# FinTech RAG Demo — Phase 1 (Basic Hybrid RAG) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** A working, hybrid-search (dense + sparse) RAG system over 5 real SEC 10-K filings, demoable via a minimal Streamlit UI, with all core logic behind one public function the UI calls.

**Architecture:** An offline ingestion pipeline (fetch → chunk → index) populates a local Qdrant collection with dense (Voyage) and sparse (BM25) vectors per filing chunk. At query time, `core.rag.answer_question()` runs hybrid retrieval (RRF-fused) and asks Claude to answer strictly from retrieved chunks, parsing out which chunks were actually used as citations. The Streamlit UI is a thin client that only calls this one function.

**Tech Stack:** Python 3.11+, Anthropic Claude (`claude-sonnet-5`), Voyage AI (`voyage-finance-2` embeddings), Qdrant (self-hosted via Docker, native hybrid search), FastEmbed (`Qdrant/bm25` sparse vectors), LangChain text splitters, BeautifulSoup, Streamlit, pytest.

## Global Constraints

- Language: Python 3.11+
- LLM: Anthropic Claude, model id `claude-sonnet-5`
- Embeddings: Voyage AI, model id `voyage-finance-2` (1024-dim dense vectors)
- Vector store: Qdrant, self-hosted via Docker Compose, no external account
- Sparse/keyword vectors: FastEmbed model `Qdrant/bm25`, fused with dense via Qdrant's built-in RRF
- Dataset: fixed local corpus only — latest 10-K for AAPL, JPM, V, BAC, WFC — no live-fetch pipeline exposed at query time
- Ingestion (fetch/chunk/index) is offline/one-time; caches to `data/raw/` and `data/chunks/` and is idempotent
- The UI (`ui/app.py`) imports only `core.rag.answer_question` and `core.models` — never Qdrant/Voyage/Anthropic clients directly
- LangGraph is reserved for Phase 2+; Phase 1 has no agent loop, single-shot retrieval only
- No automated test may make a real paid API call except `tests/test_retrieval.py`, which is skipped automatically when `VOYAGE_API_KEY` is unset

---

### Task 1: Project Scaffolding & Qdrant Infra

**Files:**
- Create: `pyproject.toml`
- Create: `docker-compose.yml`
- Create: `.env.example`
- Create: `core/__init__.py` (empty)
- Create: `ingestion/__init__.py` (empty)
- Create: `tests/__init__.py` (empty)

**Interfaces:**
- Consumes: nothing (first task)
- Produces: an installable `fintech-rag` package exposing `core` and `ingestion` as importable packages; a running local Qdrant instance at `http://localhost:6333`

- [ ] **Step 1: Create `pyproject.toml`**

```toml
[project]
name = "fintech-rag"
version = "0.1.0"
requires-python = ">=3.11"
dependencies = [
    "anthropic>=0.40.0",
    "voyageai>=0.3.0",
    "qdrant-client>=1.12.0",
    "fastembed>=0.4.0",
    "beautifulsoup4>=4.12.0",
    "langchain-text-splitters>=0.3.0",
    "tiktoken>=0.8.0",
    "requests>=2.32.0",
    "streamlit>=1.40.0",
]

[project.optional-dependencies]
dev = [
    "pytest>=8.3.0",
]

[build-system]
requires = ["setuptools>=68.0"]
build-backend = "setuptools.build_meta"

[tool.setuptools.packages.find]
include = ["core*", "ingestion*"]
```

- [ ] **Step 2: Create `docker-compose.yml`**

```yaml
services:
  qdrant:
    image: qdrant/qdrant:v1.12.4
    ports:
      - "6333:6333"
    volumes:
      - qdrant_data:/qdrant/storage

volumes:
  qdrant_data:
```

- [ ] **Step 3: Create `.env.example`**

```
ANTHROPIC_API_KEY=
VOYAGE_API_KEY=
QDRANT_URL=http://localhost:6333
```

- [ ] **Step 4: Create empty package init files**

```bash
touch core/__init__.py ingestion/__init__.py tests/__init__.py
```

- [ ] **Step 5: Start Qdrant and verify it's reachable**

```bash
docker compose up -d
sleep 3
curl -s http://localhost:6333/collections
```

Expected: JSON output like `{"result":{"collections":[]},"status":"ok","time":...}`

- [ ] **Step 6: Install the project in editable mode and verify packages import**

```bash
pip install -e ".[dev]"
python -c "import core, ingestion; print('ok')"
```

Expected: `ok` printed, no `ModuleNotFoundError`

- [ ] **Step 7: Commit**

```bash
git add pyproject.toml docker-compose.yml .env.example core/__init__.py ingestion/__init__.py tests/__init__.py
git commit -m "chore: scaffold project, add Qdrant docker-compose"
```

---

### Task 2: Shared Data Models & Clients

**Files:**
- Create: `core/models.py`
- Create: `core/clients.py`
- Test: `tests/test_models.py`
- Test: `tests/test_clients.py`

**Interfaces:**
- Consumes: nothing beyond Task 1's package scaffolding
- Produces:
  - `core.models.FilingChunk(ticker, company_name, filing_type, section, fiscal_year, text)`
  - `core.models.RetrievedChunk(ticker, company_name, filing_type, section, fiscal_year, text, score)`
  - `core.models.Citation(ticker, company_name, section, fiscal_year, text_snippet)`
  - `core.models.RAGAnswer(question, answer, citations)`
  - `core.clients.get_qdrant_client() -> QdrantClient`
  - `core.clients.get_voyage_client() -> voyageai.Client`
  - `core.clients.get_sparse_model() -> fastembed.SparseTextEmbedding`
  - `core.clients.COLLECTION_NAME`, `core.clients.DENSE_MODEL`, `core.clients.SPARSE_MODEL`, `core.clients.QDRANT_URL` (module constants)

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_models.py
from core.models import FilingChunk, RetrievedChunk, Citation, RAGAnswer


def test_filing_chunk_holds_expected_fields():
    chunk = FilingChunk(
        ticker="AAPL",
        company_name="Apple Inc.",
        filing_type="10-K",
        section="Item 7. Management's Discussion and Analysis",
        fiscal_year="2024",
        text="Revenue grew...",
    )
    assert chunk.ticker == "AAPL"
    assert chunk.section.startswith("Item 7")


def test_retrieved_chunk_holds_score():
    chunk = RetrievedChunk(
        ticker="JPM",
        company_name="JPMorgan Chase & Co.",
        filing_type="10-K",
        section="Item 7",
        fiscal_year="2024",
        text="Net interest margin was 2.7%.",
        score=0.02,
    )
    assert chunk.score == 0.02


def test_rag_answer_holds_citations():
    citation = Citation(
        ticker="V",
        company_name="Visa Inc.",
        section="Item 8",
        fiscal_year="2024",
        text_snippet="Net revenue was...",
    )
    answer = RAGAnswer(question="q", answer="a", citations=[citation])
    assert answer.citations[0].ticker == "V"
```

```python
# tests/test_clients.py
from unittest.mock import patch

import core.clients as mod


def test_get_qdrant_client_returns_same_instance_on_repeated_calls():
    mod._qdrant_client = None
    with patch.object(mod, "QdrantClient") as mock_cls:
        first = mod.get_qdrant_client()
        second = mod.get_qdrant_client()
    assert first is second
    mock_cls.assert_called_once()


def test_get_voyage_client_uses_api_key_from_env(monkeypatch):
    mod._voyage_client = None
    monkeypatch.setenv("VOYAGE_API_KEY", "test-key-123")
    with patch.object(mod, "voyageai") as mock_voyageai:
        mod.get_voyage_client()
    mock_voyageai.Client.assert_called_once_with(api_key="test-key-123")
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
pytest tests/test_models.py tests/test_clients.py -v
```

Expected: FAIL with `ModuleNotFoundError: No module named 'core.models'` (and `core.clients`)

- [ ] **Step 3: Implement `core/models.py`**

```python
from dataclasses import dataclass


@dataclass
class FilingChunk:
    """A section-tagged chunk of a filing, produced by ingestion and stored in Qdrant."""
    ticker: str
    company_name: str
    filing_type: str
    section: str
    fiscal_year: str
    text: str


@dataclass
class RetrievedChunk:
    """A chunk returned from hybrid search, with its fused relevance score."""
    ticker: str
    company_name: str
    filing_type: str
    section: str
    fiscal_year: str
    text: str
    score: float


@dataclass
class Citation:
    ticker: str
    company_name: str
    section: str
    fiscal_year: str
    text_snippet: str


@dataclass
class RAGAnswer:
    question: str
    answer: str
    citations: list[Citation]
```

- [ ] **Step 4: Implement `core/clients.py`**

```python
import os

import voyageai
from fastembed import SparseTextEmbedding
from qdrant_client import QdrantClient

QDRANT_URL = os.environ.get("QDRANT_URL", "http://localhost:6333")
DENSE_MODEL = "voyage-finance-2"
SPARSE_MODEL = "Qdrant/bm25"
COLLECTION_NAME = "sec_filings"

_qdrant_client: QdrantClient | None = None
_voyage_client: voyageai.Client | None = None
_sparse_model: SparseTextEmbedding | None = None


def get_qdrant_client() -> QdrantClient:
    global _qdrant_client
    if _qdrant_client is None:
        _qdrant_client = QdrantClient(url=QDRANT_URL)
    return _qdrant_client


def get_voyage_client() -> voyageai.Client:
    global _voyage_client
    if _voyage_client is None:
        _voyage_client = voyageai.Client(api_key=os.environ["VOYAGE_API_KEY"])
    return _voyage_client


def get_sparse_model() -> SparseTextEmbedding:
    global _sparse_model
    if _sparse_model is None:
        _sparse_model = SparseTextEmbedding(model_name=SPARSE_MODEL)
    return _sparse_model
```

- [ ] **Step 5: Run tests to verify they pass**

```bash
pytest tests/test_models.py tests/test_clients.py -v
```

Expected: `4 passed`

- [ ] **Step 6: Commit**

```bash
git add core/models.py core/clients.py tests/test_models.py tests/test_clients.py
git commit -m "feat: add shared data models and client factories"
```

---

### Task 3: EDGAR Filing Fetch

**Files:**
- Create: `ingestion/fetch_filings.py`
- Test: `tests/test_fetch_filings.py`

**Interfaces:**
- Consumes: nothing new
- Produces:
  - `ingestion.fetch_filings.COMPANIES: dict[str, dict]` — `{ticker: {"cik": str, "name": str}}`
  - `ingestion.fetch_filings.get_latest_10k_url(cik: str) -> str`
  - `ingestion.fetch_filings.fetch_filing_html(url: str) -> str`
  - `ingestion.fetch_filings.fetch_all_filings(force: bool = False) -> None` — writes `data/raw/<ticker>.html`
  - `ingestion.fetch_filings.RAW_DIR: Path`

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_fetch_filings.py
import json
from unittest.mock import patch, Mock

SUBMISSIONS_FIXTURE = {
    "filings": {
        "recent": {
            "form": ["8-K", "10-K", "10-Q"],
            "accessionNumber": [
                "0000320193-24-000010",
                "0000320193-24-000005",
                "0000320193-24-000001",
            ],
            "primaryDocument": ["form8k.htm", "aapl-10k.htm", "form10q.htm"],
        }
    }
}


def test_get_latest_10k_url_finds_first_10k_form():
    from ingestion.fetch_filings import get_latest_10k_url

    with patch("ingestion.fetch_filings.requests.get") as mock_get:
        mock_get.return_value = Mock(
            json=lambda: SUBMISSIONS_FIXTURE, raise_for_status=lambda: None
        )
        url = get_latest_10k_url("0000320193")

    assert url == (
        "https://www.sec.gov/Archives/edgar/data/320193/"
        "000032019324000005/aapl-10k.htm"
    )


def test_fetch_all_filings_skips_existing_cache(tmp_path, monkeypatch):
    import ingestion.fetch_filings as mod

    monkeypatch.setattr(mod, "RAW_DIR", tmp_path)
    cached = tmp_path / "AAPL.html"
    cached.write_text("cached content")

    with patch.object(mod, "get_latest_10k_url") as mock_url, patch.object(
        mod, "fetch_filing_html"
    ) as mock_fetch, patch.object(mod, "time") as mock_time:
        mock_url.return_value = "http://example.com/x.htm"
        mock_fetch.return_value = "<html>new</html>"
        mod.fetch_all_filings()

    assert cached.read_text() == "cached content"
    assert mock_fetch.call_count == len(mod.COMPANIES) - 1
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
pytest tests/test_fetch_filings.py -v
```

Expected: FAIL with `ModuleNotFoundError: No module named 'ingestion.fetch_filings'`

- [ ] **Step 3: Implement `ingestion/fetch_filings.py`**

```python
import time
from pathlib import Path

import requests

RAW_DIR = Path("data/raw")
USER_AGENT = "FinTech RAG Demo contact@example.com"

COMPANIES = {
    "AAPL": {"cik": "0000320193", "name": "Apple Inc."},
    "JPM": {"cik": "0000019617", "name": "JPMorgan Chase & Co."},
    "V": {"cik": "0001403161", "name": "Visa Inc."},
    "BAC": {"cik": "0000070858", "name": "Bank of America Corporation"},
    "WFC": {"cik": "0000072971", "name": "Wells Fargo & Company"},
}


def _headers() -> dict:
    return {"User-Agent": USER_AGENT}


def get_latest_10k_url(cik: str) -> str:
    """Return the URL of the most recent 10-K filing document for a given CIK."""
    resp = requests.get(
        f"https://data.sec.gov/submissions/CIK{cik}.json",
        headers=_headers(),
        timeout=30,
    )
    resp.raise_for_status()
    data = resp.json()
    recent = data["filings"]["recent"]
    for i, form in enumerate(recent["form"]):
        if form == "10-K":
            accession = recent["accessionNumber"][i].replace("-", "")
            primary_doc = recent["primaryDocument"][i]
            cik_no_zeros = str(int(cik))
            return (
                f"https://www.sec.gov/Archives/edgar/data/"
                f"{cik_no_zeros}/{accession}/{primary_doc}"
            )
    raise ValueError(f"No 10-K filing found for CIK {cik}")


def fetch_filing_html(url: str) -> str:
    resp = requests.get(url, headers=_headers(), timeout=30)
    resp.raise_for_status()
    return resp.text


def fetch_all_filings(force: bool = False) -> None:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    for ticker, info in COMPANIES.items():
        out_path = RAW_DIR / f"{ticker}.html"
        if out_path.exists() and not force:
            continue
        url = get_latest_10k_url(info["cik"])
        html = fetch_filing_html(url)
        out_path.write_text(html, encoding="utf-8")
        time.sleep(0.15)  # stay well under EDGAR's 10 req/sec rate limit


if __name__ == "__main__":
    fetch_all_filings()
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
pytest tests/test_fetch_filings.py -v
```

Expected: `2 passed`

- [ ] **Step 5: Commit**

```bash
git add ingestion/fetch_filings.py tests/test_fetch_filings.py
git commit -m "feat: fetch latest 10-K filings from SEC EDGAR"
```

---

### Task 4: Section-Aware Chunking

**Files:**
- Create: `ingestion/chunk_filings.py`
- Test: `tests/test_chunking.py`

**Interfaces:**
- Consumes: `ingestion.fetch_filings.COMPANIES`, `core.models.FilingChunk`
- Produces:
  - `ingestion.chunk_filings.html_to_text(html: str) -> str`
  - `ingestion.chunk_filings.split_into_sections(text: str) -> list[tuple[str, str]]`
  - `ingestion.chunk_filings.chunk_filing(ticker: str, html: str, fiscal_year: str) -> list[FilingChunk]`
  - `ingestion.chunk_filings.chunk_all_filings(fiscal_year: str = "2024") -> None` — writes `data/chunks/<ticker>.json`
  - `ingestion.chunk_filings.RAW_DIR`, `ingestion.chunk_filings.CHUNKS_DIR`

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_chunking.py
SYNTHETIC_HTML = """
<html><body>
<p>Item 1. Business</p>
<p>We design, manufacture and market synthetic widgets for testing purposes. """ + (
    "Widget details. " * 200
) + """</p>
<p>Item 7. Management's Discussion and Analysis</p>
<p>Revenue grew due to strong widget demand. """ + (
    "Financial detail. " * 200
) + """</p>
</body></html>
"""


def test_split_into_sections_finds_item_headers():
    from ingestion.chunk_filings import html_to_text, split_into_sections

    text = html_to_text(SYNTHETIC_HTML)
    sections = split_into_sections(text)
    titles = [title for title, _ in sections]

    assert any(t.startswith("Item 1.") for t in titles)
    assert any(t.startswith("Item 7.") for t in titles)


def test_split_into_sections_falls_back_when_no_items_found():
    from ingestion.chunk_filings import split_into_sections

    sections = split_into_sections("Just plain filing text with no item headers at all.")

    assert sections == [
        ("Full Document", "Just plain filing text with no item headers at all.")
    ]


def test_chunk_filing_tags_chunks_with_correct_section_and_metadata(monkeypatch):
    import ingestion.chunk_filings as mod

    monkeypatch.setitem(mod.COMPANIES, "TEST", {"cik": "0000000000", "name": "Test Corp"})

    chunks = mod.chunk_filing("TEST", SYNTHETIC_HTML, fiscal_year="2024")

    assert len(chunks) > 0
    assert all(c.ticker == "TEST" for c in chunks)
    assert all(c.fiscal_year == "2024" for c in chunks)

    item1_chunks = [c for c in chunks if c.section.startswith("Item 1.")]
    item7_chunks = [c for c in chunks if c.section.startswith("Item 7.")]
    assert len(item1_chunks) > 0
    assert len(item7_chunks) > 0
    assert "widget" in item1_chunks[0].text.lower()
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
pytest tests/test_chunking.py -v
```

Expected: FAIL with `ModuleNotFoundError: No module named 'ingestion.chunk_filings'`

- [ ] **Step 3: Implement `ingestion/chunk_filings.py`**

```python
import json
import re
from pathlib import Path

import tiktoken
from bs4 import BeautifulSoup
from langchain_text_splitters import RecursiveCharacterTextSplitter

from core.models import FilingChunk
from ingestion.fetch_filings import COMPANIES

RAW_DIR = Path("data/raw")
CHUNKS_DIR = Path("data/chunks")

ITEM_HEADER_RE = re.compile(
    r"(Item\s+\d+[A-Z]?\.?\s+[A-Za-z][^\n]{0,80})", re.IGNORECASE
)

_encoding = tiktoken.get_encoding("cl100k_base")


def _token_length(text: str) -> int:
    return len(_encoding.encode(text))


def html_to_text(html: str) -> str:
    soup = BeautifulSoup(html, "html.parser")
    return soup.get_text(separator="\n")


def split_into_sections(text: str) -> list[tuple[str, str]]:
    """Split filing text into (section_title, section_text) pairs using Item headers.

    Falls back to a single "Full Document" section if no Item headers are found,
    since SEC filers format Item headers inconsistently and ingestion must not fail.
    """
    matches = list(ITEM_HEADER_RE.finditer(text))
    if not matches:
        return [("Full Document", text)]

    sections = []
    for i, match in enumerate(matches):
        start = match.start()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        title = match.group(1).strip()
        body = text[start:end]
        sections.append((title, body))
    return sections


def chunk_filing(ticker: str, html: str, fiscal_year: str) -> list[FilingChunk]:
    company_name = COMPANIES[ticker]["name"]
    text = html_to_text(html)
    sections = split_into_sections(text)

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800, chunk_overlap=150, length_function=_token_length
    )
    chunks = []
    for section_title, section_text in sections:
        for piece in splitter.split_text(section_text):
            if not piece.strip():
                continue
            chunks.append(
                FilingChunk(
                    ticker=ticker,
                    company_name=company_name,
                    filing_type="10-K",
                    section=section_title,
                    fiscal_year=fiscal_year,
                    text=piece,
                )
            )
    return chunks


def chunk_all_filings(fiscal_year: str = "2024") -> None:
    CHUNKS_DIR.mkdir(parents=True, exist_ok=True)
    for ticker in COMPANIES:
        raw_path = RAW_DIR / f"{ticker}.html"
        html = raw_path.read_text(encoding="utf-8")
        chunks = chunk_filing(ticker, html, fiscal_year)
        out_path = CHUNKS_DIR / f"{ticker}.json"
        out_path.write_text(
            json.dumps([c.__dict__ for c in chunks], indent=2), encoding="utf-8"
        )


if __name__ == "__main__":
    chunk_all_filings()
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
pytest tests/test_chunking.py -v
```

Expected: `3 passed`

- [ ] **Step 5: Commit**

```bash
git add ingestion/chunk_filings.py tests/test_chunking.py
git commit -m "feat: add section-aware chunking of SEC filings"
```

---

### Task 5: Qdrant Indexing

**Files:**
- Create: `ingestion/index_to_qdrant.py`
- Test: `tests/test_index_to_qdrant.py`

**Interfaces:**
- Consumes: `core.clients.{get_qdrant_client, get_voyage_client, get_sparse_model, COLLECTION_NAME, DENSE_MODEL}`, `core.models.FilingChunk`, `ingestion.fetch_filings.COMPANIES`, `ingestion.chunk_filings.CHUNKS_DIR`
- Produces:
  - `ingestion.index_to_qdrant.ensure_collection() -> None`
  - `ingestion.index_to_qdrant.embed_dense(texts: list[str]) -> list[list[float]]`
  - `ingestion.index_to_qdrant.load_chunks(ticker: str) -> list[FilingChunk]`
  - `ingestion.index_to_qdrant.index_all_filings() -> None` — populates the `sec_filings` Qdrant collection
  - Re-exports `COLLECTION_NAME`, `CHUNKS_DIR`, `COMPANIES`, `get_qdrant_client`, `get_sparse_model` as module attributes for test patching

**Prerequisite:** Qdrant must be running (`docker compose up -d` from Task 1).

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_index_to_qdrant.py
import json
from unittest.mock import patch, MagicMock

import pytest
from qdrant_client import QdrantClient

from ingestion import index_to_qdrant as mod


@pytest.fixture
def qdrant_client():
    client = QdrantClient(url="http://localhost:6333")
    yield client
    if client.collection_exists("test_collection"):
        client.delete_collection("test_collection")


def test_ensure_collection_creates_once_and_is_idempotent(qdrant_client, monkeypatch):
    monkeypatch.setattr(mod, "COLLECTION_NAME", "test_collection")
    monkeypatch.setattr(mod, "get_qdrant_client", lambda: qdrant_client)

    assert not qdrant_client.collection_exists("test_collection")
    mod.ensure_collection()
    assert qdrant_client.collection_exists("test_collection")
    mod.ensure_collection()  # must not raise on second call
    assert qdrant_client.collection_exists("test_collection")


def test_index_all_filings_upserts_expected_point_count(tmp_path, monkeypatch, qdrant_client):
    monkeypatch.setattr(mod, "COLLECTION_NAME", "test_collection")
    monkeypatch.setattr(mod, "get_qdrant_client", lambda: qdrant_client)
    monkeypatch.setattr(mod, "CHUNKS_DIR", tmp_path)
    monkeypatch.setattr(mod, "COMPANIES", {"AAPL": {"cik": "0000320193", "name": "Apple Inc."}})

    chunk_data = [
        {
            "ticker": "AAPL", "company_name": "Apple Inc.", "filing_type": "10-K",
            "section": "Item 1", "fiscal_year": "2024", "text": "Apple makes iPhones.",
        },
        {
            "ticker": "AAPL", "company_name": "Apple Inc.", "filing_type": "10-K",
            "section": "Item 7", "fiscal_year": "2024", "text": "Revenue grew year over year.",
        },
    ]
    (tmp_path / "AAPL.json").write_text(json.dumps(chunk_data))

    fake_dense = [[0.1] * 1024, [0.2] * 1024]

    fake_sparse_embedding = MagicMock()
    fake_sparse_embedding.indices.tolist.return_value = [1, 2, 3]
    fake_sparse_embedding.values.tolist.return_value = [0.5, 0.3, 0.1]

    fake_sparse_model = MagicMock()
    fake_sparse_model.embed.return_value = iter([fake_sparse_embedding, fake_sparse_embedding])

    with patch.object(mod, "embed_dense", return_value=fake_dense), patch.object(
        mod, "get_sparse_model", return_value=fake_sparse_model
    ):
        mod.index_all_filings()

    count = qdrant_client.count("test_collection").count
    assert count == 2
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
pytest tests/test_index_to_qdrant.py -v
```

Expected: FAIL with `ModuleNotFoundError: No module named 'ingestion.index_to_qdrant'`

- [ ] **Step 3: Implement `ingestion/index_to_qdrant.py`**

```python
import json
from pathlib import Path

from qdrant_client import models

from core.clients import (
    COLLECTION_NAME,
    DENSE_MODEL,
    get_qdrant_client,
    get_sparse_model,
    get_voyage_client,
)
from core.models import FilingChunk
from ingestion.fetch_filings import COMPANIES

CHUNKS_DIR = Path("data/chunks")


def ensure_collection() -> None:
    client = get_qdrant_client()
    if client.collection_exists(COLLECTION_NAME):
        return
    client.create_collection(
        collection_name=COLLECTION_NAME,
        vectors_config={
            # voyage-finance-2 outputs 1024-dim embeddings
            "dense": models.VectorParams(size=1024, distance=models.Distance.COSINE),
        },
        sparse_vectors_config={
            "sparse": models.SparseVectorParams(),
        },
    )


def embed_dense(texts: list[str]) -> list[list[float]]:
    result = get_voyage_client().embed(texts, model=DENSE_MODEL, input_type="document")
    return result.embeddings


def load_chunks(ticker: str) -> list[FilingChunk]:
    path = CHUNKS_DIR / f"{ticker}.json"
    raw = json.loads(path.read_text(encoding="utf-8"))
    return [FilingChunk(**item) for item in raw]


def index_all_filings() -> None:
    client = get_qdrant_client()
    ensure_collection()
    sparse_model = get_sparse_model()

    point_id = 0
    for ticker in COMPANIES:
        chunks = load_chunks(ticker)
        texts = [c.text for c in chunks]
        dense_vectors = embed_dense(texts)
        sparse_vectors = list(sparse_model.embed(texts))

        points = []
        for chunk, dense_vec, sparse_vec in zip(chunks, dense_vectors, sparse_vectors):
            points.append(
                models.PointStruct(
                    id=point_id,
                    vector={
                        "dense": dense_vec,
                        "sparse": models.SparseVector(
                            indices=sparse_vec.indices.tolist(),
                            values=sparse_vec.values.tolist(),
                        ),
                    },
                    payload={
                        "ticker": chunk.ticker,
                        "company_name": chunk.company_name,
                        "filing_type": chunk.filing_type,
                        "section": chunk.section,
                        "fiscal_year": chunk.fiscal_year,
                        "text": chunk.text,
                    },
                )
            )
            point_id += 1
        client.upsert(collection_name=COLLECTION_NAME, points=points)


if __name__ == "__main__":
    index_all_filings()
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
pytest tests/test_index_to_qdrant.py -v
```

Expected: `2 passed` (requires Qdrant running at `localhost:6333`)

- [ ] **Step 5: Commit**

```bash
git add ingestion/index_to_qdrant.py tests/test_index_to_qdrant.py
git commit -m "feat: index filing chunks into Qdrant with dense+sparse vectors"
```

---

### Task 6: Hybrid Retrieval

**Files:**
- Create: `core/retrieval.py`
- Test: `tests/test_retrieval.py`

**Interfaces:**
- Consumes: `core.clients.{get_qdrant_client, get_voyage_client, get_sparse_model, COLLECTION_NAME, DENSE_MODEL}`, `core.models.RetrievedChunk`
- Produces:
  - `core.retrieval.embed_query_dense(query: str) -> list[float]`
  - `core.retrieval.embed_query_sparse(query: str) -> qdrant_client.models.SparseVector`
  - `core.retrieval.hybrid_search(query: str, top_k: int = 8) -> list[RetrievedChunk]`

**Prerequisite:** Qdrant running (Task 1). This test requires a real `VOYAGE_API_KEY` and is skipped automatically without one — dense embeddings must be real to meaningfully prove hybrid fusion beats keyword-only or vector-only search.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_retrieval.py
import os

import pytest
from qdrant_client import models

pytestmark = pytest.mark.skipif(
    not os.environ.get("VOYAGE_API_KEY"),
    reason="requires a real VOYAGE_API_KEY to generate dense embeddings",
)

TEST_COLLECTION = "test_retrieval_collection"

FIXTURE_CHUNKS = [
    {
        "ticker": "AAPL", "company_name": "Apple Inc.", "filing_type": "10-K",
        "section": "Item 7", "fiscal_year": "2024",
        "text": "Apple's net interest margin discussion is not applicable as Apple is not a bank.",
    },
    {
        "ticker": "JPM", "company_name": "JPMorgan Chase & Co.", "filing_type": "10-K",
        "section": "Item 7", "fiscal_year": "2024",
        "text": "Net interest margin increased to 2.7% for the fiscal year, driven by higher rates.",
    },
    {
        "ticker": "AAPL", "company_name": "Apple Inc.", "filing_type": "10-K",
        "section": "Item 1A", "fiscal_year": "2024",
        "text": (
            "Our supply chain depends heavily on manufacturing partners concentrated "
            "in a small number of regions, which exposes us to geopolitical and "
            "logistics risk."
        ),
    },
    {
        "ticker": "V", "company_name": "Visa Inc.", "filing_type": "10-K",
        "section": "Item 8", "fiscal_year": "2024",
        "text": "Visa's payment processing volume grew across all regions during the fiscal year.",
    },
]


@pytest.fixture
def seeded_collection(monkeypatch):
    import core.retrieval as mod
    from core.clients import get_qdrant_client, get_sparse_model, get_voyage_client

    client = get_qdrant_client()
    monkeypatch.setattr(mod, "COLLECTION_NAME", TEST_COLLECTION)

    if client.collection_exists(TEST_COLLECTION):
        client.delete_collection(TEST_COLLECTION)
    client.create_collection(
        collection_name=TEST_COLLECTION,
        vectors_config={"dense": models.VectorParams(size=1024, distance=models.Distance.COSINE)},
        sparse_vectors_config={"sparse": models.SparseVectorParams()},
    )

    voyage = get_voyage_client()
    sparse_model = get_sparse_model()
    texts = [c["text"] for c in FIXTURE_CHUNKS]
    dense_vectors = voyage.embed(texts, model="voyage-finance-2", input_type="document").embeddings
    sparse_vectors = list(sparse_model.embed(texts))

    points = [
        models.PointStruct(
            id=i,
            vector={
                "dense": dense_vectors[i],
                "sparse": models.SparseVector(
                    indices=sparse_vectors[i].indices.tolist(),
                    values=sparse_vectors[i].values.tolist(),
                ),
            },
            payload=chunk,
        )
        for i, chunk in enumerate(FIXTURE_CHUNKS)
    ]
    client.upsert(collection_name=TEST_COLLECTION, points=points)

    yield
    client.delete_collection(TEST_COLLECTION)


def test_hybrid_search_finds_keyword_heavy_match(seeded_collection):
    import core.retrieval as mod

    results = mod.hybrid_search("net interest margin", top_k=2)
    assert results[0].ticker == "JPM"


def test_hybrid_search_finds_semantic_only_match(seeded_collection):
    import core.retrieval as mod

    results = mod.hybrid_search("risk from relying on overseas factories", top_k=2)
    tickers = [r.ticker for r in results[:2]]
    assert "AAPL" in tickers
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
pytest tests/test_retrieval.py -v
```

Expected: FAIL with `ModuleNotFoundError: No module named 'core.retrieval'` (if `VOYAGE_API_KEY` is set; otherwise `SKIPPED` — set the key temporarily to confirm the fail-first step, then proceed)

- [ ] **Step 3: Implement `core/retrieval.py`**

```python
from qdrant_client import models

from core.clients import (
    COLLECTION_NAME,
    DENSE_MODEL,
    get_qdrant_client,
    get_sparse_model,
    get_voyage_client,
)
from core.models import RetrievedChunk


def embed_query_dense(query: str) -> list[float]:
    result = get_voyage_client().embed([query], model=DENSE_MODEL, input_type="query")
    return result.embeddings[0]


def embed_query_sparse(query: str) -> models.SparseVector:
    embedding = next(get_sparse_model().embed([query]))
    return models.SparseVector(
        indices=embedding.indices.tolist(), values=embedding.values.tolist()
    )


def hybrid_search(query: str, top_k: int = 8) -> list[RetrievedChunk]:
    """Run hybrid (dense + sparse) search over the sec_filings collection, fused via RRF."""
    dense_vector = embed_query_dense(query)
    sparse_vector = embed_query_sparse(query)

    results = get_qdrant_client().query_points(
        collection_name=COLLECTION_NAME,
        prefetch=[
            models.Prefetch(query=dense_vector, using="dense", limit=top_k * 2),
            models.Prefetch(query=sparse_vector, using="sparse", limit=top_k * 2),
        ],
        query=models.FusionQuery(fusion=models.Fusion.RRF),
        limit=top_k,
    )

    return [
        RetrievedChunk(
            ticker=point.payload["ticker"],
            company_name=point.payload["company_name"],
            filing_type=point.payload["filing_type"],
            section=point.payload["section"],
            fiscal_year=point.payload["fiscal_year"],
            text=point.payload["text"],
            score=point.score,
        )
        for point in results.points
    ]
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
VOYAGE_API_KEY=<your-key> pytest tests/test_retrieval.py -v
```

Expected: `2 passed` (or `2 skipped` if no key is set — acceptable for CI, but must be run with a real key at least once before the demo)

- [ ] **Step 5: Commit**

```bash
git add core/retrieval.py tests/test_retrieval.py
git commit -m "feat: add hybrid dense+sparse retrieval with RRF fusion"
```

---

### Task 7: Core RAG Answer Generation

**Files:**
- Create: `core/rag.py`
- Test: `tests/test_rag.py`

**Interfaces:**
- Consumes: `core.retrieval.hybrid_search`, `core.models.{Citation, RAGAnswer, RetrievedChunk}`
- Produces: `core.rag.answer_question(question: str) -> RAGAnswer` — **the single public entrypoint the UI depends on**

**Design note:** Qdrant's RRF fusion score is a reciprocal-rank value (small, not a normalized similarity), so it isn't suitable as an absolute "is this relevant" threshold. Instead, "not found" is signaled two ways: (1) no chunks retrieved at all, or (2) the model itself, instructed to only answer from the excerpts, returns an empty citation list — a more reliable grounding signal than thresholding a rank-based score.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_rag.py
from unittest.mock import MagicMock, patch

from core.models import RetrievedChunk
import core.rag as mod


def _make_chunk(ticker="JPM", text="Net interest margin was 2.7%.", section="Item 7", score=0.02):
    return RetrievedChunk(
        ticker=ticker,
        company_name="JPMorgan Chase & Co.",
        filing_type="10-K",
        section=section,
        fiscal_year="2024",
        text=text,
        score=score,
    )


def _mock_anthropic_response(text: str):
    mock_content = MagicMock()
    mock_content.text = text
    mock_response = MagicMock()
    mock_response.content = [mock_content]
    return mock_response


def test_answer_question_returns_grounded_answer_with_citations():
    chunk = _make_chunk()
    fake_response = _mock_anthropic_response(
        "The net interest margin was 2.7% for the fiscal year.\nCITED: 0"
    )

    with patch.object(mod, "hybrid_search", return_value=[chunk]), patch.object(
        mod, "get_anthropic_client"
    ) as mock_get_client:
        mock_get_client.return_value.messages.create.return_value = fake_response
        result = mod.answer_question("What was JPMorgan's net interest margin?")

    assert "2.7%" in result.answer
    assert len(result.citations) == 1
    assert result.citations[0].ticker == "JPM"


def test_answer_question_returns_not_found_when_no_chunks_retrieved():
    with patch.object(mod, "hybrid_search", return_value=[]):
        result = mod.answer_question("What is the capital of France?")

    assert "couldn't find" in result.answer
    assert result.citations == []


def test_answer_question_returns_not_found_when_model_cites_nothing():
    chunk = _make_chunk()
    fake_response = _mock_anthropic_response(
        "The excerpts don't contain this information.\nCITED:"
    )

    with patch.object(mod, "hybrid_search", return_value=[chunk]), patch.object(
        mod, "get_anthropic_client"
    ) as mock_get_client:
        mock_get_client.return_value.messages.create.return_value = fake_response
        result = mod.answer_question("What is the capital of France?")

    assert "couldn't find" in result.answer
    assert result.citations == []
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
pytest tests/test_rag.py -v
```

Expected: FAIL with `ModuleNotFoundError: No module named 'core.rag'`

- [ ] **Step 3: Implement `core/rag.py`**

```python
import os
import re

import anthropic

from core.models import Citation, RAGAnswer, RetrievedChunk
from core.retrieval import hybrid_search

MODEL = "claude-sonnet-5"
NOT_FOUND_MESSAGE = "I couldn't find information about this in the available filings."

_client: anthropic.Anthropic | None = None


def get_anthropic_client() -> anthropic.Anthropic:
    global _client
    if _client is None:
        _client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
    return _client


def _build_prompt(question: str, chunks: list[RetrievedChunk]) -> str:
    context_blocks = "\n\n".join(
        f"[{i}] ({chunk.company_name}, {chunk.section}, FY{chunk.fiscal_year})\n{chunk.text}"
        for i, chunk in enumerate(chunks)
    )
    return (
        "Answer the question using ONLY the numbered filing excerpts below. "
        "After your answer, on a new line, write 'CITED: ' followed by a comma-separated "
        "list of the excerpt numbers you used (e.g. 'CITED: 0,2'). "
        "If the excerpts don't contain enough information to answer, say so explicitly "
        "and write 'CITED: ' with nothing after it.\n\n"
        f"Excerpts:\n{context_blocks}\n\nQuestion: {question}"
    )


def _parse_response(
    raw_text: str, chunks: list[RetrievedChunk]
) -> tuple[str, list[Citation]]:
    text = raw_text.strip()
    match = re.search(r"CITED:\s*([\d,\s]*)\s*$", text)
    if not match:
        return text, []

    answer_text = text[: match.start()].strip()
    indices_str = match.group(1).strip()
    if not indices_str:
        return answer_text, []

    citations = []
    for idx_str in indices_str.split(","):
        idx_str = idx_str.strip()
        if not idx_str.isdigit():
            continue
        idx = int(idx_str)
        if 0 <= idx < len(chunks):
            chunk = chunks[idx]
            citations.append(
                Citation(
                    ticker=chunk.ticker,
                    company_name=chunk.company_name,
                    section=chunk.section,
                    fiscal_year=chunk.fiscal_year,
                    text_snippet=chunk.text[:300],
                )
            )
    return answer_text, citations


def answer_question(question: str) -> RAGAnswer:
    """Retrieve relevant filing chunks via hybrid search and generate a grounded, cited answer."""
    chunks = hybrid_search(question, top_k=8)

    if not chunks:
        return RAGAnswer(question=question, answer=NOT_FOUND_MESSAGE, citations=[])

    prompt = _build_prompt(question, chunks)
    response = get_anthropic_client().messages.create(
        model=MODEL,
        max_tokens=1024,
        messages=[{"role": "user", "content": prompt}],
    )
    raw_text = response.content[0].text
    answer_text, citations = _parse_response(raw_text, chunks)

    if not citations:
        return RAGAnswer(question=question, answer=NOT_FOUND_MESSAGE, citations=[])

    return RAGAnswer(question=question, answer=answer_text, citations=citations)
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
pytest tests/test_rag.py -v
```

Expected: `3 passed`

- [ ] **Step 5: Commit**

```bash
git add core/rag.py tests/test_rag.py
git commit -m "feat: add grounded answer generation with citation parsing"
```

---

### Task 8: Streamlit UI

**Files:**
- Create: `ui/app.py`
- Create: `ui/__init__.py` (empty)

**Interfaces:**
- Consumes: `core.rag.answer_question`, `core.models.RAGAnswer` — **no other core/ingestion imports allowed in this file**
- Produces: a runnable Streamlit app

This task has no automated test — Streamlit UIs aren't meaningfully unit-testable at this level of effort, and the spec calls for manual verification here. Verify in a real browser per the global guidance on UI changes.

- [ ] **Step 1: Create `ui/__init__.py`**

```bash
touch ui/__init__.py
```

- [ ] **Step 2: Implement `ui/app.py`**

```python
import streamlit as st

from core.models import RAGAnswer
from core.rag import answer_question

COMPANIES = [
    "Apple (AAPL)",
    "JPMorgan Chase (JPM)",
    "Visa (V)",
    "Bank of America (BAC)",
    "Wells Fargo (WFC)",
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


def main() -> None:
    st.set_page_config(page_title="FinTech RAG Demo")
    st.title("SEC Filing Q&A")

    with st.sidebar:
        st.subheader("Companies in this demo")
        for company in COMPANIES:
            st.write(f"- {company}")

    question = st.text_input("Ask a question about these filings")
    if st.button("Ask") and question:
        with st.spinner("Retrieving and generating answer..."):
            result = answer_question(question)
        render_answer(result)


if __name__ == "__main__":
    main()
```

- [ ] **Step 3: Verify no forbidden imports leaked into the UI**

```bash
grep -nE "^(import|from) (qdrant|voyageai|anthropic)" ui/app.py
```

Expected: no output (empty grep match — confirms the UI only depends on `core.rag`/`core.models`)

- [ ] **Step 4: Manually verify in a browser**

Requires Task 9's full pipeline to have been run at least once (real filings fetched, chunked, indexed) so there's real data to query.

```bash
streamlit run ui/app.py
```

Open the printed local URL (typically `http://localhost:8501`), type a question such as *"What was JPMorgan's net interest margin?"*, click **Ask**, and confirm:
- An answer appears
- Expanding "Sources" shows at least one citation with company/section/fiscal year
- The sidebar lists all 5 companies

- [ ] **Step 5: Commit**

```bash
git add ui/app.py ui/__init__.py
git commit -m "feat: add minimal Streamlit UI decoupled from core RAG logic"
```

---

### Task 9: Full Pipeline Run, Smoke Test & Demo Verification

**Files:**
- Create: `scripts/smoke_test.py`
- Create: `scripts/__init__.py` (empty)

**Interfaces:**
- Consumes: `core.rag.answer_question`, `ingestion.fetch_filings.fetch_all_filings`, `ingestion.chunk_filings.chunk_all_filings`, `ingestion.index_to_qdrant.index_all_filings`
- Produces: a working end-to-end demo corpus in Qdrant, and a script for pre-demo sanity checking

- [ ] **Step 1: Create `scripts/__init__.py`**

```bash
touch scripts/__init__.py
```

- [ ] **Step 2: Implement `scripts/smoke_test.py`**

```python
from core.rag import answer_question

QUESTIONS = [
    "What was JPMorgan's net interest margin discussion in its most recent 10-K?",
    "How does Apple describe supply chain risk?",
    "What does Visa say about its net revenue?",
]


def main() -> None:
    for question in QUESTIONS:
        print(f"\nQ: {question}")
        result = answer_question(question)
        print(f"A: {result.answer}")
        for citation in result.citations:
            print(f"  - {citation.company_name} / {citation.section} (FY{citation.fiscal_year})")


if __name__ == "__main__":
    main()
```

- [ ] **Step 3: Set real API keys**

```bash
cp .env.example .env
```

Edit `.env` and fill in real `ANTHROPIC_API_KEY` and `VOYAGE_API_KEY` values, then:

```bash
export $(grep -v '^#' .env | xargs)
```

- [ ] **Step 4: Run the full offline ingestion pipeline**

```bash
python -m ingestion.fetch_filings
python -m ingestion.chunk_filings
python -m ingestion.index_to_qdrant
```

Expected: no errors; `data/raw/` has 5 `.html` files, `data/chunks/` has 5 `.json` files, and Qdrant has the `sec_filings` collection populated.

- [ ] **Step 5: Verify Qdrant was populated**

```bash
curl -s http://localhost:6333/collections/sec_filings | python -m json.tool
```

Expected: JSON showing `"points_count"` greater than 0.

- [ ] **Step 6: Run the smoke test against the real pipeline**

```bash
python scripts/smoke_test.py
```

Expected: three question/answer pairs printed, each with at least one citation referencing one of the 5 companies, and answers that plausibly reflect real filing content (not hallucinated financial figures).

- [ ] **Step 7: Run the full automated test suite**

```bash
pytest -v
```

Expected: all tests pass (or `test_retrieval.py` tests skip cleanly if `VOYAGE_API_KEY` wasn't exported in this shell — re-run with the key exported to confirm they pass before the actual demo).

- [ ] **Step 8: Manually verify the UI end-to-end** (per Task 8, Step 4)

```bash
streamlit run ui/app.py
```

Ask all three smoke-test questions in the browser and confirm answers + citations render correctly.

- [ ] **Step 9: Commit**

```bash
git add scripts/smoke_test.py scripts/__init__.py
git commit -m "feat: add smoke test script and verify end-to-end pipeline"
```

Note: `data/raw/` and `data/chunks/` are generated caches — add a `.gitignore` entry for `data/` if you don't want ~5 filings' worth of HTML/JSON committed to the repo.

```bash
echo "data/" >> .gitignore
git add .gitignore
git commit -m "chore: ignore generated filing cache"
```

---
