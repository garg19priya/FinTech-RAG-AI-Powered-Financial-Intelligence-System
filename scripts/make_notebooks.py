"""Generate the FinTech RAG walkthrough notebooks into <repo>/notebooks/.

    python scripts/make_notebooks.py notebooks

Overwrites the notebooks and clears their outputs. Edit this file, not the .ipynb files.
"""
import sys
from pathlib import Path

import nbformat as nbf

OUT = Path(sys.argv[1])
OUT.mkdir(parents=True, exist_ok=True)

SETUP = '''# ── Settings ────────────────────────────────────────────────────────────
# False: use the Qdrant server from `docker compose up -d`.
# True:  no Docker. Qdrant runs inside this notebook and stores data in data/qdrant_local/.
#        Restart the kernel after changing this, and run one notebook at a time
#        (only one process can open the embedded folder).
USE_EMBEDDED_QDRANT = False

# Setup: run from the repo root so `core`/`ingestion` imports and data/ paths work.
import logging
import os
import sys
import warnings
from pathlib import Path

ROOT = Path.cwd().parent if Path.cwd().name == "notebooks" else Path.cwd()
os.chdir(ROOT)
sys.path.insert(0, str(ROOT))
logging.basicConfig(level=logging.WARNING)
if USE_EMBEDDED_QDRANT:
    os.environ["QDRANT_PATH"] = "data/qdrant_local"
warnings.filterwarnings("ignore", message="Qdrant client version")  # harmless client/server version notice

import pandas as pd

import core  # noqa: F401  (loads .env)
import core.embeddings as emb
import core.llm as llm
from core.clients import COLLECTION_NAME, QDRANT_PATH, QDRANT_URL, get_qdrant_client

pd.set_option("display.max_colwidth", 120)
print(f"LLM:        {llm.PROVIDER} / {llm.MODEL}")
print(f"Embeddings: {emb.PROVIDER} / {emb.MODEL}")
print(f"Qdrant:     {'embedded @ ' + QDRANT_PATH if QDRANT_PATH else 'server @ ' + QDRANT_URL}")'''

QDRANT_CHECK = '''# Is the vector database running?
try:
    names = [c.name for c in get_qdrant_client().get_collections().collections]
    print("Qdrant is up. Collections:", names or "(none yet)")
except Exception as exc:
    raise SystemExit(
        "Qdrant is not reachable. Either open Docker Desktop and run `docker compose up -d`, "
        "or set USE_EMBEDDED_QDRANT = True in the first cell and restart the kernel."
    ) from exc'''

READY = QDRANT_CHECK + '''

# Make sure the data is ready, so this notebook runs on its own. prepare_data() fetches,
# chunks and indexes only what's missing: a few minutes the very first time, then seconds.
from ingestion.prepare import prepare_data

for name, count in prepare_data().items():
    print(f"{name}: {count} points")'''

SETUP_NOTE = '''
### Setup
The next cell connects the notebook to the project. Read its first lines: `USE_EMBEDDED_QDRANT` picks where the database runs (Docker server, or embedded with no Docker). It prints which LLM, embedding model and database this run uses, all taken from your `.env`.
'''


def md(text):
    return nbf.v4.new_markdown_cell(text.strip())


def code(text):
    return nbf.v4.new_code_cell(text.strip())


def save(name, cells):
    nb = nbf.v4.new_notebook()
    nb.cells = cells
    nb.metadata["kernelspec"] = {"name": "python3", "display_name": "Python 3", "language": "python"}
    nb.metadata["language_info"] = {"name": "python"}
    nbf.write(nb, OUT / name)
    print("wrote", OUT / name)


# =========================================================================== 00 start here
save("00_start_here.ipynb", [
    md('''
# 00 · Start here: what RAG is, from scratch

**You'll learn:** the problem this project solves, the idea behind retrieval-augmented generation (RAG), and the five building blocks (tokens, embeddings, keyword search, rank fusion, grounded prompts), each with a tiny hands-on demo.

**Time:** about 15 minutes · **Cost:** under 1¢ (a few embedding calls and one short LLM call) · **Needs:** an API key in `.env`. No database needed for this notebook.

| Part | What it covers |
|---|---|
| 1 | The problem |
| 2 | The idea: retrieval-augmented generation |
| 3 | Check your setup |
| 4 | The building blocks, hands-on: 4.1 tokens · 4.2 embeddings · 4.3 keyword search (BM25) · 4.4 rank fusion (RRF) · 4.5 grounded prompts and citations |
| 5 | Where to go next |

Run the cells top to bottom with **Shift + Enter**.
'''),
    md('''
---
## Part 1 · The problem

A company's annual report to the US regulator (the **10-K**) is long and dense. Apple's is 121 pages of text, risk factors and financial tables. Say you want one fact: *"Why did Greater China sales fall in 2024?"*

| Approach | What goes wrong |
|---|---|
| Read it yourself | Slow. The answer is one paragraph somewhere in Item 7. |
| Ask a chatbot directly | It answers from memory: maybe outdated, maybe invented, and with no source you can check. |
| Paste the whole PDF into the chatbot | Expensive (tens of thousands of tokens every question), may not fit, and still hard to verify. |

What we want: an answer that comes **only** from the filing, **cites** the exact passages, and **says "not found"** when the filing doesn't cover the question.
'''),
    md('''
---
## Part 2 · The idea: retrieval-augmented generation

RAG splits the work in two:

1. **Retrieve**: find the handful of passages in the filing that are relevant to the question.
2. **Generate**: give only those passages to an LLM, with strict instructions to answer from them and cite them.

The project runs this in two phases:

```mermaid
flowchart LR
  subgraph P1["Phase 1 · Ingest (run once)"]
    direction LR
    A["10-K PDF"] --> B["Chunk into passages"] --> C["Embed: dense + sparse vectors"] --> D[("Qdrant vector DB")]
  end
  subgraph P2["Phase 2 · Answer (every question)"]
    direction LR
    Q["Question"] --> S["Hybrid search"] --> K["Top 8 passages"] --> G["Grounded prompt + LLM"] --> R["Answer + citations, or 'not found'"]
  end
  D -. searched by .-> S
```

Phase 1 is notebook **01**, Phase 2 is notebook **02**. Before that, Part 4 below shows each building block on toy examples.
'''),
    md('''
---
## Part 3 · Check your setup
''' + SETUP_NOTE),
    code(SETUP),
    md('''
This cell checks that the key for your chosen provider is present (it prints only yes or no, never the key itself).
'''),
    code('''
import platform

needed = {llm.PROVIDER, emb.PROVIDER}
keys = {"openai": "OPENAI_API_KEY", "gemini": "GEMINI_API_KEY"}
print("Python:", platform.python_version())
for provider in sorted(needed):
    print(f"{keys[provider]} set:", "yes" if os.environ.get(keys[provider]) else "NO: add it to .env, then restart the kernel")
'''),
    md('''
---
## Part 4 · The building blocks, hands-on

### 4.1 Tokens: how models measure text

Models don't read characters or words; they read **tokens**, pieces of words. Everything is counted in tokens: how much text fits in one request, how much it costs, and how big our chunks are. Rule of thumb: **1 token ≈ ¾ of a word**, so 800 tokens ≈ 600 words.
'''),
    code('''
sentence = "Greater China net sales decreased during 2024 compared to 2023."

# Option A: OpenAI (tiktoken, local). Shows the individual tokens.
import tiktoken

enc = tiktoken.encoding_for_model("gpt-4o")
token_ids = enc.encode(sentence)
print(f"{len(sentence.split())} words -> {len(token_ids)} tokens")
print([enc.decode([t]) for t in token_ids])

# Option B: Gemini (API call). Gemini only returns the count, not the token pieces.
# import os
# from google import genai
#
# client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])
# result = client.models.count_tokens(model="gemini-3.5-flash", contents=sentence)
# print(f"{len(sentence.split())} words -> {result.total_tokens} tokens")

# Option C: Gemini tokens shown locally. Uses Google's open Gemma 3 tokenizer, which is
# the one the google-genai SDK uses for its local tokenizer. Needs sentencepiece (in pyproject).
# (an approximation: newer Gemini models may differ slightly).
# import urllib.request
# import sentencepiece as spm
#
# url = "https://raw.githubusercontent.com/google/gemma_pytorch/014acb7ac4563a5f77c76d7ff98f31b568c16508/tokenizer/gemma3_cleaned_262144_v2.spiece.model"
# sp = spm.SentencePieceProcessor(model_proto=urllib.request.urlopen(url).read())
# pieces = sp.encode(sentence, out_type=str)
# print(f"{len(sentence.split())} words -> {len(pieces)} tokens")
# print(pieces)  # "▁" marks a leading space
'''),
    md('''
**What to notice:** common words are one token; rarer pieces split up. This is why the project measures chunk size in tokens (the advanced chunker caps chunks at 800).

### 4.2 Embeddings: turning meaning into numbers

An **embedding** is a list of numbers (here 1536 of them) that represents what a text *means*. Texts with similar meaning get similar lists, even if they share no words. We compare two embeddings with **cosine similarity**: 1.0 = same direction (same meaning), near 0 = unrelated.

The cell embeds five sentences plus a question, and ranks the sentences by similarity to the question.
'''),
    code('''
import numpy as np

from core.embeddings import embed_dense

sentences = [
    "Total net sales were $391.0 billion in 2024.",
    "Apple's revenue increased compared to the prior year.",
    "iPhone sales in Greater China declined.",
    "The company faces risks from foreign currency fluctuations.",
    "The weather in Paris is sunny today.",
]
question = "How much money did Apple make from sales?"

vectors = np.array(embed_dense(sentences + [question]))
vectors = vectors / np.linalg.norm(vectors, axis=1, keepdims=True)  # unit length, so dot product = cosine
similarity = vectors[:-1] @ vectors[-1]

print(f"Each embedding has {vectors.shape[1]} numbers.\\n")
pd.DataFrame({"sentence": sentences, "cosine similarity to the question": similarity.round(3)}).sort_values(
    "cosine similarity to the question", ascending=False
)
'''),
    md('''
**What to notice:** the sentences about sales and revenue rank highest even though the question says "money" and "make", not "net sales". The Paris sentence ranks last. That is **dense (semantic) search**: it matches meaning.

### 4.3 Keyword search (BM25): matching exact words

Dense search can miss exact names, numbers and jargon such as "Item 1C" or "Greater China". **BM25** is classic keyword scoring: a text scores higher when it contains the query's words, especially rare ones. In Qdrant each text becomes a **sparse vector** (a weight for each word it contains), and the score is the dot product of the query's and the text's sparse vectors.
'''),
    code('''
from core.clients import get_sparse_model


def sparse_score(a, b):
    weights = dict(zip(a.indices.tolist(), a.values.tolist()))
    return sum(weights.get(i, 0.0) * v for i, v in zip(b.indices.tolist(), b.values.tolist()))


bm25 = get_sparse_model()
keyword_query = "Greater China net sales"
doc_vectors = list(bm25.embed(sentences))
query_vector = next(bm25.query_embed(keyword_query))

pd.DataFrame({
    "sentence": sentences,
    "BM25 score": [round(sparse_score(query_vector, d), 3) for d in doc_vectors],
}).sort_values("BM25 score", ascending=False)
'''),
    md('''
**What to notice:** only sentences sharing words with the query score above 0; "revenue increased" scores 0 even though it means something close. Dense and sparse search fail in opposite ways, so the project uses **both** ("hybrid search").

### 4.4 Rank fusion (RRF): combining two result lists

Hybrid search gives two ranked lists. Their scores aren't comparable (cosine is 0 to 1; BM25 is unbounded), so we combine **ranks** instead. **Reciprocal Rank Fusion** gives each passage `1 / (k + rank)` from each list and adds them up. `k` (60 is the classic value) softens the gap between rank 1 and rank 2. A passage near the top of *both* lists wins.
'''),
    code('''
dense_ranking = ["Greater China paragraph", "Segment table", "Risk: China", "Revenue overview"]
sparse_ranking = ["Segment table", "Greater China paragraph", "Store count in China", "Risk: China"]
K = 60

scores = {}
for ranking in (dense_ranking, sparse_ranking):
    for rank, passage in enumerate(ranking, start=1):
        scores[passage] = scores.get(passage, 0) + 1 / (K + rank)

pd.DataFrame(
    [
        {
            "passage": p,
            "dense rank": dense_ranking.index(p) + 1 if p in dense_ranking else "-",
            "sparse rank": sparse_ranking.index(p) + 1 if p in sparse_ranking else "-",
            "RRF score": round(s, 4),
        }
        for p, s in sorted(scores.items(), key=lambda kv: -kv[1])
    ]
)
'''),
    md('''
**What to notice:** passages found by both searches rise to the top; one found by a single search drops. The real pipeline keeps the **top k = 8** after fusion (notebook 02, Part 2).

### 4.5 Grounded prompts and citations

The last block is the prompt. The retrieved passages are numbered, and the LLM is told: *answer using ONLY these excerpts, then write `CITED:` with the numbers you used*. The code then turns those numbers back into citations. No citations means the app answers "not found".

Here are two made-up excerpts run through the project's real prompt builder, an LLM call, and the citation parser:
'''),
    code('''
from core.llm import complete_with_usage
from core.models import RetrievedChunk
from core.rag import _build_prompt, _parse_response

toy_chunks = [
    RetrievedChunk(id=0, ticker="AAPL", company_name="Apple Inc.", filing_type="10-K", section="Item 7",
                   fiscal_year="2024", text="Greater China net sales decreased due to lower iPhone sales.", score=1.0),
    RetrievedChunk(id=1, ticker="AAPL", company_name="Apple Inc.", filing_type="10-K", section="Item 1A",
                   fiscal_year="2024", text="The Company is exposed to foreign currency risk.", score=0.5),
]
prompt = _build_prompt("Why did Greater China sales fall?", toy_chunks)
print(prompt, "\\n\\n----- LLM reply -----")
reply = complete_with_usage(prompt, max_tokens=200)
print(reply.text)
answer, citations = _parse_response(reply.text, toy_chunks)
print("\\nParsed citations:", [c.section for c in citations])
'''),
    md('''
**What to notice:** the reply ends with a `CITED:` line pointing at excerpt `0`, and the parser turned it into a citation for Item 7. Ask something the excerpts don't cover and `CITED:` comes back empty, which is how the app knows to say "not found".

---
## Part 5 · Where to go next

| Notebook | What you'll do |
|---|---|
| **01_ingest** | Phase 1 for real: fetch the 10-K, chunk it two ways, embed, and index into Qdrant |
| **02_answer** | Phase 2 for real: follow one question through search, fusion, prompt, LLM and citations |
| **03_models_and_experiments** | Compare OpenAI vs Gemini, the two chunkers, five retrieval strategies, and run a mini evaluation |
| **04_agentic_rag** | Turn this pipeline into an agent that decides what to search for, and how often |

Each notebook runs on its own: it builds any missing data itself.
'''),
])

# =========================================================================== 01 ingest
save("01_ingest.ipynb", [
    md('''
# 01 · Ingest: turn a 10-K PDF into searchable chunks

**You'll learn:** how the filing becomes a searchable database: downloading it, cutting it into passages two different ways, turning passages into vectors, and storing them in Qdrant.

**Time:** about 10 minutes (longer the first time docling runs) · **Cost:** under 1¢, plus about a cent if it has to embed everything · **Needs:** an API key in `.env`, and a database: Docker Qdrant running, or `USE_EMBEDDED_QDRANT = True`.

```mermaid
flowchart LR
  PDF["10-K PDF<br/>121 pages"] --> B["Basic chunker<br/>PyPDF text + regex Items<br/>800-token pieces"]
  PDF --> A["Advanced chunker<br/>docling layout<br/>tables kept whole"]
  B --> EB["Embed<br/>dense + sparse"] --> QB[("sec_filings_basic")]
  A --> EA["Embed<br/>dense + sparse"] --> QA[("sec_filings_advanced<br/>used by Q&A")]
```

| Part | What it covers | Code |
|---|---|---|
| 1 | Setup | |
| 2 | Fetch the filing | `ingestion/fetch_filing_pdf.py` |
| 3 | Chunk it two ways, and compare | `ingestion/chunk_filings_basic.py`, `ingestion/chunk_filings_advanced.py` |
| 4 | Embed: dense and sparse vectors | `core/embeddings.py`, `core/clients.py` |
| 5 | Index into Qdrant | `ingestion/index_to_qdrant.py`, `ingestion/index_comparison_collections.py` |
| 6 | Recap and things to try | |

New to tokens, embeddings or BM25? Run `00_start_here.ipynb` first.
'''),
    md('''
---
## Part 1 · Setup
''' + SETUP_NOTE),
    code(SETUP),
    md('''
---
## Part 2 · Fetch the filing

A 10-K is organised into numbered **Items**. These are the ones you'll see most in answers:

| Item | Contents |
|---|---|
| Item 1 | Business: products, segments, competition |
| Item 1A | Risk Factors |
| Item 1C | Cybersecurity |
| Item 7 | Management's Discussion and Analysis (MD&A): why the numbers moved |
| Item 8 | Financial Statements: income statement, balance sheet, notes |

`fetch_filing_pdf()` downloads Apple's own published PDF once to `data/raw_pdf/AAPL.pdf`; later runs reuse the file.
'''),
    code('''
from pypdf import PdfReader

from ingestion.fetch_filing_pdf import fetch_filing_pdf

pdf_path = fetch_filing_pdf("AAPL")
reader = PdfReader(pdf_path)
print(f"{pdf_path}: {pdf_path.stat().st_size / 1e6:.1f} MB, {len(reader.pages)} pages\\n")
print("Start of page 1:\\n", reader.pages[0].extract_text()[:500])
'''),
    md('''
---
## Part 3 · Chunk it two ways

### 3.1 Why chunk at all?

We can't search or prompt with the whole 121-page PDF at once. We cut it into **chunks**: passages small enough that:
- a search can point at the *specific* passage that answers a question, not a whole chapter;
- 8 of them fit comfortably in one LLM prompt;
- each one still has enough surrounding context to make sense alone.

How you cut matters a lot, especially for financial tables. The project tries two ways so you can compare them.

### 3.2 Basic chunker

1. Extract plain text page by page with PyPDF.
2. Find the "Item 1A", "Item 7"… headers with a regular expression, so each chunk knows its section.
3. Split each section into pieces of up to **800 tokens**, with **150 tokens of overlap** between neighbouring pieces, so a sentence cut at a boundary still appears whole in one of them.

It's fast, but PDF text extraction flattens tables into lines of loose numbers.
'''),
    code('''
import json
from collections import Counter

from core.models import FilingChunk
from ingestion.chunk_filings_basic import chunk_filing_basic


def save_chunks(chunks, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps([c.__dict__ for c in chunks], indent=2), encoding="utf-8")


basic_chunks = chunk_filing_basic("AAPL", pdf_path, fiscal_year="2024")
save_chunks(basic_chunks, Path("data/chunks_basic/AAPL.json"))

print(f"basic: {len(basic_chunks)} chunks across {len({c.section for c in basic_chunks})} sections\\n")
for section, count in Counter(c.section[:60] for c in basic_chunks).most_common(6):
    print(f"{count:4} chunks  {section}")
'''),
    md('''
### 3.3 Advanced chunker

**docling** reads the PDF's *layout*, not just its text: it recognises headings, paragraphs and tables. Its `HybridChunker` then:
- keeps each table together and writes it as a **markdown table**, so rows and columns survive;
- merges small neighbouring pieces and splits big ones, so each chunk is **at most 800 tokens** (counted with the gpt-4o tokenizer);
- records the heading path, e.g. `Item 8 > Note 13 - Segment Information`.

The first run is slow because docling downloads its layout models, so the result is reused from `data/chunks_advanced/` when it exists. Set `RECHUNK = True` to rebuild it (several minutes).
'''),
    code('''
from ingestion.chunk_filings_advanced import chunk_filing_advanced

RECHUNK = False  # True = run docling again

adv_path = Path("data/chunks_advanced/AAPL.json")
if RECHUNK or not adv_path.exists():
    advanced_chunks = chunk_filing_advanced("AAPL", pdf_path, fiscal_year="2024")
    save_chunks(advanced_chunks, adv_path)
else:
    advanced_chunks = [FilingChunk(**c) for c in json.loads(adv_path.read_text(encoding="utf-8"))]

print(f"advanced: {len(advanced_chunks)} chunks")
print(f"chunks containing a markdown table: {sum('|' in c.text for c in advanced_chunks)}")
'''),
    md('''
### 3.4 What one chunk looks like

Every chunk is a `FilingChunk` (`core/models.py`). Its fields become the **payload** stored next to the vectors in Qdrant, which is how an answer can cite "Apple Inc. · Item 7 · FY2024".
'''),
    code('''
example = advanced_chunks[len(advanced_chunks) // 2]
pd.Series({k: (v[:200] + "…" if isinstance(v, str) and len(v) > 200 else v) for k, v in example.__dict__.items()})
'''),
    md('''
### 3.5 How big are the chunks?

The dashed line is the 800-token cap. Basic chunks pile up just under it, because the splitter fills each piece. Advanced chunks vary more: a short paragraph stays short. A few advanced chunks land slightly above 800 because docling adds the heading path after sizing the chunk.
'''),
    code('''
import matplotlib.pyplot as plt
import tiktoken

enc = tiktoken.encoding_for_model("gpt-4o")
sizes = pd.DataFrame({
    "pipeline": ["basic"] * len(basic_chunks) + ["advanced"] * len(advanced_chunks),
    "tokens": [len(enc.encode(c.text)) for c in basic_chunks + advanced_chunks],
})
display(sizes.groupby("pipeline")["tokens"].describe().round(0))

fig, ax = plt.subplots(figsize=(9, 3.5))
for name, color in [("basic", "#8A96A8"), ("advanced", "#2459C2")]:
    ax.hist(sizes[sizes.pipeline == name].tokens, bins=40, alpha=0.7, label=name, color=color)
ax.axvline(800, color="#C2560F", linestyle="--", label="800-token cap")
ax.set_xlabel("tokens per chunk")
ax.set_ylabel("chunks")
ax.legend()
plt.show()
'''),
    md('''
### 3.6 Same table, two chunkers

The clearest difference is on tables. Below is the geographic net-sales table as each chunker stored it. In the basic version the numbers run together on lines; in the advanced version each row and column is still there, which helps both the search and the LLM.
'''),
    code('''
def first_with(chunks, needle):
    return next((c for c in chunks if needle in c.text), None)


for name, chunks in [("BASIC", basic_chunks), ("ADVANCED", advanced_chunks)]:
    chunk = first_with(chunks, "Total net sales")
    print(f"===== {name} · section: {chunk.section[:70]}")
    start = max(chunk.text.find("Total net sales") - 300, 0)
    print(chunk.text[start:start + 700], "\\n")
'''),
    md('''
> **Pitfall:** chunks are tagged `fiscal_year="2024"` by a default in the chunkers, whichever year the PDF actually covers. If Apple publishes a newer 10-K, the label won't change by itself.

---
## Part 4 · Embed: dense and sparse vectors

Each chunk gets **two** vectors:

| | Dense | Sparse (BM25) |
|---|---|---|
| What it captures | meaning | exact words |
| Shape | 1536 numbers, all filled in | a weight for each distinct word in the chunk |
| Made by | OpenAI `text-embedding-3-small` or Gemini `gemini-embedding-001` (`EMBEDDING_PROVIDER`) | fastembed `Qdrant/bm25`, locally, free |
| Good at | paraphrases: "revenue" ≈ "net sales" | names, numbers, jargon: "Item 1C", "Greater China" |

Here we embed just 3 chunks to see the shapes (a fraction of a cent).
'''),
    code('''
import tiktoken

from core.clients import get_sparse_model
from core.embeddings import embed_dense

enc = tiktoken.encoding_for_model("gpt-4o")  # rough token count, used only for the estimate

sample = [c.text for c in advanced_chunks[:3]]
dense = embed_dense(sample)
print(f"dense:  {len(dense)} vectors x {len(dense[0])} numbers, first five: {[round(x, 4) for x in dense[0][:5]]}")

sparse = next(get_sparse_model().embed([sample[0]]))
print(f"sparse: {len(sparse.indices)} non-zero entries (one per distinct word), e.g. {list(zip(sparse.indices[:3].tolist(), sparse.values[:3].round(3).tolist()))}")
print("        (each index is a hashed word id; each value is that word's BM25 weight)")

total_tokens = sum(len(enc.encode(c.text)) for c in basic_chunks + advanced_chunks)
print(f"\\nEmbedding all {len(basic_chunks) + len(advanced_chunks)} chunks = about {total_tokens:,} tokens")
'''),
    md('''
---
## Part 5 · Index into Qdrant

Qdrant stores **points** inside **collections**:

```mermaid
flowchart LR
  subgraph COL["Collection: sec_filings_advanced"]
    P1["Point #42<br/>dense vector (1536)<br/>sparse vector<br/>payload: ticker, section, fiscal_year, text"]
    P2["Point #43 …"]
  end
```

- A **collection** is like a table. There is one per chunker: `sec_filings_basic` and `sec_filings_advanced`.
- A **point** is one chunk: an id, its two vectors (named `dense` and `sparse`), and its payload.

This cell embeds **every** chunk of both chunkers and stores them. It skips the work when both collections already exist. Set `REINDEX = True` after changing `EMBEDDING_PROVIDER` or `EMBEDDING_MODEL`: vectors from different models can't be mixed in one collection.
'''),
    code(QDRANT_CHECK),
    code('''
from ingestion.index_comparison_collections import (
    ADVANCED_COLLECTION,
    BASIC_COLLECTION,
    index_comparison_collections,
)

REINDEX = False

client = get_qdrant_client()
collections = (BASIC_COLLECTION, ADVANCED_COLLECTION)
if REINDEX or not all(client.collection_exists(name) for name in collections):
    for name in collections:
        if client.collection_exists(name):
            client.delete_collection(name)
    index_comparison_collections()  # embeds every chunk: a minute or two on OpenAI, ~10 min on Gemini's free tier (progress bar below)

for name in collections:
    print(f"{name}: {client.count(name).count} points")
'''),
    md('''
Peek inside: the collection's vector settings, and one stored point's payload.
'''),
    code('''
info = client.get_collection(ADVANCED_COLLECTION)
print("dense vectors: ", info.config.params.vectors)
print("sparse vectors:", list(info.config.params.sparse_vectors))

point = client.scroll(ADVANCED_COLLECTION, limit=1, with_payload=True, with_vectors=False)[0][0]
print(f"\\npoint id {point.id}, payload:")
pd.Series({k: (v[:150] + "…" if isinstance(v, str) and len(v) > 150 else v) for k, v in point.payload.items()})
'''),
    md('''
---
## Part 6 · Recap and things to try

**What happened**
1. The 10-K PDF was downloaded once and cached.
2. It was chunked two ways: basic (fast, flattens tables) and advanced (layout-aware, tables kept whole).
3. Each chunk got a dense vector (meaning) and a sparse vector (exact words).
4. Both chunk sets were stored as points in two Qdrant collections. The Q&A uses `sec_filings_advanced`.

One command does all of this outside the notebook: `python -m ingestion.prepare`.

**Try this**
- Search the chunks for a word: `[c.section for c in advanced_chunks if "Vision Pro" in c.text]`.
- Compare how many chunks each chunker gave the Risk Factors section (`Item 1A`).
- Find the biggest advanced chunk and read it: `max(advanced_chunks, key=lambda c: len(c.text))`.

**Next:** `02_answer.ipynb` follows one question through search, fusion, prompt and citations.
'''),
])

# =========================================================================== 02 answer
save("02_answer.ipynb", [
    md('''
# 02 · Answer: search, ground, cite

**You'll learn:** exactly what happens between typing a question and getting a cited answer, one step at a time, with the real data.

**Time:** about 10 minutes · **Cost:** about 2¢ · **Needs:** an API key in `.env`, and Docker Qdrant running or `USE_EMBEDDED_QDRANT = True`. **Runs on its own:** the second cell builds any missing data.

```mermaid
flowchart LR
  Q["1 · Question"] --> D["2a · Dense search<br/>meaning"]
  Q --> S["2b · Sparse search<br/>exact words"]
  D --> F["3 · RRF fusion"]
  S --> F
  F --> K["4 · Top-k = 8 chunks"]
  K --> P["5 · Grounded prompt"]
  P --> L["6 · LLM"]
  L --> C["7 · Parse CITED"]
  C -->|"cited ≥ 1"| A["8 · Answer + citations"]
  C -->|"cited none"| N["8 · 'Not found'"]
```

| Part | Steps | Code |
|---|---|---|
| 1 | Setup | |
| 2 | Retrieval: question, dense search, sparse search, fusion, top-k (steps 1–4) | `core/retrieval.py` |
| 3 | Generation: prompt, LLM, citations (steps 5–7) | `core/rag.py`, `core/llm.py` |
| 4 | The "not found" guardrail (step 8) | `core/rag.py` |
| 5 | Recap and your turn | |
'''),
    md('''
---
## Part 1 · Setup
''' + SETUP_NOTE),
    code(SETUP),
    code(READY),
    md('''
---
## Part 2 · Retrieval (steps 1–4)

### Step 1 · The question

This question needs one specific paragraph from Item 7. Change it later and re-run the notebook.
'''),
    code('''
QUESTION = "Why did Greater China net sales decrease in fiscal 2024?"
TOP_K = 8
'''),
    md('''
### Step 2a · Dense search (meaning)

The question is embedded with the same model as the chunks, and Qdrant returns the chunks whose vectors point in the most similar direction. **Score = cosine similarity**: higher is closer, and values usually fall between 0.2 and 0.8.

`hybrid_search()` fetches **2 × k = 16** candidates from each search before fusing, so we fetch 16 here too and show the first 8.
'''),
    code('''
from core.retrieval import embed_query_dense, embed_query_sparse

client = get_qdrant_client()
CANDIDATES = TOP_K * 2


def as_table(points):
    return pd.DataFrame([
        {
            "rank": i + 1,
            "id": p.id,
            "score": round(p.score, 4),
            "section": p.payload["section"][:60],
            "text": p.payload["text"][:110].replace("\\n", " "),
        }
        for i, p in enumerate(points)
    ])


dense_hits = client.query_points(
    COLLECTION_NAME, query=embed_query_dense(QUESTION), using="dense", limit=CANDIDATES
).points
as_table(dense_hits).head(TOP_K)
'''),
    md('''
### Step 2b · Sparse search (exact words)

The question becomes a BM25 sparse vector and is matched against each chunk's sparse vector. **Score = BM25**: it has no fixed upper limit, so don't compare these numbers with the dense scores. Chunks that literally contain "Greater China" and "net sales" do well here.
'''),
    code('''
sparse_hits = client.query_points(
    COLLECTION_NAME, query=embed_query_sparse(QUESTION), using="sparse", limit=CANDIDATES
).points
as_table(sparse_hits).head(TOP_K)
'''),
    md('''
How much do the two lists agree? Chunks found by both are strong candidates; chunks found by only one are what hybrid search adds.
'''),
    code('''
dense_ids = [p.id for p in dense_hits]
sparse_ids = [p.id for p in sparse_hits]
both = set(dense_ids) & set(sparse_ids)
print(f"found by both: {len(both)} · dense only: {len(set(dense_ids) - both)} · sparse only: {len(set(sparse_ids) - both)}  (of {CANDIDATES} each)")
'''),
    md('''
### Step 3 · RRF fusion, by hand and by Qdrant

**Reciprocal Rank Fusion** scores each chunk by its *rank* in each list, `Σ 1 / (k + rank)`, so the two kinds of scores never need to be compared. First by hand with the classic `k = 60`:
'''),
    code('''
K = 60
rrf = {}
for ranking in (dense_ids, sparse_ids):
    for rank, chunk_id in enumerate(ranking, start=1):
        rrf[chunk_id] = rrf.get(chunk_id, 0) + 1 / (K + rank)
manual_top = sorted(rrf, key=lambda i: -rrf[i])[:TOP_K]

pd.DataFrame([
    {
        "rank": n + 1,
        "id": i,
        "dense rank": dense_ids.index(i) + 1 if i in dense_ids else "-",
        "sparse rank": sparse_ids.index(i) + 1 if i in sparse_ids else "-",
        "RRF score": round(rrf[i], 4),
    }
    for n, i in enumerate(manual_top)
])
'''),
    md('''
### Step 4 · Top-k: what the pipeline actually keeps

`hybrid_search()` does steps 2a, 2b and 3 in **one** Qdrant call (two `Prefetch` searches plus `FusionQuery(fusion=Fusion.RRF)`) and keeps the best `TOP_K = 8`. Qdrant uses its own constant inside RRF, so its scores differ from ours, but the idea is the same; the last line shows how many chunks both versions agree on.
'''),
    code('''
from core.retrieval import hybrid_search

chunks = hybrid_search(QUESTION, top_k=TOP_K)


def found_by(chunk_id):
    sources = [name for name, ids in (("dense", dense_ids), ("sparse", sparse_ids)) if chunk_id in ids]
    return " + ".join(sources)


display(pd.DataFrame([
    {
        "rank": i + 1,
        "qdrant RRF score": round(c.score, 4),
        "found_by": found_by(c.id),
        "section": c.section[:60],
        "text": c.text[:100].replace("\\n", " "),
    }
    for i, c in enumerate(chunks)
]))
print(f"Same chunks as the hand-made top {TOP_K}: {len(set(manual_top) & {c.id for c in chunks})} of {TOP_K}")
'''),
    md('''
**Choosing k.** More chunks give the LLM more context but cost more tokens and can add noise. This cell builds the prompt for a few values of k (no LLM call) to show the token cost:
'''),
    code('''
import tiktoken

from core.rag import _build_prompt

enc = tiktoken.get_encoding("cl100k_base")
pd.DataFrame([
    {"top_k": k, "prompt tokens": len(enc.encode(_build_prompt(QUESTION, hybrid_search(QUESTION, top_k=k))))}
    for k in (3, 8, 16)
])
'''),
    md('''
---
## Part 3 · Generation (steps 5–7)

### Step 5 · The grounded prompt

`_build_prompt()` numbers the chunks `[0]…[7]`, labels each with company, section and fiscal year, and gives three instructions: answer **only** from the excerpts; end with `CITED: <numbers>`; if the excerpts aren't enough, say so and leave `CITED:` empty. Here is the beginning and the end of the real prompt:
'''),
    code('''
prompt = _build_prompt(QUESTION, chunks)
print(len(enc.encode(prompt)), "tokens\\n")
print(prompt[:700], "\\n\\n[... more excerpts ...]\\n\\n", prompt[-300:])
'''),
    md('''
### Step 6 · Call the LLM

`complete_with_usage()` sends the prompt to the provider in `.env` (`LLM_PROVIDER`) and returns the reply text plus token counts and an estimated cost. Look at the last line of the reply: the `CITED:` line.
'''),
    code('''
from core.llm import complete_with_usage

response = complete_with_usage(prompt)
print(response.text)
print(f"\\n{llm.PROVIDER}/{llm.MODEL}: {response.input_tokens} tokens in, {response.output_tokens} out, ≈ ${response.cost_usd:.4f}")
'''),
    md('''
### Step 7 · Parse `CITED:` into citations

`_parse_response()` finds the final `CITED:` line with a regular expression, removes it from the answer, and maps each number back to its chunk. Numbers that don't match a chunk are ignored, so the model can't cite something it wasn't shown.
'''),
    code('''
from core.rag import _parse_response

answer, citations = _parse_response(response.text, chunks)
print("ANSWER:", answer, "\\n")
for c in citations:
    print(f"- {c.company_name} · {c.section[:60]} · FY{c.fiscal_year}")
    print(f"  “{c.text_snippet[:160].strip()}…”")
'''),
    md('''
---
## Part 4 · The "not found" guardrail (step 8)

`answer_question()` runs steps 2–7 in one call. It refuses in two cases: the search returns nothing, or the LLM cites nothing. The second question isn't covered by this filing, so it should be refused instead of answered from the model's memory.
'''),
    code('''
from core.rag import answer_question

for q in [QUESTION, "What was Apple's revenue in fiscal year 2019?"]:
    result = answer_question(q)
    print(f"Q: {q}\\nA: {result.answer}")
    print(f"   {len(result.citations)} citation(s): {[c.section[:40] for c in result.citations]}\\n")
'''),
    md('''
---
## Part 5 · Recap and your turn

**What happened:** the question was searched by meaning and by keywords, the two lists were fused by rank, the best 8 chunks went into a strict prompt, and the model's `CITED:` line became the citations. No citation, no answer.

**Your turn:** change the question and run the cell. Ideas:
- *Which geographic segment had the largest revenue decline in 2024?*
- *What cybersecurity risks does Apple disclose in Item 1C?*
- Change `TOP_K` at the top of Part 2 to 3, re-run, and see whether the answer changes.
'''),
    code('''
result = answer_question("What was Apple's gross margin percentage in 2024 vs. 2023?")
print(result.answer)
for c in result.citations:
    print("-", c.section[:70])
'''),
    md('''
**Next:** `03_models_and_experiments.ipynb` compares models, chunkers and retrieval strategies; `04_agentic_rag.ipynb` turns this pipeline into an agent.
'''),
])

# =========================================================================== 03 experiments
save("03_models_and_experiments.ipynb", [
    md('''
# 03 · Models and experiments

**You'll learn:** how much each design choice matters: which LLM answers, how the filing was chunked, and how retrieval works. Then you'll measure quality with a small evaluation.

**Time:** about 10 minutes · **Cost:** a few cents · **Runs on its own:** run the two setup cells, then **any section in any order**; each section has its own imports.

```mermaid
flowchart LR
  Q["Question"] --> R{"Retrieval strategy<br/>Section C"}
  R --> COL{"Chunk collection<br/>Section B"}
  COL --> L{"LLM<br/>Section A"}
  L --> E["Answer → scored by<br/>Section D"]
```

| Section | Question it answers | Code |
|---|---|---|
| A | Does the answer change with OpenAI vs Gemini? | `core/llm.py` |
| B | Does layout-aware chunking help on tables? | `sec_filings_basic` vs `sec_filings_advanced` |
| C | Do smarter retrieval strategies find better chunks? | `core/retrieval_strategies.py` |
| D | How good is the pipeline, in numbers? | `eval/` |
'''),
    md('''
---
## Setup
''' + SETUP_NOTE),
    code(SETUP),
    code(READY),
    md('''
---
## Section A · OpenAI vs Gemini for the answer

The provider normally comes from `.env`. Here `use_llm()` switches it in memory, so the same question, retrieval and prompt go to each model. Providers without an API key in `.env` are skipped.

**What to look for:** whether both models cite the same section, and how time and cost compare.

> Only the **LLM** can be swapped like this. Changing the **embedding** model needs a re-index (`REINDEX = True` in notebook 01).
'''),
    code('''
import time

from core.rag import answer_question

QUESTION = "Why did Greater China net sales decrease in fiscal 2024?"


def use_llm(provider, model=None):
    llm.PROVIDER = provider
    llm.MODEL = model or llm._DEFAULT_MODEL[provider]


KEYS = {"openai": "OPENAI_API_KEY", "gemini": "GEMINI_API_KEY"}
original = (llm.PROVIDER, llm.MODEL)

rows = []
for provider in ("openai", "gemini"):
    if not os.environ.get(KEYS[provider]):
        print(f"skipping {provider}: no {KEYS[provider]} in .env")
        continue
    use_llm(provider)
    start = time.time()
    try:
        result = answer_question(QUESTION)
    except Exception as exc:
        print(f"{provider} failed: {str(exc)[:150]}")
        continue
    rows.append({
        "model": f"{provider}/{llm.MODEL}",
        "seconds": round(time.time() - start, 1),
        "citations": len(result.citations),
        "cost_usd≈": round(result.cost_usd, 5),
        "answer": result.answer,
    })

use_llm(*original)
pd.DataFrame(rows)
'''),
    md('''
---
## Section B · Basic vs advanced chunks

Same question, same search, same LLM; only the collection changes. Gross margin lives in tables, which the basic chunker flattens and the advanced one keeps.

**What to look for:** which sections come back on top, and whether the answer gives both years.
'''),
    code('''
from core.rag import answer_question
from core.retrieval import hybrid_search
from ingestion.index_comparison_collections import ADVANCED_COLLECTION, BASIC_COLLECTION

TABLE_QUESTION = "What was Apple's gross margin percentage in 2024 vs. 2023?"

for collection in (BASIC_COLLECTION, ADVANCED_COLLECTION):
    result = answer_question(TABLE_QUESTION, collection_name=collection)
    top = hybrid_search(TABLE_QUESTION, top_k=3, collection_name=collection)
    print(f"===== {collection}")
    print("top sections:", [c.section[:45] for c in top])
    print("answer:", result.answer, "\\n")
'''),
    md('''
---
## Section C · Retrieval strategies

`hybrid_search()` searches with the user's exact words. The strategies in `core/retrieval_strategies.py` try to search better:

| Strategy | Idea | Extra cost |
|---|---|---|
| `baseline` | plain hybrid search | none |
| `query_rewrite` | the LLM rewrites the question into a cleaner search query | 1 LLM call |
| `hyde` | the LLM writes a *hypothetical* 10-K passage that would answer it, and that passage is searched for | 1 LLM call |
| `multi_query` | the LLM writes several rephrasings; their results are fused with RRF | 1 LLM call + extra searches |
| `rerank` | fetch 3 × k candidates, then a cross-encoder model re-scores each (question, chunk) pair | local model (downloaded on first use) |

**What to look for:** whether the top sections change, and the time each strategy adds.
'''),
    code('''
import time

from core.retrieval_strategies import STRATEGIES

QUESTION = "Why did Greater China net sales decrease in fiscal 2024?"

rows = []
for name, strategy in STRATEGIES.items():
    start = time.time()
    try:
        found = strategy(QUESTION, top_k=5)
        sections = [c.section[:35] for c in found[:3]]
    except Exception as exc:
        sections = [f"error: {str(exc)[:80]}"]
    rows.append({"strategy": name, "seconds": round(time.time() - start, 1), "top 3 sections": sections})
pd.DataFrame(rows)
'''),
    md('''
---
## Section D · Mini evaluation

Opinions about answers don't scale; numbers do. `eval/golden_dataset.py` has 12 hand-labeled questions with the section that should be found. This scores a few of them:

| Metric | Meaning |
|---|---|
| `hit` | did an expected section appear in the top-k? |
| `mrr` | `1 / rank` of the first relevant chunk (1.0 = ranked first) |
| `recall` | share of the question's expected sections found |
| `correctly_refused` | for "not found" questions: did the app decline? |

`N` keeps it cheap. The full run, with optional LLM-judge scores for faithfulness and correctness, is `python -m eval.run_eval --pipeline advanced`.
'''),
    code('''
from eval.golden_dataset import GOLDEN_SET
from eval.run_eval import run_example, summarize

N = 5
examples = [e for e in GOLDEN_SET if e.should_find][: N - 1] + [e for e in GOLDEN_SET if not e.should_find][:1]
results = [run_example(e, collection=None, strategy="baseline", top_k=8, use_judge=False) for e in examples]

display(pd.DataFrame(results)[["id", "category", "hit", "mrr", "recall", "correctly_refused", "cost_usd"]])
summarize(results)
'''),
    md('''
---
## Recap and things to try

- **A:** the LLM mostly changes wording, speed and cost; retrieval decides what it can say.
- **B:** chunking matters most for tables.
- **C:** smarter strategies cost extra calls; check they find better sections before paying for them.
- **D:** measure before and after any change.

**Try this:** in Section D set `strategy="rerank"` and compare `mrr`; in Section B try *How did Services net sales compare to iPhone net sales in 2024?*

**Next:** `04_agentic_rag.ipynb`: let the LLM decide what to search for.
'''),
])

# =========================================================================== 04 agentic
save("04_agentic_rag.ipynb", [
    md('''
# 04 · Agentic RAG: let the model decide what to search

**You'll learn:** what changes when RAG becomes an *agent*, how this project's `core/agentic_rag.py` does it in about 150 lines, when it beats classic RAG, what it costs, and a checklist for converting your own RAG pipeline.

**Time:** about 10 minutes · **Cost:** about 5¢ · **Runs on its own:** the second cell builds any missing data.

| Part | What it covers |
|---|---|
| 1 | Classic vs agentic RAG |
| 2 | The building blocks: tool, decision format, loop, guardrails |
| 3 | Watch the agent work |
| 4 | Classic vs agentic, side by side |
| 5 | How to convert a RAG pipeline into an agent |
| 6 | Things to try |
'''),
    md('''
---
## Setup
''' + SETUP_NOTE),
    code(SETUP),
    code(READY),
    md('''
---
## Part 1 · Classic vs agentic RAG

**Classic RAG** (notebooks 02–03) searches **once**, with the user's exact words, and answers from whatever comes back:

```mermaid
flowchart LR
  Q["Question"] --> S["Search once<br/>with the question"] --> P["Prompt + LLM"] --> A["Answer"]
```

That breaks on questions with several parts ("compare X with Y, and tell me Z") or questions whose words don't match the filing's words. **Agentic RAG** lets the LLM drive the search: it decides what to search for, reads the results, and searches again until it has enough or runs out of budget:

```mermaid
flowchart TD
  Q["Question"] --> T{"LLM decides<br/>next action"}
  T -->|"search(query)"| S["hybrid_search(query)"]
  S --> R["Results added<br/>to the transcript"]
  R --> T
  T -->|"answer + cited ids"| V{"Cited ids were<br/>actually retrieved?"}
  V -->|"yes"| A["Answer + citations"]
  V -->|"no"| N["'Not found'"]
```

| | Classic (`core/rag.py`) | Agentic (`core/agentic_rag.py`) |
|---|---|---|
| Who writes the search query | the user's question, as is | the LLM, one focused query at a time |
| Searches per question | 1 | 1 to 4, decided by the LLM |
| LLM calls per question | 1 | one per decision (searches + 1) |
| Multi-part questions | one search must cover every part | one search per part |
| Cost and latency | lowest | higher, grows with each search |
| Grounding and refusal | cite retrieved chunks, else "not found" | the same rules |
'''),
    md('''
---
## Part 2 · The building blocks

### 2.1 The tool

The agent has exactly one tool: the same `hybrid_search()` the classic pipeline uses, returning 5 chunks per search. An agent is only as good as its tools; nothing about search itself changes.

### 2.2 The decision format

Each turn, the LLM must reply with **one JSON object**: either `{"action": "search", ...}` or `{"action": "answer", ...}`. Plain JSON, instead of a provider's native tool-calling API, keeps the same code working with OpenAI and Gemini. These are the real instructions:
'''),
    code('''
from core.agentic_rag import AGENT_INSTRUCTIONS, MAX_SEARCHES, SEARCH_TOP_K

print(AGENT_INSTRUCTIONS.format(max_searches=MAX_SEARCHES))
print(f"\\n(each search returns {SEARCH_TOP_K} chunks)")
'''),
    md('''
### 2.3 The loop

`answer_question_agentic()` keeps a running **transcript**: the question, then every search and its results. Each turn it:

1. sends the instructions + transcript + "searches left: N" to the LLM;
2. reads the JSON decision (a small parser tolerates code fences and extra text);
3. on **search**: runs `hybrid_search(query)`, appends the results to the transcript, and goes again;
4. on **answer**: keeps only cited ids that were really retrieved, and returns the answer with citations.

### 2.4 Guardrails

| Risk | Guardrail in the code |
|---|---|
| Endless searching | at most `max_searches` (4); after that the agent must answer |
| Repeating itself | a search it already made is refused, without using up the budget |
| Citing something it never saw | cited ids are checked against the chunks it actually retrieved |
| Answering from memory | no valid citation → the same "not found" message as classic RAG |
| Broken reply | invalid JSON → it is asked again; the loop has a hard turn limit |
'''),
    md('''
---
## Part 3 · Watch the agent work

A two-part question: margins live in one place in the filing, the geographic decline in another. The trace shows each search the agent chose and why.
'''),
    code('''
from core.agentic_rag import answer_question_agentic

AGENT_QUESTION = (
    "Compare Products gross margin to Services gross margin in 2024, "
    "and say which geographic segment had the largest net sales decline."
)
run = answer_question_agentic(AGENT_QUESTION)

display(pd.DataFrame([
    {"search": i + 1, "query the agent wrote": s.query, "why": s.reason, "chunks found": len(s.excerpt_ids)}
    for i, s in enumerate(run.steps)
]))
print("ANSWER:", run.result.answer, "\\n")
for c in run.result.citations:
    print(f"- {c.section[:80]}")
print(f"\\n{run.result.input_tokens} tokens in, {run.result.output_tokens} out, ≈ ${run.result.cost_usd:.4f}")
'''),
    md('''
**What to notice:** the agent rewrote the question into short, specific queries, one per topic, and each citation comes from a search it made.

---
## Part 4 · Classic vs agentic, side by side

Three kinds of question: a simple one, a multi-part one, and one the filing can't answer.
'''),
    code('''
import time

from core.rag import answer_question

questions = {
    "simple": "Why did Greater China net sales decrease in fiscal 2024?",
    "multi-part": AGENT_QUESTION,
    "not in filing": "What was Apple's revenue in fiscal year 2019?",
}

rows = []
for kind, q in questions.items():
    start = time.time()
    classic = answer_question(q)
    rows.append({"question": kind, "pipeline": "classic", "searches": 1, "citations": len(classic.citations),
                 "seconds": round(time.time() - start, 1), "cost_usd≈": round(classic.cost_usd, 4), "answer": classic.answer[:160]})
    start = time.time()
    agentic = answer_question_agentic(q)
    rows.append({"question": kind, "pipeline": "agentic", "searches": len(agentic.steps), "citations": len(agentic.result.citations),
                 "seconds": round(time.time() - start, 1), "cost_usd≈": round(agentic.result.cost_usd, 4), "answer": agentic.result.answer[:160]})
pd.DataFrame(rows)
'''),
    md('''
**How to read it:** on a simple question the agent often makes one search and matches classic RAG at about the same cost. Each extra search adds an LLM call, so cost and time grow with the number of searches (roughly 2× on a two-part question). It earns that on multi-part or loosely-worded questions. Both refuse the out-of-scope question. A common production pattern is to use classic RAG by default and route only complex questions to the agent.

---
## Part 5 · How to convert a RAG pipeline into an agent

The steps this project took, in order. Each maps to code in `core/agentic_rag.py`:

1. **Turn retrieval into a tool.** Keep your existing search function (`hybrid_search`) and give the LLM a way to call it with its own query.
2. **Define the decision format.** A small, strict menu of actions (here `search` and `answer`), returned as JSON. Native tool calling (OpenAI tools, Gemini function calling) is the provider-specific alternative.
3. **Write the loop.** Prompt → decision → run the tool → add the results to the transcript → repeat.
4. **Keep an evidence store.** Remember every chunk retrieved (`seen`), so citations can be validated against it.
5. **Keep your guardrails.** The search budget, refusing repeats, validating citations, and the same "not found" rule as before.
6. **Log the trace.** Return the steps (`AgentRun.steps`) so you can see why it answered what it did.
7. **Measure both.** Run classic and agentic on the same golden set and compare quality, latency and cost before switching.

```mermaid
flowchart LR
  A["Classic RAG"] --> B["1 · Retrieval as a tool"] --> C["2 · Decision format"] --> D["3 · Loop"] --> E["4–5 · Evidence + guardrails"] --> F["6–7 · Trace + measure"] --> G["Agentic RAG"]
```

**Natural next tools** to add to the menu: look up a section by name (`Item 1C`), a calculator for percentages, or a second collection (another company or year).
'''),
    md('''
---
## Part 6 · Things to try

- Ask a question that needs **three** facts and watch how many searches it makes.
- Lower the budget: `answer_question_agentic(AGENT_QUESTION, max_searches=1)`. Does the answer get worse?
- Point the agent at the basic chunks: `answer_question_agentic(AGENT_QUESTION, collection_name="sec_filings_basic")`.
'''),
    code('''
run = answer_question_agentic(AGENT_QUESTION, max_searches=1)
print([s.query for s in run.steps])
print(run.result.answer)
'''),
])
