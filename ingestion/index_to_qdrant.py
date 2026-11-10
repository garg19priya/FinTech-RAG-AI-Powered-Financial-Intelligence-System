import json
import logging
from pathlib import Path

from qdrant_client import models

from core.clients import COLLECTION_NAME, get_qdrant_client, get_sparse_model
from core.embeddings import EMBEDDING_DIM
from core.embeddings import embed_dense as _embed_dense
from core.models import FilingChunk
from ingestion.fetch_filings import COMPANIES

logger = logging.getLogger(__name__)

# Defaults match COLLECTION_NAME ("sec_filings_advanced"); index_comparison_collections.py
# passes both chunk dirs explicitly.
CHUNKS_DIR = Path("data/chunks_advanced")


def ensure_collection(collection_name: str | None = None) -> None:
    collection_name = collection_name or COLLECTION_NAME
    client = get_qdrant_client()
    if client.collection_exists(collection_name):
        logger.info("Collection %r already exists", collection_name)
        return
    logger.info("Creating collection %r", collection_name)
    client.create_collection(
        collection_name=collection_name,
        vectors_config={
            "dense": models.VectorParams(size=EMBEDDING_DIM, distance=models.Distance.COSINE),
        },
        sparse_vectors_config={
            "sparse": models.SparseVectorParams(),
        },
    )


def embed_dense(texts: list[str]) -> list[list[float]]:
    return _embed_dense(texts)


def load_chunks(ticker: str, chunks_dir: Path | None = None) -> list[FilingChunk]:
    chunks_dir = chunks_dir or CHUNKS_DIR
    path = chunks_dir / f"{ticker}.json"
    raw = json.loads(path.read_text(encoding="utf-8"))
    return [FilingChunk(**item) for item in raw]


def index_all_filings(collection_name: str | None = None, chunks_dir: Path | None = None) -> None:
    collection_name = collection_name or COLLECTION_NAME
    chunks_dir = chunks_dir or CHUNKS_DIR
    client = get_qdrant_client()
    ensure_collection(collection_name)
    sparse_model = get_sparse_model()

    point_id = 0
    for ticker in COMPANIES:
        chunks = load_chunks(ticker, chunks_dir)
        logger.info("Loaded %d chunks for %s from %s", len(chunks), ticker, chunks_dir)
        texts = [c.text for c in chunks]
        dense_vectors = embed_dense(texts)
        sparse_vectors = list(sparse_model.embed(texts))
        logger.info("Embedded %d chunks for %s (dense + sparse)", len(texts), ticker)

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
        client.upsert(collection_name=collection_name, points=points)
        logger.info("Upserted %d points for %s into %r", len(points), ticker, collection_name)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
    index_all_filings()
