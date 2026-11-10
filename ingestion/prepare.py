"""One-command ingestion: fetch, chunk and index only what is missing.

    python -m ingestion.prepare              # build whatever is missing
    python -m ingestion.prepare --reindex    # re-embed after changing EMBEDDING_PROVIDER/MODEL
    python -m ingestion.prepare --rechunk    # re-run docling too (slow)

Notebooks call prepare_data() in their setup so each one runs on its own.
"""
import argparse
import logging
from pathlib import Path

from core.clients import get_qdrant_client
from ingestion.chunk_filings_advanced import CHUNKS_DIR as ADVANCED_CHUNKS_DIR
from ingestion.chunk_filings_advanced import chunk_all_filings_advanced
from ingestion.chunk_filings_basic import CHUNKS_DIR as BASIC_CHUNKS_DIR
from ingestion.chunk_filings_basic import chunk_all_filings_basic
from ingestion.fetch_filing_pdf import fetch_all_filing_pdfs
from ingestion.fetch_filings import COMPANIES
from ingestion.index_comparison_collections import (
    ADVANCED_COLLECTION,
    BASIC_COLLECTION,
    index_comparison_collections,
)

logger = logging.getLogger(__name__)


def _chunks_exist(chunks_dir: Path) -> bool:
    return all((chunks_dir / f"{ticker}.json").exists() for ticker in COMPANIES)


def prepare_data(rechunk: bool = False, reindex: bool = False) -> dict[str, int]:
    """Make sure the PDFs, both chunk sets and both Qdrant collections exist.

    Each step is skipped when its output is already there, so calling this is cheap
    after the first run. Returns {collection_name: point_count}.
    """
    fetch_all_filing_pdfs()  # cached per ticker

    if rechunk or not _chunks_exist(BASIC_CHUNKS_DIR):
        chunk_all_filings_basic()
    if rechunk or not _chunks_exist(ADVANCED_CHUNKS_DIR):
        logger.info("Running docling (slow on first run: downloads layout models)")
        chunk_all_filings_advanced()

    client = get_qdrant_client()
    collections = (BASIC_COLLECTION, ADVANCED_COLLECTION)
    if rechunk or reindex or not all(client.collection_exists(name) for name in collections):
        for name in collections:
            if client.collection_exists(name):
                client.delete_collection(name)
        index_comparison_collections()

    return {name: client.count(name).count for name in collections}


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
    parser = argparse.ArgumentParser(description="Fetch, chunk and index only what is missing.")
    parser.add_argument("--rechunk", action="store_true", help="re-run both chunkers (docling is slow)")
    parser.add_argument("--reindex", action="store_true", help="re-embed and re-index both collections")
    args = parser.parse_args()
    for name, count in prepare_data(rechunk=args.rechunk, reindex=args.reindex).items():
        print(f"{name}: {count} points")
