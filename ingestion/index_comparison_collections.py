from pathlib import Path

from core.clients import collection_for
from ingestion.index_to_qdrant import index_all_filings

BASIC_COLLECTION = collection_for("sec_filings_basic")
ADVANCED_COLLECTION = collection_for("sec_filings_advanced")


def index_comparison_collections() -> None:
    print(f"[1/2] Indexing {BASIC_COLLECTION}")
    index_all_filings(collection_name=BASIC_COLLECTION, chunks_dir=Path("data/chunks_basic"))
    print(f"[2/2] Indexing {ADVANCED_COLLECTION}")
    index_all_filings(collection_name=ADVANCED_COLLECTION, chunks_dir=Path("data/chunks_advanced"))
    print("Done.")


if __name__ == "__main__":
    index_comparison_collections()
