import atexit
import os

from fastembed import SparseTextEmbedding
from qdrant_client import QdrantClient

QDRANT_URL = os.environ.get("QDRANT_URL", "http://localhost:6333")
# Optional embedded mode: set QDRANT_PATH (e.g. "data/qdrant_local") to run Qdrant inside
# this Python process, storing data in that folder, with no Docker. Takes precedence over
# QDRANT_URL. Only one process can open the folder at a time.
QDRANT_PATH = os.environ.get("QDRANT_PATH") or None
SPARSE_MODEL = "Qdrant/bm25"


def collection_for(base: str) -> str:
    """Collection name for the active embedding provider.

    Each provider has its own vector space, so non-OpenAI providers get a suffix
    (e.g. "sec_filings_advanced_gemini") and never mix with the OpenAI collections.
    """
    provider = (os.environ.get("EMBEDDING_PROVIDER") or "openai").lower()
    return base if provider == "openai" else f"{base}_{provider}"


COLLECTION_NAME = collection_for("sec_filings_advanced")

_qdrant_client: QdrantClient | None = None
_sparse_model: SparseTextEmbedding | None = None


def get_qdrant_client() -> QdrantClient:
    global _qdrant_client
    if _qdrant_client is None:
        if QDRANT_PATH:
            _qdrant_client = QdrantClient(path=QDRANT_PATH)
            # Release the folder lock cleanly instead of during interpreter teardown.
            atexit.register(_qdrant_client.close)
        else:
            _qdrant_client = QdrantClient(url=QDRANT_URL)
    return _qdrant_client


def get_sparse_model() -> SparseTextEmbedding:
    global _sparse_model
    if _sparse_model is None:
        _sparse_model = SparseTextEmbedding(model_name=SPARSE_MODEL)
    return _sparse_model
