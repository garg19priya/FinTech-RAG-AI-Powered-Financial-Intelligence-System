import logging

from qdrant_client import models

from core.clients import COLLECTION_NAME, get_qdrant_client, get_sparse_model
from core.embeddings import embed_dense
from core.models import RetrievedChunk

logger = logging.getLogger(__name__)


def embed_query_dense(query: str) -> list[float]:
    return embed_dense([query])[0]


def embed_query_sparse(query: str) -> models.SparseVector:
    embedding = next(get_sparse_model().embed([query]))
    return models.SparseVector(
        indices=embedding.indices.tolist(), values=embedding.values.tolist()
    )


def hybrid_search(
    query: str, top_k: int = 8, collection_name: str | None = None
) -> list[RetrievedChunk]:
    """Run hybrid (dense + sparse) search over the given collection, fused via RRF."""
    collection_name = collection_name or COLLECTION_NAME
    logger.info("hybrid_search: collection=%r top_k=%d query=%r", collection_name, top_k, query)
    dense_vector = embed_query_dense(query)
    sparse_vector = embed_query_sparse(query)

    results = get_qdrant_client().query_points(
        collection_name=collection_name,
        prefetch=[
            models.Prefetch(query=dense_vector, using="dense", limit=top_k * 2),
            models.Prefetch(query=sparse_vector, using="sparse", limit=top_k * 2),
        ],
        query=models.FusionQuery(fusion=models.Fusion.RRF),
        limit=top_k,
    )
    logger.info("hybrid_search: retrieved %d points from %r", len(results.points), collection_name)

    return [
        RetrievedChunk(
            id=point.id,
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
