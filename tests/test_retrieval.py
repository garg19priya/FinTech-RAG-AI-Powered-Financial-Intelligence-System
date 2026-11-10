import os

import pytest
import voyageai
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
        "text": "Apple does not report metrics used by banks to assess deposit and loan spreads.",
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
    from core.clients import get_qdrant_client, get_sparse_model

    client = get_qdrant_client()
    monkeypatch.setattr(mod, "COLLECTION_NAME", TEST_COLLECTION)

    if client.collection_exists(TEST_COLLECTION):
        client.delete_collection(TEST_COLLECTION)
    client.create_collection(
        collection_name=TEST_COLLECTION,
        vectors_config={"dense": models.VectorParams(size=1024, distance=models.Distance.COSINE)},
        sparse_vectors_config={"sparse": models.SparseVectorParams()},
    )

    voyage = voyageai.Client(api_key=os.environ["VOYAGE_API_KEY"])
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
