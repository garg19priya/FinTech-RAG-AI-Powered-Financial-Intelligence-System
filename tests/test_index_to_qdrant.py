import json
from unittest.mock import patch, MagicMock

import pytest
from qdrant_client import QdrantClient

from ingestion import index_to_qdrant as mod


@pytest.fixture
def qdrant_client():
    client = QdrantClient(url="http://localhost:6333")
    try:
        client.get_collections()
    except Exception:
        pytest.skip("needs the Docker Qdrant server: docker compose up -d")
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

    fake_dense = [[0.1] * 1536, [0.2] * 1536]

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

    points = qdrant_client.retrieve("test_collection", ids=[0], with_vectors=True)
    assert len(points) == 1
    assert "dense" in points[0].vector
    assert "sparse" in points[0].vector
