from types import SimpleNamespace

import ingestion.prepare as prepare


class FakeClient:
    def __init__(self, existing):
        self.existing = set(existing)
        self.deleted = []

    def collection_exists(self, name):
        return name in self.existing

    def delete_collection(self, name):
        self.deleted.append(name)
        self.existing.discard(name)

    def count(self, name):
        return SimpleNamespace(count=7)


def _patch(monkeypatch, client, chunks_exist):
    calls = []
    monkeypatch.setattr(prepare, "fetch_all_filing_pdfs", lambda: calls.append("fetch"))
    monkeypatch.setattr(prepare, "chunk_all_filings_basic", lambda: calls.append("basic"))
    monkeypatch.setattr(prepare, "chunk_all_filings_advanced", lambda: calls.append("advanced"))
    monkeypatch.setattr(prepare, "index_comparison_collections", lambda: calls.append("index"))
    monkeypatch.setattr(prepare, "_chunks_exist", lambda _dir: chunks_exist)
    monkeypatch.setattr(prepare, "get_qdrant_client", lambda: client)
    return calls


def test_skips_everything_already_built(monkeypatch):
    client = FakeClient({prepare.BASIC_COLLECTION, prepare.ADVANCED_COLLECTION})
    calls = _patch(monkeypatch, client, chunks_exist=True)

    counts = prepare.prepare_data()

    assert calls == ["fetch"]
    assert counts == {prepare.BASIC_COLLECTION: 7, prepare.ADVANCED_COLLECTION: 7}


def test_builds_missing_chunks_and_collections(monkeypatch):
    client = FakeClient({prepare.BASIC_COLLECTION})
    calls = _patch(monkeypatch, client, chunks_exist=False)

    prepare.prepare_data()

    assert calls == ["fetch", "basic", "advanced", "index"]
    assert client.deleted == [prepare.BASIC_COLLECTION]


def test_reindex_rebuilds_existing_collections(monkeypatch):
    client = FakeClient({prepare.BASIC_COLLECTION, prepare.ADVANCED_COLLECTION})
    calls = _patch(monkeypatch, client, chunks_exist=True)

    prepare.prepare_data(reindex=True)

    assert calls == ["fetch", "index"]
    assert client.deleted == [prepare.BASIC_COLLECTION, prepare.ADVANCED_COLLECTION]
