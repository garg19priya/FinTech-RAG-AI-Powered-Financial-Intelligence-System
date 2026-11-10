from core.embeddings import embed_dense


class FakeEmbeddingsResponse:
    def __init__(self, embeddings):
        self.data = [type("Item", (), {"embedding": embedding})() for embedding in embeddings]


class FakeClient:
    def __init__(self):
        self.embeddings = self
        self.calls = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        return FakeEmbeddingsResponse([[0.1, 0.2], [0.3, 0.4]])


def test_embed_dense_does_not_require_input_type(monkeypatch):
    fake_client = FakeClient()

    monkeypatch.setattr("core.embeddings._get_openai_client", lambda: fake_client)

    embeddings = embed_dense(["one", "two"])

    assert embeddings == [[0.1, 0.2], [0.3, 0.4]]
    assert fake_client.calls[0]["input"] == ["one", "two"]
