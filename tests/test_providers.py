from types import SimpleNamespace

import pytest

import core.embeddings as embeddings
import core.llm as llm
from core.retry import with_retry


class FakeGeminiModels:
    def __init__(self, dim=3):
        self.dim = dim
        self.embed_calls = []
        self.generate_calls = []

    def embed_content(self, **kwargs):
        self.embed_calls.append(kwargs)
        return SimpleNamespace(
            embeddings=[SimpleNamespace(values=[float(i)] * self.dim) for i, _ in enumerate(kwargs["contents"])]
        )

    def generate_content(self, **kwargs):
        self.generate_calls.append(kwargs)
        return SimpleNamespace(
            text="Answer.\nCITED: 0",
            usage_metadata=SimpleNamespace(prompt_token_count=1_000_000, candidates_token_count=1_000_000),
        )


@pytest.fixture
def fake_gemini(monkeypatch):
    models = FakeGeminiModels()
    client = SimpleNamespace(models=models)
    monkeypatch.setattr(embeddings, "_get_gemini_client", lambda: client)
    monkeypatch.setattr(llm, "_get_gemini_client", lambda: client)
    return models


def test_gemini_embeddings_batch_at_100_and_request_embedding_dim(monkeypatch, fake_gemini):
    monkeypatch.setattr(embeddings, "PROVIDER", "gemini")
    monkeypatch.setattr(embeddings, "MODEL", "gemini-embedding-001")

    vectors = embeddings.embed_dense([f"t{i}" for i in range(250)])

    assert len(vectors) == 250
    assert [len(c["contents"]) for c in fake_gemini.embed_calls] == [100, 100, 50]
    assert all(c["model"] == "gemini-embedding-001" for c in fake_gemini.embed_calls)
    assert fake_gemini.embed_calls[0]["config"].output_dimensionality == embeddings.EMBEDDING_DIM


def test_gemini_completion_returns_text_tokens_and_cost(monkeypatch, fake_gemini):
    monkeypatch.setattr(llm, "PROVIDER", "gemini")
    monkeypatch.setattr(llm, "MODEL", "gemini-3.5-flash")

    response = llm.complete_with_usage("prompt", max_tokens=256)

    assert response.text == "Answer.\nCITED: 0"
    assert (response.input_tokens, response.output_tokens) == (1_000_000, 1_000_000)
    assert response.cost_usd == pytest.approx(0.30 + 2.50)  # unlisted model -> gemini-2.5-flash price
    call = fake_gemini.generate_calls[0]
    assert call["model"] == "gemini-3.5-flash"
    assert call["contents"] == "prompt"
    assert call["config"].max_output_tokens == 256


def test_openai_remains_default_provider(monkeypatch):
    assert llm.PROVIDER == "openai" and llm.MODEL == "gpt-4o"
    assert embeddings.PROVIDER == "openai" and embeddings.MODEL == "text-embedding-3-small"


def test_retry_retries_gemini_rate_limit_but_not_other_errors(monkeypatch):
    from google.genai import errors

    monkeypatch.setattr("core.retry.time.sleep", lambda _: None)
    attempts = []

    def flaky():
        attempts.append(1)
        if len(attempts) < 3:
            raise errors.ClientError(429, {"error": {"message": "rate limited"}})
        return "ok"

    assert with_retry(flaky) == "ok"
    assert len(attempts) == 3

    def bad_request():
        raise errors.ClientError(400, {"error": {"message": "bad"}})

    with pytest.raises(errors.ClientError):
        with_retry(bad_request)
