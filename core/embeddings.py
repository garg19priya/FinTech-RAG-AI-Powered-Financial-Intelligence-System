import os
import time

from tqdm.auto import tqdm

from core.retry import with_retry

EMBEDDING_DIM = 1536  # must match the Qdrant "dense" vector size; see ingestion/index_to_qdrant.py

# "openai" (default) or "gemini". Switching provider changes the vector space, so
# re-run ingestion/index_comparison_collections.py after changing it.
PROVIDER = (os.environ.get("EMBEDDING_PROVIDER") or "openai").lower()
_DEFAULT_MODEL = {"openai": "text-embedding-3-small", "gemini": "gemini-embedding-001"}
MODEL = os.environ.get("EMBEDDING_MODEL") or _DEFAULT_MODEL[PROVIDER]

_GEMINI_BATCH_TOKENS = 15_000  # keep each request well under the per-minute quota
_GEMINI_TOKENS_PER_MINUTE = 25_000
_GEMINI_MAX_BATCH = 100  # Gemini embed_content accepts at most 100 inputs per request

_clients: dict[str, object] = {}


def _get_openai_client():
    if "openai" not in _clients:
        import openai

        _clients["openai"] = openai.OpenAI(api_key=os.environ["OPENAI_API_KEY"])
    return _clients["openai"]


def _get_gemini_client():
    if "gemini" not in _clients:
        from google import genai

        _clients["gemini"] = genai.Client(api_key=os.environ["GEMINI_API_KEY"])
    return _clients["gemini"]


def _embed_openai(texts: list[str]) -> list[list[float]]:
    client = _get_openai_client()
    response = with_retry(
        client.embeddings.create, model=MODEL, input=texts, dimensions=EMBEDDING_DIM
    )
    return [item.embedding for item in response.data]


def _embed_content_paced(client, *, contents, config):
    """embed_content that waits out the free-tier tokens-per-minute quota (429) instead of failing."""
    from google.genai import errors as genai_errors

    for attempt in range(8):
        try:
            return with_retry(
                client.models.embed_content, model=MODEL, contents=contents, config=config
            )
        except genai_errors.ClientError as exc:
            if exc.code != 429 or attempt == 7:
                raise
            time.sleep(40)


def _embed_gemini(texts: list[str]) -> list[list[float]]:
    from google.genai import types

    client = _get_gemini_client()
    config = types.EmbedContentConfig(output_dimensionality=EMBEDDING_DIM)
    vectors: list[list[float]] = []
    batch: list[str] = []
    batch_tokens = 0
    # Free tier is rate limited, so this can take minutes: show progress.
    progress = tqdm(total=len(texts), desc="Embedding with Gemini", unit="chunk")

    def flush() -> None:
        nonlocal batch, batch_tokens
        if not batch:
            return
        response = _embed_content_paced(client, contents=batch, config=config)
        vectors.extend(embedding.values for embedding in response.embeddings)
        progress.update(len(batch))
        # Stay under the free-tier quota of 30k embedding tokens per minute.
        time.sleep(60 * batch_tokens / _GEMINI_TOKENS_PER_MINUTE)
        batch, batch_tokens = [], 0

    for text in texts:
        tokens = len(text) // 3 + 1  # rough estimate; real count is lower for English
        if batch and (len(batch) >= _GEMINI_MAX_BATCH or batch_tokens + tokens > _GEMINI_BATCH_TOKENS):
            flush()
        batch.append(text)
        batch_tokens += tokens
    flush()
    progress.close()
    return vectors


def embed_dense(texts: list[str]) -> list[list[float]]:
    """Embed texts with the configured provider (EMBEDDING_PROVIDER: "openai" or "gemini").

    Embeddings are requested at EMBEDDING_DIM so either provider fits the same Qdrant
    schema; the caller does not need to pass an input type. The model can be overridden
    via the EMBEDDING_MODEL env var. Retries transient API errors with exponential backoff.
    """
    if PROVIDER == "gemini":
        return _embed_gemini(texts)
    return _embed_openai(texts)
