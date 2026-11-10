import time
from typing import Callable, TypeVar

T = TypeVar("T")


def _is_retryable(exc: Exception) -> bool:
    import openai

    if isinstance(
        exc,
        (
            openai.APIConnectionError,
            openai.RateLimitError,
            openai.APITimeoutError,
            openai.InternalServerError,
        ),
    ):
        return True

    try:
        from google.genai import errors as genai_errors
    except ImportError:
        return False
    if isinstance(exc, genai_errors.ServerError):
        return True
    return isinstance(exc, genai_errors.ClientError) and exc.code == 429


def _gemini_retry_delay(exc: Exception) -> float | None:
    """Seconds Gemini asks us to wait on a 429 (its RetryInfo "retryDelay", e.g. "51s"), else None."""
    try:
        from google.genai import errors as genai_errors
    except ImportError:
        return None
    if not (isinstance(exc, genai_errors.ClientError) and exc.code == 429):
        return None
    try:
        for detail in exc.details["error"]["details"]:
            if detail.get("@type", "").endswith("RetryInfo"):
                return float(detail["retryDelay"].rstrip("s"))
    except (AttributeError, KeyError, TypeError, ValueError):
        pass
    return 60.0


def with_retry(
    fn: Callable[..., T], *args, max_retries: int = 3, backoff_base: float = 1.0, **kwargs
) -> T:
    """Call fn(*args, **kwargs), retrying on transient OpenAI/Gemini API errors with exponential backoff.

    A Gemini 429 (free-tier per-minute quota) waits the delay Gemini asks for, and gets
    extra attempts, since the quota resets within a minute.
    """
    last_exc: Exception | None = None
    attempt = 0
    rate_limit_waits = 0
    while attempt < max_retries:
        try:
            return fn(*args, **kwargs)
        except Exception as exc:
            if not _is_retryable(exc):
                raise
            last_exc = exc
            delay = _gemini_retry_delay(exc)
            if delay is not None and rate_limit_waits < 5:
                rate_limit_waits += 1
                time.sleep(delay + 1)
                continue
            attempt += 1
            if attempt < max_retries:
                time.sleep(backoff_base * (2 ** (attempt - 1)))
    assert last_exc is not None
    raise last_exc
