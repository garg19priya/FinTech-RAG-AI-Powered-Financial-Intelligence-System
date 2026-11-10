import os
from dataclasses import dataclass

from core.retry import with_retry

# "openai" (default) or "gemini"; each provider has its own default model.
PROVIDER = (os.environ.get("LLM_PROVIDER") or "openai").lower()
_DEFAULT_MODEL = {"openai": "gpt-4o", "gemini": "gemini-3.5-flash"}
MODEL = os.environ.get("LLM_MODEL") or _DEFAULT_MODEL[PROVIDER]

# USD per 1M tokens, used for the cost estimate only. Unlisted models fall back to
# _PRICE_FALLBACK_MODEL's price for their provider, so update these if you track real spend.
_PRICE_PER_1M_INPUT = {
    "gpt-4o": 2.50,
    "gpt-4o-mini": 0.15,
    "gemini-2.5-flash": 0.30,
    "gemini-2.5-pro": 1.25,
}
_PRICE_PER_1M_OUTPUT = {
    "gpt-4o": 10.00,
    "gpt-4o-mini": 0.60,
    "gemini-2.5-flash": 2.50,
    "gemini-2.5-pro": 10.00,
}

_PRICE_FALLBACK_MODEL = {"openai": "gpt-4o", "gemini": "gemini-2.5-flash"}

_GEMINI_TIMEOUT_MS = 60_000  # fail instead of hanging when a Gemini model is overloaded

_clients: dict[str, object] = {}


@dataclass
class LLMResponse:
    text: str
    input_tokens: int
    output_tokens: int
    cost_usd: float


def _get_openai_client():
    if "openai" not in _clients:
        import openai

        _clients["openai"] = openai.OpenAI(api_key=os.environ["OPENAI_API_KEY"])
    return _clients["openai"]


def _get_gemini_client():
    if "gemini" not in _clients:
        from google import genai
        from google.genai import types

        _clients["gemini"] = genai.Client(
            api_key=os.environ["GEMINI_API_KEY"],
            http_options=types.HttpOptions(timeout=_GEMINI_TIMEOUT_MS),
        )
    return _clients["gemini"]


def _cost_usd(input_tokens: int, output_tokens: int) -> float:
    fallback = _PRICE_FALLBACK_MODEL[PROVIDER]
    return (
        input_tokens / 1_000_000 * _PRICE_PER_1M_INPUT.get(MODEL, _PRICE_PER_1M_INPUT[fallback])
        + output_tokens / 1_000_000 * _PRICE_PER_1M_OUTPUT.get(MODEL, _PRICE_PER_1M_OUTPUT[fallback])
    )


def _complete_openai(prompt: str, max_tokens: int) -> LLMResponse:
    client = _get_openai_client()
    response = with_retry(
        client.chat.completions.create,
        model=MODEL,
        max_tokens=max_tokens,
        messages=[{"role": "user", "content": prompt}],
    )
    usage = response.usage
    input_tokens = usage.prompt_tokens if usage else 0
    output_tokens = usage.completion_tokens if usage else 0
    return LLMResponse(
        text=response.choices[0].message.content,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        cost_usd=_cost_usd(input_tokens, output_tokens),
    )


def _complete_gemini(prompt: str, max_tokens: int) -> LLMResponse:
    from google.genai import types

    client = _get_gemini_client()
    response = with_retry(
        client.models.generate_content,
        model=MODEL,
        contents=prompt,
        # Thinking tokens count against max_output_tokens, so with thinking on, the
        # small budgets used by retrieval_strategies.py / eval/judge.py come back empty.
        config=types.GenerateContentConfig(
            max_output_tokens=max_tokens,
            thinking_config=types.ThinkingConfig(thinking_budget=0),
        ),
    )
    usage = response.usage_metadata
    input_tokens = (usage.prompt_token_count or 0) if usage else 0
    output_tokens = (usage.candidates_token_count or 0) if usage else 0
    return LLMResponse(
        text=response.text or "",
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        cost_usd=_cost_usd(input_tokens, output_tokens),
    )


def complete_with_usage(prompt: str, max_tokens: int = 1024) -> LLMResponse:
    """Call the configured LLM provider and return the response text plus token usage and estimated cost.

    Provider is chosen by the LLM_PROVIDER env var ("openai" or "gemini"); the model
    can be overridden via LLM_MODEL. Retries transient API errors with exponential backoff.
    """
    if PROVIDER == "gemini":
        return _complete_gemini(prompt, max_tokens)
    return _complete_openai(prompt, max_tokens)


def complete(prompt: str, max_tokens: int = 1024) -> str:
    """Call the configured LLM provider and return its raw text response.

    Provider/model can be overridden via the LLM_PROVIDER / LLM_MODEL env vars.
    """
    return complete_with_usage(prompt, max_tokens=max_tokens).text
