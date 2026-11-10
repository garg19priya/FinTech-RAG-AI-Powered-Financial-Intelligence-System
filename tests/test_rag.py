from unittest.mock import patch

from core.llm import LLMResponse
from core.models import RetrievedChunk
import core.rag as mod


def _make_chunk(ticker="JPM", text="Net interest margin was 2.7%.", section="Item 7", score=0.02):
    return RetrievedChunk(
        id=0,
        ticker=ticker,
        company_name="JPMorgan Chase & Co.",
        filing_type="10-K",
        section=section,
        fiscal_year="2024",
        text=text,
        score=score,
    )


def test_answer_question_returns_grounded_answer_with_citations():
    chunk = _make_chunk()

    llm_response = LLMResponse(
        text="The net interest margin was 2.7% for the fiscal year.\nCITED: 0",
        input_tokens=100,
        output_tokens=20,
        cost_usd=0.001,
    )

    with patch.object(mod, "hybrid_search", return_value=[chunk]), patch.object(
        mod, "complete_with_usage", return_value=llm_response
    ):
        result = mod.answer_question("What was JPMorgan's net interest margin?")

    assert "2.7%" in result.answer
    assert len(result.citations) == 1
    assert result.citations[0].ticker == "JPM"
    assert result.input_tokens == 100
    assert result.output_tokens == 20
    assert result.cost_usd == 0.001


def test_answer_question_returns_not_found_when_no_chunks_retrieved():
    with patch.object(mod, "hybrid_search", return_value=[]):
        result = mod.answer_question("What is the capital of France?")

    assert "couldn't find" in result.answer
    assert result.citations == []


def test_answer_question_returns_not_found_when_model_cites_nothing():
    chunk = _make_chunk()

    llm_response = LLMResponse(
        text="The excerpts don't contain this information.\nCITED:",
        input_tokens=80,
        output_tokens=15,
        cost_usd=0.0008,
    )

    with patch.object(mod, "hybrid_search", return_value=[chunk]), patch.object(
        mod, "complete_with_usage", return_value=llm_response
    ):
        result = mod.answer_question("What is the capital of France?")

    assert "couldn't find" in result.answer
    assert result.citations == []
    assert result.input_tokens == 80
