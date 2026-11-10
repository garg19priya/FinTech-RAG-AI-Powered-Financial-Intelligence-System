from unittest.mock import patch

from core.llm import LLMResponse
from core.models import Citation, RAGAnswer, RetrievedChunk
import eval.judge as judge_mod
import eval.run_eval as run_eval_mod
from eval.golden_dataset import GOLDEN_SET, EvalExample
from eval.retrieval_metrics import hit_rate, recall_at_k, reciprocal_rank


def _chunk(ticker="AAPL", section="Item 7. MD&A", score=0.9):
    return RetrievedChunk(
        id=0,
        ticker=ticker,
        company_name="Apple Inc.",
        filing_type="10-K",
        section=section,
        fiscal_year="2024",
        text="some filing text",
        score=score,
    )


def _example(expected_sections=("Item 7",), expected_ticker="AAPL"):
    return EvalExample(
        id="test",
        category="exact_term",
        question="q",
        expected_ticker=expected_ticker,
        expected_sections=list(expected_sections),
        reference_answer="ref",
    )


# --- retrieval_metrics ---


def test_hit_rate_true_when_matching_chunk_present():
    example = _example()
    chunks = [_chunk(section="Item 1A"), _chunk(section="Item 7.    MD&A")]
    assert hit_rate(chunks, example) is True


def test_hit_rate_false_when_no_matching_chunk():
    example = _example()
    chunks = [_chunk(section="Item 1A"), _chunk(ticker="JPM", section="Item 7")]
    assert hit_rate(chunks, example) is False


def test_reciprocal_rank_scores_by_position_of_first_match():
    example = _example()
    chunks = [_chunk(section="Item 1A"), _chunk(section="Item 7. MD&A")]
    assert reciprocal_rank(chunks, example) == 0.5


def test_reciprocal_rank_zero_when_no_match():
    example = _example()
    chunks = [_chunk(section="Item 1A")]
    assert reciprocal_rank(chunks, example) == 0.0


def test_recall_at_k_gives_partial_credit_for_multi_section_example():
    example = _example(expected_sections=["Item 7", "Item 8"])
    chunks = [_chunk(section="Item 7. MD&A")]
    assert recall_at_k(chunks, example, k=8) == 0.5


def test_recall_at_k_full_credit_when_all_sections_present():
    example = _example(expected_sections=["Item 7", "Item 8"])
    chunks = [_chunk(section="Item 7. MD&A"), _chunk(section="Item 8. Financial Statements")]
    assert recall_at_k(chunks, example, k=8) == 1.0


# --- judge ---


def test_judge_faithfulness_parses_well_formed_score():
    chunks = [_chunk(section="Item 7. MD&A")]
    with patch.object(
        judge_mod,
        "complete_with_usage",
        return_value=LLMResponse(text="Grounded in the excerpt.\nSCORE: 5", input_tokens=10, output_tokens=5, cost_usd=0.001),
    ):
        result = judge_mod.judge_faithfulness("answer text", chunks)

    assert result.score == 5
    assert result.rationale == "Grounded in the excerpt."
    assert result.cost_usd == 0.001


def test_judge_correctness_returns_none_score_when_unparseable():
    with patch.object(
        judge_mod,
        "complete_with_usage",
        return_value=LLMResponse(text="I'm not sure how to rate this.", input_tokens=10, output_tokens=5, cost_usd=0.0),
    ):
        result = judge_mod.judge_correctness("q", "ref", "answer")

    assert result.score is None


# --- run_eval.run_example ---


def test_run_example_marks_correct_refusal_and_skips_judge():
    example = EvalExample(
        id="not_found", category="not_found", question="q",
        expected_ticker=None, expected_sections=[], reference_answer=None, should_find=False,
    )
    refused_answer = RAGAnswer(question="q", answer="not found", citations=[], cost_usd=0.002)

    with patch.object(run_eval_mod, "answer_question", return_value=refused_answer), patch.object(
        run_eval_mod, "judge_faithfulness"
    ) as mock_faith, patch.object(run_eval_mod, "judge_correctness") as mock_correct:
        row = run_eval_mod.run_example(example, "sec_filings_basic", "baseline", 8, True)

    assert row["should_find"] is False
    assert row["correctly_refused"] is True
    assert row["recall"] is None
    mock_faith.assert_not_called()
    mock_correct.assert_not_called()


def test_run_example_populates_metrics_for_findable_example():
    example = _example()
    chunk = _chunk(section="Item 7. MD&A")
    citation = Citation(ticker="AAPL", company_name="Apple", section="Item 7", fiscal_year="2024", text_snippet="x")
    answer = RAGAnswer(question="q", answer="the answer", citations=[citation], cost_usd=0.003)
    faith_result = judge_mod.JudgeResult(score=4, rationale="ok", cost_usd=0.0005)
    correct_result = judge_mod.JudgeResult(score=5, rationale="ok", cost_usd=0.0005)

    fake_strategies = {"baseline": lambda *a, **k: [chunk]}
    with patch.object(run_eval_mod, "answer_question", return_value=answer), patch.object(
        run_eval_mod, "STRATEGIES", fake_strategies
    ), patch.object(run_eval_mod, "judge_faithfulness", return_value=faith_result), patch.object(
        run_eval_mod, "judge_correctness", return_value=correct_result
    ):
        row = run_eval_mod.run_example(example, "sec_filings_basic", "baseline", 8, True)

    assert row["hit"] is True
    assert row["mrr"] == 1.0
    assert row["recall"] == 1.0
    assert row["faithfulness"] == 4
    assert row["correctness"] == 5
    assert row["cost_usd"] == 0.003 + 0.0005 + 0.0005


# --- golden set sanity ---


def test_golden_set_covers_all_categories():
    categories = {e.category for e in GOLDEN_SET}
    assert categories == {"exact_term", "segment_numerical", "semantic", "cross_section", "not_found"}


def test_not_found_examples_have_no_reference_answer():
    for example in GOLDEN_SET:
        if example.category == "not_found":
            assert example.should_find is False
            assert example.reference_answer is None
