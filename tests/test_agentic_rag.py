import json

import core.agentic_rag as agent
from core.llm import LLMResponse
from core.models import RetrievedChunk
from core.rag import NOT_FOUND_MESSAGE


def _chunk(chunk_id, section):
    return RetrievedChunk(
        id=chunk_id, ticker="AAPL", company_name="Apple Inc.", filing_type="10-K",
        section=section, fiscal_year="2024", text=f"text of chunk {chunk_id}", score=1.0,
    )


def _script(monkeypatch, replies, search_results):
    """Feed the agent canned LLM replies and canned search results; record its searches."""
    replies = iter(replies)
    searches = []

    def fake_complete(prompt, max_tokens=1024):
        return LLMResponse(text=next(replies), input_tokens=10, output_tokens=5, cost_usd=0.001)

    def fake_search(query, top_k=5, collection_name=None):
        searches.append(query)
        return search_results[len(searches) - 1]

    monkeypatch.setattr(agent, "complete_with_usage", fake_complete)
    monkeypatch.setattr(agent, "hybrid_search", fake_search)
    return searches


def test_searches_twice_then_answers_with_valid_citations(monkeypatch):
    searches = _script(
        monkeypatch,
        [
            json.dumps({"action": "search", "query": "Products gross margin", "reason": "part 1"}),
            "```json\n" + json.dumps({"action": "search", "query": "Services gross margin", "reason": "part 2"}) + "\n```",
            json.dumps({"action": "answer", "answer": "Services is higher.", "cited": [1, 2, 99]}),
        ],
        [[_chunk(1, "Item 7")], [_chunk(2, "Item 8")]],
    )

    run = agent.answer_question_agentic("Compare Products and Services gross margin")

    assert searches == ["Products gross margin", "Services gross margin"]
    assert [s.excerpt_ids for s in run.steps] == [[1], [2]]
    assert run.result.answer == "Services is higher."
    assert [c.section for c in run.result.citations] == ["Item 7", "Item 8"]  # 99 was never retrieved
    assert run.result.input_tokens == 30 and run.result.output_tokens == 15


def test_no_valid_citation_returns_not_found(monkeypatch):
    _script(
        monkeypatch,
        [
            json.dumps({"action": "search", "query": "revenue 2019"}),
            json.dumps({"action": "answer", "answer": "Not in the excerpts.", "cited": []}),
        ],
        [[_chunk(5, "Item 7")]],
    )

    run = agent.answer_question_agentic("What was Apple's revenue in fiscal year 2019?")

    assert run.result.answer == NOT_FOUND_MESSAGE
    assert run.result.citations == []


def test_search_budget_is_enforced(monkeypatch):
    searches = _script(
        monkeypatch,
        [json.dumps({"action": "search", "query": f"q{i}"}) for i in range(3)]
        + [json.dumps({"action": "answer", "answer": "done", "cited": [1]})],
        [[_chunk(1, "Item 1")], [_chunk(2, "Item 2")]],
    )

    run = agent.answer_question_agentic("anything", max_searches=2)

    assert searches == ["q0", "q1"]  # third search request refused
    assert run.result.answer == "done"


def test_unparseable_replies_end_in_not_found(monkeypatch):
    _script(monkeypatch, ["not json"] * 10, [])

    run = agent.answer_question_agentic("anything", max_searches=1)

    assert run.result.answer == NOT_FOUND_MESSAGE
    assert run.steps == []


def test_repeated_query_is_not_searched_again(monkeypatch):
    searches = _script(
        monkeypatch,
        [
            json.dumps({"action": "search", "query": "revenue 2019"}),
            json.dumps({"action": "search", "query": "Revenue 2019 "}),
            json.dumps({"action": "answer", "answer": "none", "cited": []}),
        ],
        [[_chunk(1, "Item 7")]],
    )

    run = agent.answer_question_agentic("What was Apple's revenue in fiscal year 2019?")

    assert searches == ["revenue 2019"]
    assert len(run.steps) == 1
    assert run.result.answer == NOT_FOUND_MESSAGE
