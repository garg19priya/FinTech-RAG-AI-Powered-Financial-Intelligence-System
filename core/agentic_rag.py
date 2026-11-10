"""Agentic RAG: the LLM decides what to search for, and how often, before it answers.

Classic RAG (core/rag.py) searches once with the user's exact question. Here the LLM
runs a short loop instead:

    read the question -> search(query) -> read the results -> search again, or answer

Each turn the LLM replies with one JSON object ({"action": "search", ...} or
{"action": "answer", ...}). Plain JSON instead of provider-specific tool calling keeps
it working the same with OpenAI and Gemini. Guardrails: at most `max_searches`
searches, citations must be excerpts the agent actually retrieved, and no valid
citation means the same "not found" refusal as classic RAG.
"""
import json
import logging
import re
from dataclasses import dataclass, field

from core.llm import complete_with_usage
from core.models import Citation, RAGAnswer, RetrievedChunk
from core.rag import NOT_FOUND_MESSAGE
from core.retrieval import hybrid_search

logger = logging.getLogger(__name__)

MAX_SEARCHES = 4
SEARCH_TOP_K = 5

AGENT_INSTRUCTIONS = """You answer questions about companies' SEC 10-K filings. You cannot see the
filings directly; you have one tool, a search over filing excerpts.

Reply with exactly ONE JSON object and nothing else, in one of these two forms:
{{"action": "search", "query": "<a focused search query>", "reason": "<why this search>"}}
{{"action": "answer", "answer": "<your answer>", "cited": [<excerpt ids you used>]}}

Rules:
- Answer ONLY from excerpts returned by your searches. Cite the [id] of every excerpt you use.
- Split multi-part questions into separate, focused searches (one topic per search).
- If a search result doesn't cover what you need, search again with different words.
- If the excerpts don't contain the answer, say so and reply with "cited": [].
- If two searches in a row find nothing relevant, stop searching and answer with "cited": [].
- Never repeat a search you already made.
- You have at most {max_searches} searches. Searches left are shown below."""


@dataclass
class AgentStep:
    """One search the agent chose to make."""

    query: str
    reason: str
    excerpt_ids: list[int]


@dataclass
class AgentRun:
    """The final answer plus the trace of searches that produced it."""

    result: RAGAnswer
    steps: list[AgentStep] = field(default_factory=list)


def _parse_decision(raw_text: str) -> dict | None:
    """Pull the first JSON object out of the reply (tolerates ```json fences and extra text)."""
    match = re.search(r"\{.*\}", raw_text, re.S)
    if not match:
        return None
    try:
        decision = json.loads(match.group(0))
    except json.JSONDecodeError:
        return None
    return decision if decision.get("action") in ("search", "answer") else None


def _format_results(query: str, chunks: list[RetrievedChunk]) -> str:
    if not chunks:
        return f"Search: {query}\nResults: (nothing found)"
    blocks = "\n\n".join(
        f"[{c.id}] ({c.company_name}, {c.section}, FY{c.fiscal_year})\n{c.text}" for c in chunks
    )
    return f"Search: {query}\nResults:\n{blocks}"


def answer_question_agentic(
    question: str,
    collection_name: str | None = None,
    max_searches: int = MAX_SEARCHES,
    top_k: int = SEARCH_TOP_K,
) -> AgentRun:
    """Let the LLM search the filings as many times as it needs (up to max_searches), then answer."""
    instructions = AGENT_INSTRUCTIONS.format(max_searches=max_searches)
    transcript = [f"Question: {question}"]
    seen: dict[int, RetrievedChunk] = {}
    steps: list[AgentStep] = []
    input_tokens = output_tokens = 0
    cost_usd = 0.0

    # One extra turn after the last search, so the agent always gets a chance to answer.
    for _ in range(max_searches + 2):
        searches_left = max_searches - len(steps)
        prompt = f"{instructions}\n\n" + "\n\n".join(transcript) + f"\n\nSearches left: {searches_left}. Reply with one JSON object."
        response = complete_with_usage(prompt, max_tokens=800)
        input_tokens += response.input_tokens
        output_tokens += response.output_tokens
        cost_usd += response.cost_usd

        decision = _parse_decision(response.text)
        if decision is None:
            logger.info("answer_question_agentic: unparseable reply, asking again")
            transcript.append("Your last reply was not a valid JSON object. Reply with one JSON object.")
            continue

        if decision["action"] == "search":
            if searches_left <= 0:
                transcript.append("No searches left. Answer now from the excerpts you have.")
                continue
            query = str(decision.get("query") or question)
            if query.strip().lower() in {s.query.strip().lower() for s in steps}:
                transcript.append(f"You already searched {query!r}. Use different words, or answer now.")
                continue
            chunks = hybrid_search(query, top_k=top_k, collection_name=collection_name)
            for chunk in chunks:
                seen[chunk.id] = chunk
            steps.append(AgentStep(query=query, reason=str(decision.get("reason", "")), excerpt_ids=[c.id for c in chunks]))
            logger.info("answer_question_agentic: search %d %r -> %d chunks", len(steps), query, len(chunks))
            transcript.append(_format_results(query, chunks))
            continue

        cited_ids = []
        for raw_id in decision.get("cited") or []:
            try:
                chunk_id = int(raw_id)
            except (TypeError, ValueError):
                continue
            if chunk_id in seen and chunk_id not in cited_ids:
                cited_ids.append(chunk_id)
        citations = [
            Citation(
                ticker=seen[i].ticker,
                company_name=seen[i].company_name,
                section=seen[i].section,
                fiscal_year=seen[i].fiscal_year,
                text_snippet=seen[i].text[:300],
            )
            for i in cited_ids
        ]
        answer = str(decision.get("answer", "")).strip() if citations else NOT_FOUND_MESSAGE
        logger.info(
            "answer_question_agentic: %d searches, %d cited, %d input tokens, %d output tokens",
            len(steps), len(citations), input_tokens, output_tokens,
        )
        return AgentRun(
            result=RAGAnswer(question, answer, citations, input_tokens, output_tokens, cost_usd),
            steps=steps,
        )

    logger.info("answer_question_agentic: no answer after %d turns, returning not-found", max_searches + 2)
    return AgentRun(
        result=RAGAnswer(question, NOT_FOUND_MESSAGE, [], input_tokens, output_tokens, cost_usd),
        steps=steps,
    )
