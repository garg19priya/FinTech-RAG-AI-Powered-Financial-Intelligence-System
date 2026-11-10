import logging
import re

from core.llm import complete_with_usage
from core.models import Citation, RAGAnswer, RetrievedChunk
from core.retrieval import hybrid_search
from core.retrieval_strategies import STRATEGIES

logger = logging.getLogger(__name__)

NOT_FOUND_MESSAGE = "I couldn't find information about this in the available filings."


def _build_prompt(question: str, chunks: list[RetrievedChunk]) -> str:
    context_blocks = "\n\n".join(
        f"[{i}] ({chunk.company_name}, {chunk.section}, FY{chunk.fiscal_year})\n{chunk.text}"
        for i, chunk in enumerate(chunks)
    )
    return (
        "Answer the question using ONLY the numbered filing excerpts below. "
        "After your answer, on a new line, write 'CITED: ' followed by a comma-separated "
        "list of the excerpt numbers you used (e.g. 'CITED: 0,2'). "
        "If the excerpts don't contain enough information to answer, say so explicitly "
        "and write 'CITED: ' with nothing after it.\n\n"
        f"Excerpts:\n{context_blocks}\n\nQuestion: {question}"
    )


def _parse_response(
    raw_text: str, chunks: list[RetrievedChunk]
) -> tuple[str, list[Citation]]:
    text = raw_text.strip()
    match = re.search(r"CITED:\s*([\d,\s]*)\s*$", text)
    if not match:
        return text, []

    answer_text = text[: match.start()].strip()
    indices_str = match.group(1).strip()
    if not indices_str:
        return answer_text, []

    citations = []
    for idx_str in indices_str.split(","):
        idx_str = idx_str.strip()
        if not idx_str.isdigit():
            continue
        idx = int(idx_str)
        if 0 <= idx < len(chunks):
            chunk = chunks[idx]
            citations.append(
                Citation(
                    ticker=chunk.ticker,
                    company_name=chunk.company_name,
                    section=chunk.section,
                    fiscal_year=chunk.fiscal_year,
                    text_snippet=chunk.text[:300],
                )
            )
    return answer_text, citations


def answer_question(
    question: str, collection_name: str | None = None, strategy: str = "baseline"
) -> RAGAnswer:
    """Retrieve relevant filing chunks and generate a grounded, cited answer.

    strategy selects a retrieval approach from core.retrieval_strategies.STRATEGIES
    (query rewriting, HyDE, multi-query fusion, re-ranking); "baseline" calls
    hybrid_search directly.
    """
    logger.info("answer_question: strategy=%r collection=%r question=%r", strategy, collection_name, question)

    if strategy == "baseline":
        chunks = hybrid_search(question, top_k=8, collection_name=collection_name)
    else:
        chunks = STRATEGIES[strategy](question, collection_name=collection_name, top_k=8)

    if not chunks:
        logger.info("answer_question: no chunks retrieved, returning not-found")
        return RAGAnswer(question=question, answer=NOT_FOUND_MESSAGE, citations=[])

    prompt = _build_prompt(question, chunks)
    llm_response = complete_with_usage(prompt)
    answer_text, citations = _parse_response(llm_response.text, chunks)
    logger.info(
        "answer_question: %d chunks retrieved, %d cited, %d input tokens, %d output tokens",
        len(chunks),
        len(citations),
        llm_response.input_tokens,
        llm_response.output_tokens,
    )

    if not citations:
        return RAGAnswer(
            question=question,
            answer=NOT_FOUND_MESSAGE,
            citations=[],
            input_tokens=llm_response.input_tokens,
            output_tokens=llm_response.output_tokens,
            cost_usd=llm_response.cost_usd,
        )

    return RAGAnswer(
        question=question,
        answer=answer_text,
        citations=citations,
        input_tokens=llm_response.input_tokens,
        output_tokens=llm_response.output_tokens,
        cost_usd=llm_response.cost_usd,
    )
