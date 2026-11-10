import re
from dataclasses import dataclass

from core.llm import complete_with_usage
from core.models import RetrievedChunk


@dataclass
class JudgeResult:
    score: int | None
    rationale: str
    input_tokens: int = 0
    output_tokens: int = 0
    cost_usd: float = 0.0


def _parse_score(raw_text: str) -> tuple[int | None, str]:
    text = raw_text.strip()
    match = re.search(r"SCORE:\s*(\d+)\s*$", text)
    if not match:
        return None, text
    rationale = text[: match.start()].strip()
    return int(match.group(1)), rationale


def judge_faithfulness(answer: str, chunks: list[RetrievedChunk]) -> JudgeResult:
    """LLM-as-judge: is the answer fully grounded in the retrieved chunks (1-5)?

    Judges against the full text of the chunks the model actually saw, not
    Citation.text_snippet -- that field is truncated to 300 chars for UI display
    (core/rag.py:_parse_response), so a fact appearing later in a chunk would
    otherwise look unsupported even though the model had access to it.
    """
    excerpts = "\n\n".join(f"[{i}] {c.text}" for i, c in enumerate(chunks))
    prompt = (
        "Rate how faithfully the ANSWER is grounded in the RETRIEVED EXCERPTS below, on a "
        "scale of 1 (contradicts or invents facts not in the excerpts) to 5 (every claim is "
        "directly supported by the excerpts). Give a one-sentence rationale, then on a new "
        "line write 'SCORE: <n>'.\n\n"
        f"Retrieved excerpts:\n{excerpts}\n\nAnswer: {answer}"
    )
    response = complete_with_usage(prompt, max_tokens=200)
    score, rationale = _parse_score(response.text)
    return JudgeResult(
        score=score,
        rationale=rationale,
        input_tokens=response.input_tokens,
        output_tokens=response.output_tokens,
        cost_usd=response.cost_usd,
    )


def judge_correctness(question: str, reference_answer: str, answer: str) -> JudgeResult:
    """LLM-as-judge: does the answer match the reference answer's key facts (1-5)?"""
    prompt = (
        "Rate how well the ANSWER matches the key facts in the REFERENCE ANSWER for the given "
        "QUESTION, on a scale of 1 (missing or wrong) to 5 (matches all key facts). Minor "
        "wording differences don't matter. Give a one-sentence rationale, then on a new line "
        "write 'SCORE: <n>'.\n\n"
        f"Question: {question}\n\nReference answer: {reference_answer}\n\nAnswer: {answer}"
    )
    response = complete_with_usage(prompt, max_tokens=200)
    score, rationale = _parse_score(response.text)
    return JudgeResult(
        score=score,
        rationale=rationale,
        input_tokens=response.input_tokens,
        output_tokens=response.output_tokens,
        cost_usd=response.cost_usd,
    )
