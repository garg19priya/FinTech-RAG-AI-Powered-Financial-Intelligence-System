from core.models import RetrievedChunk
from eval.golden_dataset import EvalExample


def _is_relevant(chunk: RetrievedChunk, example: EvalExample) -> bool:
    if example.expected_ticker and chunk.ticker != example.expected_ticker:
        return False
    if not example.expected_sections:
        return True
    return any(expected in chunk.section for expected in example.expected_sections)


def hit_rate(chunks: list[RetrievedChunk], example: EvalExample) -> bool:
    """Did any retrieved chunk match the expected ticker/section?"""
    return any(_is_relevant(c, example) for c in chunks)


def recall_at_k(chunks: list[RetrievedChunk], example: EvalExample, k: int) -> float:
    """Fraction of example.expected_sections found among the top-k chunks.

    Gives partial credit for cross_section examples that expect chunks from
    multiple distinct sections. Examples with no expected_sections score 1.0
    if any expected-ticker chunk is present, else 0.0.
    """
    top_k = chunks[:k]
    if not example.expected_sections:
        return 1.0 if hit_rate(top_k, example) else 0.0

    matched_sections = {
        expected
        for expected in example.expected_sections
        for chunk in top_k
        if (not example.expected_ticker or chunk.ticker == example.expected_ticker)
        and expected in chunk.section
    }
    return len(matched_sections) / len(example.expected_sections)


def reciprocal_rank(chunks: list[RetrievedChunk], example: EvalExample) -> float:
    """1/rank of the first relevant chunk (1-indexed), or 0.0 if none found."""
    for rank, chunk in enumerate(chunks, start=1):
        if _is_relevant(chunk, example):
            return 1.0 / rank
    return 0.0
