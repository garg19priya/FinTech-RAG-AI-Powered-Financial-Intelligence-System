from core.llm import complete
from core.models import RetrievedChunk
from core.retrieval import hybrid_search

_RERANK_MODEL = "Xenova/ms-marco-MiniLM-L-6-v2"
_rerank_encoder = None


def _get_rerank_encoder():
    global _rerank_encoder
    if _rerank_encoder is None:
        from fastembed.rerank.cross_encoder import TextCrossEncoder

        _rerank_encoder = TextCrossEncoder(model_name=_RERANK_MODEL)
    return _rerank_encoder


def baseline(question: str, collection_name: str | None = None, top_k: int = 8) -> list[RetrievedChunk]:
    return hybrid_search(question, top_k=top_k, collection_name=collection_name)


def query_rewrite(
    question: str, collection_name: str | None = None, top_k: int = 8
) -> list[RetrievedChunk]:
    prompt = (
        "Rewrite the following question as a focused search query for retrieving relevant "
        "passages from SEC 10-K filings. Return ONLY the rewritten query, nothing else.\n\n"
        f"Question: {question}"
    )
    rewritten = complete(prompt, max_tokens=100).strip()
    return hybrid_search(rewritten, top_k=top_k, collection_name=collection_name)


def hyde(question: str, collection_name: str | None = None, top_k: int = 8) -> list[RetrievedChunk]:
    prompt = (
        "Write a short, plausible-sounding passage from a company's 10-K filing that would "
        "answer this question. It's fine if the specific facts are invented -- this is used "
        "only to guide document retrieval, not shown to the user.\n\n"
        f"Question: {question}"
    )
    hypothetical_doc = complete(prompt, max_tokens=300).strip()
    return hybrid_search(hypothetical_doc, top_k=top_k, collection_name=collection_name)


def multi_query(
    question: str, collection_name: str | None = None, top_k: int = 8, n_variants: int = 3
) -> list[RetrievedChunk]:
    prompt = (
        f"Generate {n_variants} different search queries for retrieving SEC 10-K passages "
        "relevant to the question below. Each query should approach the question from a "
        f"different angle. Return ONLY the {n_variants} queries, one per line, no numbering.\n\n"
        f"Question: {question}"
    )
    raw = complete(prompt, max_tokens=200).strip()
    variants = [line.strip() for line in raw.splitlines() if line.strip()][:n_variants]
    if not variants:
        variants = [question]

    rrf_scores: dict[int, float] = {}
    chunks_by_id: dict[int, RetrievedChunk] = {}
    for variant in variants:
        results = hybrid_search(variant, top_k=top_k, collection_name=collection_name)
        for rank, chunk in enumerate(results):
            chunks_by_id[chunk.id] = chunk
            rrf_scores[chunk.id] = rrf_scores.get(chunk.id, 0.0) + 1.0 / (60 + rank)

    ranked_ids = sorted(rrf_scores, key=lambda cid: rrf_scores[cid], reverse=True)[:top_k]
    return [
        RetrievedChunk(**{**chunks_by_id[cid].__dict__, "score": rrf_scores[cid]})
        for cid in ranked_ids
    ]


def rerank(question: str, collection_name: str | None = None, top_k: int = 8) -> list[RetrievedChunk]:
    candidates = hybrid_search(question, top_k=top_k * 3, collection_name=collection_name)
    if not candidates:
        return []

    encoder = _get_rerank_encoder()
    scores = list(encoder.rerank(question, [c.text for c in candidates]))
    ranked = sorted(zip(candidates, scores), key=lambda pair: pair[1], reverse=True)[:top_k]
    return [
        RetrievedChunk(**{**chunk.__dict__, "score": float(score)}) for chunk, score in ranked
    ]


STRATEGIES = {
    "baseline": baseline,
    "query_rewrite": query_rewrite,
    "hyde": hyde,
    "multi_query": multi_query,
    "rerank": rerank,
}
