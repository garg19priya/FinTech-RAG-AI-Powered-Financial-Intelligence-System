import argparse
import json
import time

from core.rag import answer_question
from core.retrieval_strategies import STRATEGIES
from eval.golden_dataset import GOLDEN_SET, EvalExample
from eval.judge import judge_correctness, judge_faithfulness
from eval.retrieval_metrics import hit_rate, reciprocal_rank, recall_at_k

PIPELINE_COLLECTIONS = {"basic": "sec_filings_basic", "advanced": "sec_filings_advanced"}


def run_example(
    example: EvalExample,
    collection: str | None,
    strategy: str,
    top_k: int,
    use_judge: bool,
) -> dict:
    result = answer_question(example.question, collection_name=collection, strategy=strategy)

    if not example.should_find:
        return {
            "id": example.id,
            "category": example.category,
            "should_find": False,
            "correctly_refused": len(result.citations) == 0,
            "hit": None,
            "mrr": None,
            "recall": None,
            "faithfulness": None,
            "correctness": None,
            "cost_usd": result.cost_usd,
        }

    chunks = STRATEGIES[strategy](example.question, collection_name=collection, top_k=top_k)
    hit = hit_rate(chunks, example)
    mrr = reciprocal_rank(chunks, example)
    recall = recall_at_k(chunks, example, k=top_k)

    faithfulness = None
    correctness = None
    judge_cost = 0.0
    if use_judge and result.citations:
        faith_result = judge_faithfulness(result.answer, chunks)
        faithfulness = faith_result.score
        judge_cost += faith_result.cost_usd
        if example.reference_answer:
            correct_result = judge_correctness(
                example.question, example.reference_answer, result.answer
            )
            correctness = correct_result.score
            judge_cost += correct_result.cost_usd

    return {
        "id": example.id,
        "category": example.category,
        "should_find": True,
        "correctly_refused": None,
        "hit": hit,
        "mrr": mrr,
        "recall": recall,
        "faithfulness": faithfulness,
        "correctness": correctness,
        "cost_usd": result.cost_usd + judge_cost,
    }


def _mean(values: list[float]) -> float | None:
    values = [v for v in values if v is not None]
    return sum(values) / len(values) if values else None


def summarize(rows: list[dict]) -> dict:
    findable = [r for r in rows if r["should_find"]]
    not_found = [r for r in rows if not r["should_find"]]
    return {
        "n_examples": len(rows),
        "hit_rate": _mean([1.0 if r["hit"] else 0.0 for r in findable]),
        "mean_mrr": _mean([r["mrr"] for r in findable]),
        "mean_recall": _mean([r["recall"] for r in findable]),
        "mean_faithfulness": _mean([r["faithfulness"] for r in findable]),
        "mean_correctness": _mean([r["correctness"] for r in findable]),
        "correct_refusal_rate": _mean(
            [1.0 if r["correctly_refused"] else 0.0 for r in not_found]
        ),
        "total_cost_usd": sum(r["cost_usd"] for r in rows),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate the RAG pipeline against the golden set.")
    parser.add_argument("--strategy", choices=list(STRATEGIES), default="baseline")
    parser.add_argument("--pipeline", choices=list(PIPELINE_COLLECTIONS), default="basic")
    parser.add_argument("--collection", default=None, help="Raw Qdrant collection name override")
    parser.add_argument("--top-k", type=int, default=8)
    parser.add_argument("--category", default=None)
    parser.add_argument("--no-judge", action="store_true")
    parser.add_argument("--output", default=None, help="Write per-example JSON rows to this path")
    args = parser.parse_args()

    collection = args.collection or PIPELINE_COLLECTIONS[args.pipeline]
    examples = [e for e in GOLDEN_SET if not args.category or e.category == args.category]

    start = time.perf_counter()
    rows = []
    for example in examples:
        row = run_example(example, collection, args.strategy, args.top_k, not args.no_judge)
        rows.append(row)
        status = "REFUSED" if row["should_find"] is False else ("HIT" if row["hit"] else "MISS")
        print(f"[{status}] {example.category:<18} {example.question}")

    elapsed = time.perf_counter() - start
    summary = summarize(rows)

    print("\n--- Summary ---")
    print(f"strategy={args.strategy} pipeline={args.pipeline} collection={collection}")
    print(f"examples: {summary['n_examples']}  wall time: {elapsed:.1f}s")
    print(f"hit rate: {summary['hit_rate']}")
    print(f"mean MRR: {summary['mean_mrr']}")
    print(f"mean recall: {summary['mean_recall']}")
    print(f"mean faithfulness: {summary['mean_faithfulness']}")
    print(f"mean correctness: {summary['mean_correctness']}")
    print(f"correct refusal rate: {summary['correct_refusal_rate']}")
    print(f"total cost (USD): {summary['total_cost_usd']:.4f}")

    if args.output:
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump({"summary": summary, "rows": rows}, f, indent=2)
        print(f"\nWrote per-example results to {args.output}")


if __name__ == "__main__":
    main()
