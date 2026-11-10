import streamlit as st

from eval.golden_dataset import GOLDEN_SET, EvalExample
from eval.run_eval import run_example
from ui.markdown_table import render_table
from ui.tabs.pipeline_comparison import PIPELINES, STRATEGY_LABELS

CATEGORIES = sorted({e.category for e in GOLDEN_SET})
TOP_K = 8


@st.cache_data(show_spinner=False)
def run_eval_example_cached(
    example_id: str,
    category: str,
    question: str,
    collection: str,
    strategy: str,
    expected_ticker: str | None,
    expected_sections: tuple[str, ...],
    reference_answer: str | None,
    should_find: bool,
    run_judge: bool,
) -> dict:
    example = EvalExample(
        id=example_id,
        category=category,
        question=question,
        expected_ticker=expected_ticker,
        expected_sections=list(expected_sections),
        reference_answer=reference_answer,
        should_find=should_find,
    )
    return run_example(example, collection, strategy, TOP_K, run_judge)


def _run_golden_set(collection: str, strategy: str, examples: list[EvalExample], run_judge: bool) -> list[dict]:
    rows = []
    for example in examples:
        row = run_eval_example_cached(
            example.id,
            example.category,
            example.question,
            collection,
            strategy,
            example.expected_ticker,
            tuple(example.expected_sections),
            example.reference_answer,
            example.should_find,
            run_judge,
        )
        rows.append({"Category": example.category, "Question": example.question, **row})
    return rows


def _mean(values: list) -> float | None:
    values = [v for v in values if v is not None]
    return sum(values) / len(values) if values else None


def render_single_run() -> None:
    st.subheader("Run the golden set")
    pipeline_label = st.selectbox("Chunking pipeline", list(PIPELINES.keys()), key="eval_pipeline")
    strategy_label = st.selectbox("Retrieval strategy", list(STRATEGY_LABELS.keys()), key="eval_strategy")
    category = st.selectbox("Category", ["All"] + CATEGORIES, key="eval_category")
    run_judge = st.checkbox("Score answer quality with LLM-as-judge", value=True, key="eval_judge")

    if st.button("Run evaluation"):
        collection = PIPELINES[pipeline_label]["collection"]
        strategy = STRATEGY_LABELS[strategy_label]
        examples = [e for e in GOLDEN_SET if category == "All" or e.category == category]

        with st.spinner(f"Running {len(examples)} questions..."):
            rows = _run_golden_set(collection, strategy, examples, run_judge)

        findable = [r for r in rows if r["should_find"]]
        not_found = [r for r in rows if not r["should_find"]]

        cols = st.columns(6)
        hit_rate = _mean([1.0 if r["hit"] else 0.0 for r in findable])
        cols[0].metric("Hit rate", f"{hit_rate * 100:.0f}%" if hit_rate is not None else "-")
        mean_mrr = _mean([r["mrr"] for r in findable])
        cols[1].metric("Mean MRR", f"{mean_mrr:.2f}" if mean_mrr is not None else "-")
        mean_recall = _mean([r["recall"] for r in findable])
        cols[2].metric("Mean recall", f"{mean_recall * 100:.0f}%" if mean_recall is not None else "-")
        mean_faith = _mean([r["faithfulness"] for r in findable])
        cols[3].metric("Faithfulness", f"{mean_faith:.1f}/5" if mean_faith is not None else "-")
        mean_correct = _mean([r["correctness"] for r in findable])
        cols[4].metric("Correctness", f"{mean_correct:.1f}/5" if mean_correct is not None else "-")
        refusal_rate = _mean([1.0 if r["correctly_refused"] else 0.0 for r in not_found])
        cols[5].metric(
            "Correct refusals", f"{refusal_rate * 100:.0f}%" if refusal_rate is not None else "-"
        )

        st.caption(f"Total cost: ${sum(r['cost_usd'] for r in rows):.4f}")
        render_table(rows)


def render_compare_all() -> None:
    with st.expander(
        "Compare all (2 pipelines x 5 strategies x golden set -- slow, many LLM calls)"
    ):
        run_judge = st.checkbox(
            "Score answer quality with LLM-as-judge", value=False, key="eval_compare_judge"
        )
        if st.button("Run full matrix", key="eval_run_full_matrix"):
            combos = [(p_label, s_label) for p_label in PIPELINES for s_label in STRATEGY_LABELS]
            summary_rows = []
            progress = st.progress(0.0)
            for i, (p_label, s_label) in enumerate(combos):
                collection = PIPELINES[p_label]["collection"]
                strategy = STRATEGY_LABELS[s_label]
                rows = _run_golden_set(collection, strategy, GOLDEN_SET, run_judge)
                findable = [r for r in rows if r["should_find"]]
                hit_rate = _mean([1.0 if r["hit"] else 0.0 for r in findable])
                mean_mrr = _mean([r["mrr"] for r in findable])
                mean_recall = _mean([r["recall"] for r in findable])
                summary_rows.append(
                    {
                        "Pipeline": p_label,
                        "Strategy": s_label,
                        "Hit rate": f"{hit_rate * 100:.0f}%" if hit_rate is not None else "-",
                        "Mean MRR": f"{mean_mrr:.2f}" if mean_mrr is not None else "-",
                        "Mean recall": f"{mean_recall * 100:.0f}%" if mean_recall is not None else "-",
                        "Total cost ($)": round(sum(r["cost_usd"] for r in rows), 4),
                    }
                )
                progress.progress((i + 1) / len(combos))
            render_table(summary_rows)


def render() -> None:
    st.caption(
        "Runs the golden question set (README's sample-question categories, including the "
        "'not found' trap) against a chosen pipeline/strategy. Retrieval metrics (hit rate, "
        "MRR, recall) are deterministic; faithfulness and correctness are scored by an LLM judge "
        "(core/llm.py, reusing the same retry/cost tracking as generation). Cost totals "
        "under-count query_rewrite/HyDE/multi_query/rerank, since only their final generation "
        "call is tracked, not the LLM pre-processing step those strategies add."
    )
    with st.expander("Metric definitions"):
        st.markdown(
            "- **Hit rate** — did any of the top-k retrieved chunks come from the "
            "expected ticker/section? Deterministic, no LLM involved.\n"
            "- **Mean MRR** (mean reciprocal rank) — `1 / rank` of the first relevant "
            "chunk, averaged across questions. Rewards ranking the right chunk *first*, "
            "not just retrieving it somewhere in the top-k.\n"
            "- **Mean recall** — fraction of a question's expected sections actually "
            "present in the top-k. Identical to hit rate for single-section questions, "
            "but gives partial credit on cross-section questions that need chunks from "
            "more than one Item (e.g. tax rate in Item 7 + segment income in Item 8) — "
            "catches retrieval that finds one but not both.\n"
            "- **Faithfulness** — LLM-judge score (1-5): is the generated answer fully "
            "grounded in the retrieved chunks, with no invented or contradicted facts?\n"
            "- **Correctness** — LLM-judge score (1-5): does the generated answer match "
            "the key facts in the question's hand-written reference answer?\n"
            "- **Correct refusals** — for the 'not found' trap questions only: did the "
            "app correctly decline to answer (zero citations) instead of guessing?"
        )
    render_single_run()
    st.divider()
    render_compare_all()
