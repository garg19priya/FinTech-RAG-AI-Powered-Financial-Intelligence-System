import streamlit as st

from core.models import RAGAnswer
from core.rag import answer_question

COMPANIES = [
    "Apple (AAPL)",
]

# Kept in sync with the "Sample questions" section in README.md.
SAMPLE_QUESTIONS = [
    ("Net sales FY24", "What was Apple's total net sales for fiscal year 2024?"),
    (
        "Segment decline",
        "Which geographic segment had the largest revenue decline in 2024, and why?",
    ),
    (
        "AI risk",
        "How does Apple describe the risks of \"Apple Intelligence\" and generative AI in its filing?",
    ),
    (
        "New products",
        "What new products did Apple launch in fiscal 2024, and how did they affect segment performance?",
    ),
    ("Not-found trap", "What was Apple's revenue in fiscal year 2019?"),
]


def render_answer(result: RAGAnswer) -> None:
    st.write(result.answer)
    if result.citations:
        with st.expander(f"Sources ({len(result.citations)})"):
            for citation in result.citations:
                st.markdown(
                    f"**{citation.company_name}** — {citation.section} "
                    f"(FY{citation.fiscal_year})"
                )
                st.caption(citation.text_snippet)


def render() -> None:
    st.caption("Try a sample question:")
    cols = st.columns(len(SAMPLE_QUESTIONS))
    for col, (label, sample) in zip(cols, SAMPLE_QUESTIONS):
        if col.button(label, help=sample, width="stretch"):
            st.session_state["qa_question"] = sample

    question = st.text_input("Ask a question about these filings", key="qa_question")
    if st.button("Ask") and question:
        with st.spinner("Retrieving and generating answer..."):
            result = answer_question(question)
        render_answer(result)
