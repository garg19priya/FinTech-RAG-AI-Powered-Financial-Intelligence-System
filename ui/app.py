import logging

import streamlit as st

from ui.tabs import architecture, evaluation, pipeline_comparison, qa

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")

CUSTOM_CSS = """
<style>
.block-container {
    padding-top: 2rem;
    padding-bottom: 3rem;
    max-width: 1100px;
}
h1 {
    font-weight: 700;
    letter-spacing: -0.02em;
}
.stTabs [data-baseweb="tab-list"] {
    gap: 4px;
}
.stTabs [data-baseweb="tab"] {
    padding: 8px 18px;
    border-radius: 8px 8px 0 0;
}
div[data-testid="stExpander"] {
    border: 1px solid rgba(45, 212, 191, 0.25);
    border-radius: 10px;
}
div[data-testid="stMetric"] {
    background: rgba(45, 212, 191, 0.06);
    border: 1px solid rgba(45, 212, 191, 0.18);
    border-radius: 10px;
    padding: 12px 14px;
}
</style>
"""


def render_sidebar() -> None:
    with st.sidebar:
        st.subheader("Companies in this demo")
        for company in qa.COMPANIES:
            st.write(f"- {company}")


def main() -> None:
    st.set_page_config(page_title="FinTech RAG Demo", layout="wide")
    st.markdown(CUSTOM_CSS, unsafe_allow_html=True)
    st.title("SEC Filing Q&A")
    st.caption(
        "A retrieval-augmented generation demo over Apple's FY2024 10-K, "
        "with citations grounded in the source text."
    )
    render_sidebar()

    tab_qa, tab_compare, tab_eval, tab_arch = st.tabs(
        ["Q&A", "Pipeline Comparison", "Evaluation", "Architecture"]
    )
    with tab_qa:
        qa.render()
    with tab_compare:
        pipeline_comparison.render()
    with tab_eval:
        evaluation.render()
    with tab_arch:
        architecture.render()


if __name__ == "__main__":
    main()
