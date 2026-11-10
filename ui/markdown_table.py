import streamlit as st


def render_table(rows: list[dict]) -> None:
    """Render rows as a Markdown table via st.markdown.

    st.dataframe/st.table both convert data through pandas -> pyarrow before
    sending it to the frontend; on this environment that conversion segfaults
    inside pyarrow's mimalloc allocator (confirmed via macOS crash reports).
    Markdown rendering avoids pandas/pyarrow entirely.
    """
    if not rows:
        st.caption("No data.")
        return
    columns = list(rows[0].keys())
    header = "| " + " | ".join(columns) + " |"
    separator = "| " + " | ".join(["---"] * len(columns)) + " |"
    body_lines = [
        "| " + " | ".join(str(row.get(col, "")).replace("\n", " ").replace("|", "\\|") for col in columns) + " |"
        for row in rows
    ]
    st.markdown("\n".join([header, separator, *body_lines]))
