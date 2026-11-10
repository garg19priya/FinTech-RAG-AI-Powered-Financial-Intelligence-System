from core.models import FilingChunk, RetrievedChunk, Citation, RAGAnswer


def test_filing_chunk_holds_expected_fields():
    chunk = FilingChunk(
        ticker="AAPL",
        company_name="Apple Inc.",
        filing_type="10-K",
        section="Item 7. Management's Discussion and Analysis",
        fiscal_year="2024",
        text="Revenue grew...",
    )
    assert chunk.ticker == "AAPL"
    assert chunk.section.startswith("Item 7")


def test_retrieved_chunk_holds_score():
    chunk = RetrievedChunk(
        id=1,
        ticker="JPM",
        company_name="JPMorgan Chase & Co.",
        filing_type="10-K",
        section="Item 7",
        fiscal_year="2024",
        text="Net interest margin was 2.7%.",
        score=0.02,
    )
    assert chunk.score == 0.02


def test_rag_answer_holds_citations():
    citation = Citation(
        ticker="V",
        company_name="Visa Inc.",
        section="Item 8",
        fiscal_year="2024",
        text_snippet="Net revenue was...",
    )
    answer = RAGAnswer(question="q", answer="a", citations=[citation])
    assert answer.citations[0].ticker == "V"
