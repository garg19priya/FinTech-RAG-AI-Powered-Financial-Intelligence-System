from dataclasses import dataclass


@dataclass
class FilingChunk:
    """A section-tagged chunk of a filing, produced by ingestion and stored in Qdrant."""
    ticker: str
    company_name: str
    filing_type: str
    section: str
    fiscal_year: str
    text: str
    pipeline: str = "html_regex"


@dataclass
class RetrievedChunk:
    """A chunk returned from hybrid search, with its fused relevance score."""
    id: int
    ticker: str
    company_name: str
    filing_type: str
    section: str
    fiscal_year: str
    text: str
    score: float


@dataclass
class Citation:
    ticker: str
    company_name: str
    section: str
    fiscal_year: str
    text_snippet: str


@dataclass
class RAGAnswer:
    question: str
    answer: str
    citations: list[Citation]
    input_tokens: int = 0
    output_tokens: int = 0
    cost_usd: float = 0.0
