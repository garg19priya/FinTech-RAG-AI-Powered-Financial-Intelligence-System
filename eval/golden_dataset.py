from dataclasses import dataclass, field


@dataclass
class EvalExample:
    """One golden question with the retrieval/answer expectations used to score it.

    expected_sections must be matched as a prefix/substring against RetrievedChunk.section
    (e.g. "Item 7" matches both "Item 7.    Management's Discussion..." and
    "Item 7. Management's Discussion... 27" -- the corpus tags the same Item two different
    ways depending on where in the PDF the heading was found).
    """

    id: str
    category: str  # "exact_term" | "segment_numerical" | "semantic" | "cross_section" | "not_found"
    question: str
    expected_ticker: str | None
    expected_sections: list[str] = field(default_factory=list)
    reference_answer: str | None = None
    should_find: bool = True


# All facts below are sourced verbatim from data/chunks_basic/AAPL.json (Apple FY2024 10-K).
GOLDEN_SET: list[EvalExample] = [
    EvalExample(
        id="exact_net_sales",
        category="exact_term",
        question="What was Apple's total net sales for fiscal year 2024?",
        expected_ticker="AAPL",
        expected_sections=["Item 7"],
        reference_answer=(
            "Total net sales were $391,035 million in fiscal 2024, up 2% from "
            "$383,285 million in fiscal 2023."
        ),
    ),
    EvalExample(
        id="exact_gross_margin",
        category="exact_term",
        question="What was Apple's gross margin percentage in 2024 vs. 2023?",
        expected_ticker="AAPL",
        expected_sections=["Item 7"],
        reference_answer="Total gross margin percentage was 46.2% in fiscal 2024, up from 44.1% in fiscal 2023.",
    ),
    EvalExample(
        id="exact_rd_spend",
        category="exact_term",
        question="How much did Apple spend on R&D in fiscal 2024, and what percent of net sales is that?",
        expected_ticker="AAPL",
        expected_sections=["Item 8"],
        reference_answer=(
            "Research and development expense was $31,370 million in fiscal 2024 (up from "
            "$29,915 million in fiscal 2023), about 8.0% of total net sales of $391,035 million."
        ),
    ),
    EvalExample(
        id="segment_greater_china_decline",
        category="segment_numerical",
        question="Which geographic segment had the largest revenue decline in 2024, and why?",
        expected_ticker="AAPL",
        expected_sections=["Item 8"],
        reference_answer=(
            "Greater China, with net sales declining from $72,559 million to $66,952 million "
            "(about -7.7%), due primarily to lower net sales of iPhone and iPad and weakness "
            "in the renminbi relative to the U.S. dollar."
        ),
    ),
    EvalExample(
        id="segment_products_vs_services_margin",
        category="segment_numerical",
        question="Compare Products gross margin to Services gross margin — which is higher and why?",
        expected_ticker="AAPL",
        expected_sections=["Item 7"],
        reference_answer=(
            "Services gross margin (73.9%) is substantially higher than Products gross margin "
            "(37.2%) in fiscal 2024, reflecting Services' different cost structure from hardware."
        ),
    ),
    EvalExample(
        id="segment_effective_tax_rate",
        category="segment_numerical",
        question="How did Apple's effective tax rate in fiscal 2024 compare to fiscal 2023, and why?",
        expected_ticker="AAPL",
        expected_sections=["Item 7"],
        reference_answer=(
            "The effective tax rate was 24.1% in fiscal 2024, up from 14.7% in fiscal 2023, "
            "driven primarily by a one-time $10.2 billion net income tax charge related to the "
            "EU State Aid Decision."
        ),
    ),
    EvalExample(
        id="semantic_cybersecurity",
        category="semantic",
        question="What cybersecurity risks does Apple disclose in Item 1C?",
        expected_ticker="AAPL",
        expected_sections=["Item 1C"],
        reference_answer=(
            "Item 1C describes Apple's cybersecurity risk oversight: a dedicated Information "
            "Security team led by the Head of Corporate Information Security is responsible for "
            "identifying, assessing, and managing risks from cybersecurity threats, including "
            "policies, employee training, security controls, monitoring, and incident response."
        ),
    ),
    EvalExample(
        id="semantic_foreign_currency",
        category="semantic",
        question="What does Apple say about risks from foreign currency fluctuation?",
        expected_ticker="AAPL",
        expected_sections=["Item 7A"],
        reference_answer=(
            "Apple is exposed to economic risk from interest rates and foreign exchange rates "
            "and uses various strategies, including derivative instruments, to manage this risk "
            "to its investment portfolio, term debt, and consolidated financial statements."
        ),
    ),
    EvalExample(
        id="semantic_legal_proceedings",
        category="semantic",
        question="What legal proceedings is Apple currently involved in?",
        expected_ticker="AAPL",
        expected_sections=["Item 3"],
        reference_answer=(
            "Apple discloses EU Digital Markets Act (DMA) noncompliance investigations "
            "(covering App Store developer communication/contract rules and default settings "
            "on iOS), plus other ordinary-course legal proceedings and claims."
        ),
    ),
    EvalExample(
        id="cross_tax_rate_and_segment_income",
        category="cross_section",
        question=(
            "What drove the change in Apple's effective tax rate in fiscal 2024, and how does "
            "that relate to its segment operating income for the year?"
        ),
        expected_ticker="AAPL",
        expected_sections=["Item 7", "Item 8"],
        reference_answer=(
            "The effective tax rate rose to 24.1% in fiscal 2024 (from 14.7%) mainly due to a "
            "one-time $10.2 billion net charge from the EU State Aid Decision. Segment operating "
            "income for fiscal 2024 was $162,044 million (up from $150,888 million), with R&D "
            "expense of $31,370 million and other corporate expenses bringing total operating "
            "income to $123,216 million."
        ),
    ),
    EvalExample(
        id="not_found_2019_revenue",
        category="not_found",
        question="What was Apple's revenue in fiscal year 2019?",
        expected_ticker=None,
        expected_sections=[],
        reference_answer=None,
        should_find=False,
    ),
    EvalExample(
        id="not_found_2026_buyback",
        category="not_found",
        question="What is Apple's stock buyback plan for fiscal 2026?",
        expected_ticker=None,
        expected_sections=[],
        reference_answer=None,
        should_find=False,
    ),
]
