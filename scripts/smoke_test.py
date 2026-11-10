from core.rag import answer_question

QUESTIONS = [
    "What was Apple's total net sales for fiscal year 2024?",
    "How does Apple describe supply chain risk?",
    "What was Apple's revenue in fiscal year 2019?",  # not in the filing: should be refused
]


def main() -> None:
    for question in QUESTIONS:
        print(f"\nQ: {question}")
        result = answer_question(question)
        print(f"A: {result.answer}")
        for citation in result.citations:
            print(f"  - {citation.company_name} / {citation.section} (FY{citation.fiscal_year})")


if __name__ == "__main__":
    main()
