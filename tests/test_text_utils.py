def test_split_into_sections_finds_item_headers():
    from ingestion.text_utils import split_into_sections

    text = (
        "Item 1. Business\n"
        + "We design, manufacture and market synthetic widgets for testing purposes. "
        + "Widget details. " * 200
        + "\nItem 7. Management's Discussion and Analysis\n"
        + "Revenue grew due to strong widget demand. "
        + "Financial detail. " * 200
    )
    sections = split_into_sections(text)
    titles = [title for title, _ in sections]

    assert any(t.startswith("Item 1.") for t in titles)
    assert any(t.startswith("Item 7.") for t in titles)


def test_split_into_sections_falls_back_when_no_items_found():
    from ingestion.text_utils import split_into_sections

    sections = split_into_sections("Just plain filing text with no item headers at all.")

    assert sections == [
        ("Full Document", "Just plain filing text with no item headers at all.")
    ]
