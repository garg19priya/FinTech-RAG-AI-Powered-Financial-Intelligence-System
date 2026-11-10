import re

import tiktoken

ITEM_HEADER_RE = re.compile(
    r"^(Item\s+\d+[A-Z]?\.?\s+[A-Za-z][^\n]{0,80})", re.IGNORECASE | re.MULTILINE
)

_encoding = tiktoken.get_encoding("cl100k_base")


def token_length(text: str) -> int:
    return len(_encoding.encode(text))


def split_into_sections(text: str) -> list[tuple[str, str]]:
    """Split filing text into (section_title, section_text) pairs using Item headers.

    Falls back to a single "Full Document" section if no Item headers are found,
    since SEC filers format Item headers inconsistently and ingestion must not fail.
    """
    matches = list(ITEM_HEADER_RE.finditer(text))
    if not matches:
        return [("Full Document", text)]

    sections = []
    for i, match in enumerate(matches):
        start = match.start()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        title = match.group(1).strip()
        body = text[start:end]
        sections.append((title, body))
    return sections
