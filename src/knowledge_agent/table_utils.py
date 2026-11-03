from __future__ import annotations

import re

_TABLE_SEPARATOR = re.compile(
    r"^\s*\|?\s*:?-{3,}:?\s*(?:\|\s*:?-{3,}:?\s*)+\|?\s*$"
)


def _cell(value: str) -> str:
    return (value or "").replace("\r", " ").replace("\n", "<br>").replace("|", "\\|").strip()


def rows_to_markdown(rows: list[list[str]]) -> str:
    """Serialize a detected table into stable Markdown with a visible header."""
    normalized = [[_cell(value) for value in row] for row in rows if row]
    if not normalized:
        return ""
    width = max(len(row) for row in normalized)
    normalized = [row + [""] * (width - len(row)) for row in normalized]
    header = normalized[0]
    body = normalized[1:]
    lines = [
        "| " + " | ".join(header) + " |",
        "| " + " | ".join("---" for _ in range(width)) + " |",
    ]
    lines.extend("| " + " | ".join(row) + " |" for row in body)
    return "\n".join(lines)


def is_markdown_table_start(lines: list[str], index: int) -> bool:
    if index + 1 >= len(lines):
        return False
    first = lines[index].strip()
    second = lines[index + 1].strip()
    return "|" in first and bool(_TABLE_SEPARATOR.match(second))


def count_markdown_tables(text: str) -> int:
    lines = text.splitlines()
    return sum(1 for index in range(len(lines)) if is_markdown_table_start(lines, index))
