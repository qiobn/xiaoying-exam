from __future__ import annotations

import re
from dataclasses import dataclass

from .table_utils import is_markdown_table_start

_HEADING = re.compile(r"^(#{1,6})\s+(.+?)\s*#*\s*$")
_FENCE = re.compile(r"^\s*(`{3,}|~{3,})")


@dataclass(frozen=True)
class ChunkPiece:
    text: str
    start: int
    end: int
    section_path: str = ""
    method: str = "structured"


class StructuredChunker:
    """Markdown-aware splitter that preserves source offsets and heading paths."""

    def __init__(self, size: int, overlap: int):
        if size < 1 or overlap < 0 or overlap >= size:
            raise ValueError("chunk_size 必须大于 0，chunk_overlap 必须小于 chunk_size")
        self.size = size
        self.overlap = overlap

    def split(self, text: str) -> list[ChunkPiece]:
        content_end = len(text.rstrip())
        content_start = len(text) - len(text.lstrip())
        if content_start >= content_end:
            return []

        pieces: list[ChunkPiece] = []
        headings: list[tuple[int, str]] = []
        section_start = content_start
        section_path = ""
        fence_marker: str | None = None

        def flush(end: int) -> None:
            nonlocal section_start
            if end <= section_start:
                return
            section = text[section_start:end]
            for local_text, local_start, local_end in self._split_section(section):
                start = section_start + local_start
                finish = section_start + local_end
                pieces.append(ChunkPiece(text=local_text, start=start, end=finish, section_path=section_path))
            section_start = end

        offset = 0
        for raw_line in text.splitlines(keepends=True):
            line_end = offset + len(raw_line)
            stripped = raw_line.rstrip("\r\n")
            fence = _FENCE.match(stripped)
            if fence:
                marker = fence.group(1)
                if fence_marker is None:
                    fence_marker = marker
                elif marker[0] == fence_marker[0] and len(marker) >= len(fence_marker):
                    fence_marker = None
                offset = line_end
                continue

            heading = _HEADING.match(stripped) if fence_marker is None else None
            if heading and offset >= section_start:
                flush(offset)
                level = len(heading.group(1))
                headings = [(old_level, title) for old_level, title in headings if old_level < level]
                headings.append((level, heading.group(2).strip()))
                section_path = " > ".join(title for _level, title in headings)
                section_start = line_end
            offset = line_end

        flush(content_end)
        return pieces

    def _split_section(self, text: str) -> list[tuple[str, int, int]]:
        result: list[tuple[str, int, int]] = []
        for start, end, is_table in self._section_units(text):
            if is_table:
                piece_start, piece_end = start, end
                while piece_start < piece_end and text[piece_start].isspace():
                    piece_start += 1
                while piece_end > piece_start and text[piece_end - 1].isspace():
                    piece_end -= 1
                if piece_start < piece_end:
                    result.append((text[piece_start:piece_end], piece_start, piece_end))
                continue
            result.extend(self._split_plain_span(text, start, end))
        return result

    def _section_units(self, text: str) -> list[tuple[int, int, bool]]:
        lines = text.splitlines(keepends=True)
        units: list[tuple[int, int, bool]] = []
        offsets: list[int] = []
        offset = 0
        for line in lines:
            offsets.append(offset)
            offset += len(line)

        normal_start = 0
        index = 0
        while index < len(lines):
            plain_lines = [line.rstrip("\r\n") for line in lines]
            if not is_markdown_table_start(plain_lines, index):
                index += 1
                continue
            if offsets[index] > normal_start:
                units.append((normal_start, offsets[index], False))
            table_start = offsets[index]
            index += 2
            while index < len(lines):
                stripped = lines[index].strip()
                if not stripped or "|" not in stripped:
                    break
                index += 1
            table_end = offsets[index] if index < len(lines) else len(text)
            units.append((table_start, table_end, True))
            normal_start = table_end
        if normal_start < len(text):
            units.append((normal_start, len(text), False))
        return units or [(0, len(text), False)]

    def _split_plain_span(self, text: str, start: int, end: int) -> list[tuple[str, int, int]]:
        base_start = start
        text = text[start:end]
        content_end = len(text.rstrip())
        start = len(text) - len(text.lstrip())
        if start >= content_end:
            return []
        result: list[tuple[str, int, int]] = []
        while start < content_end:
            end = min(start + self.size, content_end)
            if end < content_end:
                boundary = max(
                    text.rfind("\n", start, end),
                    text.rfind("。", start, end),
                    text.rfind(". ", start, end),
                )
                if boundary > start + self.size // 2:
                    end = boundary + (1 if text[boundary] in {"。", "."} else 0)
            piece_start, piece_end = start, end
            while piece_start < piece_end and text[piece_start].isspace():
                piece_start += 1
            while piece_end > piece_start and text[piece_end - 1].isspace():
                piece_end -= 1
            if piece_start < piece_end:
                result.append((
                    text[piece_start:piece_end],
                    base_start + piece_start,
                    base_start + piece_end,
                ))
            if end >= content_end:
                break
            start = max(start + 1, end - self.overlap)
        return result
