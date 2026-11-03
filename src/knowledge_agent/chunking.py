from __future__ import annotations

import re
from dataclasses import dataclass

_HEADING = re.compile(r"^(#{1,6})\s+(.+?)\s*#*\s*$")
_FENCE = re.compile(r"^\s*(`{3,}|~{3,})")


@dataclass(frozen=True)
class ChunkPiece:
    text: str
    start: int
    end: int
    section_path: str = ""


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
                result.append((text[piece_start:piece_end], piece_start, piece_end))
            if end >= content_end:
                break
            start = max(start + 1, end - self.overlap)
        return result
