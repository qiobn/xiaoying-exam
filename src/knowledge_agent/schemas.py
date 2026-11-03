from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import UTC, datetime


def now_iso() -> str:
    return datetime.now(UTC).isoformat()


@dataclass
class Document:
    id: str
    knowledge_base_id: str
    filename: str
    title: str
    version_date: str
    scope: str
    content: str
    status: str = "uploaded"


@dataclass
class Chunk:
    id: str
    document_id: str
    index: int
    text: str
    start_offset: int
    end_offset: int


@dataclass
class Citation:
    document_id: str
    title: str
    version_date: str
    quote: str
    chunk_id: str


@dataclass
class AgentResult:
    answer: str
    status: str
    citations: list[dict]
    gaps: list[str]
    question_to_user: str | None
    trace: list[dict]
    mode: str

    def to_dict(self) -> dict:
        return asdict(self)
