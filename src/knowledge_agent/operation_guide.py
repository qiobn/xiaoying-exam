from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from rank_bm25 import BM25Okapi

from .retrieval import terms


@dataclass(frozen=True)
class OperationGuideEntry:
    id: str
    title: str
    keywords: list[str]
    content: str

    def retrieval_text(self) -> str:
        return " ".join([self.title, *self.keywords, self.content])


class OperationGuideService:
    """Hidden built-in help index; never exposed through knowledge-base APIs."""

    def __init__(self, path: Path | None = None):
        root = Path(__file__).resolve().parents[2]
        guide_path = path or root / "data" / "operation_guide.jsonl"
        self.entries = [
            OperationGuideEntry(
                id=value["id"],
                title=value["title"],
                keywords=list(value.get("keywords", [])),
                content=value["content"],
            )
            for line in guide_path.read_text(encoding="utf-8-sig").splitlines()
            if line.strip()
            for value in [json.loads(line)]
        ]
        self.corpus = [sorted(terms(entry.retrieval_text())) for entry in self.entries]
        self.bm25 = BM25Okapi(self.corpus) if self.corpus else None

    def search(self, query: str, top_k: int = 3) -> list[OperationGuideEntry]:
        query_tokens = sorted(terms(query))
        if not query_tokens or self.bm25 is None:
            return []
        scores = self.bm25.get_scores(query_tokens)
        ranked = sorted(
            zip(self.entries, scores),
            key=lambda item: (-float(item[1]), item[0].id),
        )
        selected = [
            entry for entry, score in ranked
            if float(score) > 0
        ][:top_k]
        return selected

    @staticmethod
    def render(entries: list[OperationGuideEntry]) -> str:
        return "\n\n".join(f"{entry.title}：{entry.content}" for entry in entries)
