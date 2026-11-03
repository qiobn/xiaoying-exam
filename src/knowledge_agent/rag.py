from __future__ import annotations

from dataclasses import dataclass

from .config import Settings
from .persistence import Store
from .retrieval import search_with_diagnostics


@dataclass(frozen=True)
class RetrievalOutcome:
    hits: list[dict]
    diagnostics: dict


class KnowledgeRetrievalService:
    """One retrieval entry point shared by Agent tools and API diagnostics."""

    def __init__(
        self,
        store: Store,
        kb_id: str,
        settings: Settings,
        document_ids: list[str] | None = None,
        required_terms: tuple[str, ...] = (),
    ):
        self.store = store
        self.kb_id = kb_id
        self.settings = settings
        self.document_ids = document_ids
        self.required_terms = required_terms

    def retrieve(self, query: str, top_k: int = 5) -> RetrievalOutcome:
        hits, diagnostics = search_with_diagnostics(
            self.store,
            self.kb_id,
            query,
            top_k,
            document_ids=self.document_ids,
            settings=self.settings,
        )
        if self.required_terms and hits:
            def searchable(hit: dict) -> str:
                return " ".join(str(hit.get(key, "")) for key in ("title", "scope", "section_path", "text"))

            hits = [hit for hit in hits if any(term in searchable(hit) for term in self.required_terms)]
            diagnostics = {
                **diagnostics,
                "required_terms": list(self.required_terms),
                "required_terms_filtered": True,
            }
        return RetrievalOutcome(hits=hits, diagnostics=diagnostics)
