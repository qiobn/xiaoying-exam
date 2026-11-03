from __future__ import annotations

import hashlib
from dataclasses import asdict, dataclass

from .chunking import StructuredChunker
from .config import Settings
from .semantic_chunking import SmartChunker


@dataclass
class BuildReport:
    documents: int
    chunks: int
    chunk_methods: dict[str, int]
    diagnostics: list[dict]

    def to_dict(self):
        return asdict(self)


def extract_text(filename: str, data: bytes) -> str:
    from .parsers import parse_attachment

    return parse_attachment(filename, data, Settings()).text


def split_chunks(text: str, size: int, overlap: int):
    """Backward-compatible tuple view of the reusable structured chunker."""
    return [
        (piece.text, piece.start, piece.end)
        for piece in StructuredChunker(size, overlap).split(text)
    ]


def sha256(data: bytes):
    return hashlib.sha256(data).hexdigest()


def build_kb(store, kb_id: str, settings: Settings):
    chunker = SmartChunker(settings)
    result = store.build(kb_id, chunker.split)
    if settings.embedding_backend == "none":
        return BuildReport(
            documents=result["documents"],
            chunks=result["chunks"],
            chunk_methods=result["chunk_methods"],
            diagnostics=chunker.diagnostics_history,
        )
    from .retrieval_models import embed_texts

    chunks = store.all_chunks(kb_id)
    vectors, diagnostic = embed_texts([chunk["retrieval_text"] for chunk in chunks], settings)
    if diagnostic.ok and len(vectors) == len(chunks):
        store.clear_embeddings(kb_id)
        store.save_embeddings([(chunk["id"], settings.embedding_model, vector) for chunk, vector in zip(chunks, vectors)])
    report = BuildReport(
        documents=result["documents"],
        chunks=result["chunks"],
        chunk_methods=result["chunk_methods"],
        diagnostics=chunker.diagnostics_history,
    )
    report.diagnostics.append({
        "embedding_backend": settings.embedding_backend,
        "embedding_model": settings.embedding_model,
        "embedding_ok": diagnostic.ok,
        "embedding_error": diagnostic.error,
    })
    return report
