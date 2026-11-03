from __future__ import annotations

import hashlib
import io
from pathlib import Path

from .chunking import StructuredChunker
from .config import Settings


def extract_text(filename: str, data: bytes) -> str:
    suffix = Path(filename).suffix.lower()
    if suffix in {".txt", ".md", ".markdown"}:
        return data.decode("utf-8-sig", errors="replace")
    if suffix == ".pdf":
        from pypdf import PdfReader
        reader = PdfReader(io.BytesIO(data))
        return "\n".join(page.extract_text() or "" for page in reader.pages)
    if suffix == ".docx":
        from docx import Document
        doc = Document(io.BytesIO(data))
        return "\n".join(p.text for p in doc.paragraphs)
    raise ValueError("仅支持 TXT、Markdown、PDF 和 DOCX 文件")


def split_chunks(text: str, size: int, overlap: int):
    """Backward-compatible tuple view of the reusable structured chunker."""
    return [
        (piece.text, piece.start, piece.end)
        for piece in StructuredChunker(size, overlap).split(text)
    ]


def sha256(data: bytes):
    return hashlib.sha256(data).hexdigest()


def build_kb(store, kb_id: str, settings: Settings):
    chunker = StructuredChunker(settings.chunk_size, settings.chunk_overlap)
    count = store.build(kb_id, chunker.split)
    if settings.embedding_backend == "none":
        return count
    from .retrieval_models import embed_texts

    chunks = store.all_chunks(kb_id)
    vectors, diagnostic = embed_texts([chunk["retrieval_text"] for chunk in chunks], settings)
    if diagnostic.ok and len(vectors) == len(chunks):
        store.clear_embeddings(kb_id)
        store.save_embeddings([(chunk["id"], settings.embedding_model, vector) for chunk, vector in zip(chunks, vectors)])
    return count
