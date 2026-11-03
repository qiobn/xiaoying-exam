from __future__ import annotations

import json
import re

import jieba
from rank_bm25 import BM25Okapi

from .retrieval_models import embed_texts, normalize_scores, normalize_vector, rerank

_GENERIC = {"员工", "申请", "提交", "需要", "应该", "规定", "制度", "工作日", "多少", "几天", "多久"}


def terms(text: str) -> set[str]:
    """Return deterministic Chinese/Latin terms used by retrieval and evidence checks."""
    tokens: list[str] = []
    for token in jieba.cut(text.lower(), cut_all=False):
        token = token.strip()
        if not token:
            continue
        if re.fullmatch(r"[\u4e00-\u9fff]+", token) and len(token) > 1:
            tokens.append(token)
            tokens.extend(token[i : i + 2] for i in range(len(token) - 1))
        elif re.fullmatch(r"[a-z0-9]+(?:[._-][a-z0-9]+)*", token):
            tokens.append(token)
    return set(tokens)


def _tokens(text: str) -> list[str]:
    return sorted(terms(text))


def _direct_hits(
    rows,
    indices: list[int],
    top_k: int,
    *,
    method: str,
    fallback: str,
):
    hits = []
    for index in indices[: max(1, min(top_k, 10))]:
        row = rows[index]
        hits.append({
            "chunk_id": row["id"],
            "document_id": row["document_id"],
            "title": row["title"],
            "version_date": row["version_date"],
            "scope": row["scope"],
            "section_path": row.get("section_path", ""),
            "text": row["text"],
            "score": 0.0,
            "bm25_score": 0.0,
            "dense_score": 0.0,
            "fusion_score": 0.0,
            "rerank_score": None,
            "matched_terms": [],
        })
    return hits, {
        "method": method,
        "fallback": fallback,
        "dense_available": False,
        "rerank": None,
    }


def search_with_diagnostics(store, kb_id: str, query: str, top_k: int = 5, document_ids: list[str] | None = None, settings=None):
    """Hybrid BM25+dense retrieval with rerank and explicit lexical fallback diagnostics."""
    query = query.strip()
    if not query:
        return [], {"method": "empty", "fallback": "empty_query"}
    rows = store.all_chunks(kb_id)
    if document_ids:
        allowed = set(document_ids)
        rows = [row for row in rows if row["document_id"] in allowed]
    query_tokens = _tokens(query)
    if not rows:
        return [], {"method": "jieba_bm25", "fallback": "no_terms_or_chunks"}
    image_intent = bool(re.search(r"(?:图片|照片|截图|图中|这张图)", query))
    if not query_tokens:
        if image_intent:
            indices = [
                index for index, row in enumerate(rows)
                if "[图片" in row.get("text", "") or "图片描述：" in row.get("text", "")
            ]
            if indices:
                return _direct_hits(
                    rows,
                    indices,
                    top_k,
                    method="image_intent",
                    fallback="image_chunk_fallback",
                )
        if document_ids:
            return _direct_hits(
                rows,
                list(range(len(rows))),
                top_k,
                method="attachment_scope",
                fallback="attachment_scope_fallback",
            )
        return [], {"method": "jieba_bm25", "fallback": "no_terms_or_chunks"}

    corpus = [_tokens(row.get("retrieval_text") or " ".join((row["title"], row["scope"], row["text"]))) for row in rows]
    bm25 = BM25Okapi(corpus)
    bm25_raw = [float(value) for value in bm25.get_scores(query_tokens)]
    lexical_hits = []
    for index, (row, raw_score, doc_tokens) in enumerate(zip(rows, bm25_raw, corpus)):
        overlap = set(query_tokens) & set(doc_tokens)
        if not overlap:
            continue
        coverage = len(overlap) / max(1, len(query_tokens))
        lexical_hits.append((index, raw_score + coverage, overlap))
    exact_tokens = re.findall(r"\b(?:[a-z]{1,8}[-_]?[0-9]{2,}|[0-9]{4,})\b", query.lower())
    if not lexical_hits:
        if image_intent:
            indices = [
                index for index, row in enumerate(rows)
                if "[图片" in row.get("text", "") or "图片描述：" in row.get("text", "")
            ]
            if indices:
                return _direct_hits(
                    rows,
                    indices,
                    top_k,
                    method="image_intent",
                    fallback="image_chunk_fallback",
                )
        if document_ids:
            return _direct_hits(
                rows,
                list(range(len(rows))),
                top_k,
                method="attachment_scope",
                fallback="attachment_scope_fallback",
            )
    if not lexical_hits and (settings is None or settings.embedding_backend == "none"):
        return [], {"method": "jieba_bm25", "fallback": "no_lexical_overlap"}

    lexical_norm = normalize_scores([score for _index, score, _overlap in lexical_hits])
    lexical_by_index = {index: (score, lexical_norm[position], overlap) for position, (index, score, overlap) in enumerate(lexical_hits)}
    candidate_indices = {index for index, _score, _overlap in lexical_hits}
    dense_scores: dict[int, float] = {}
    dense_diag = None
    if settings is not None and settings.embedding_backend != "none" and not exact_tokens:
        usable = [index for index, row in enumerate(rows) if row.get("embedding_model") == settings.embedding_model and row.get("embedding_vector")]
        if usable:
            vectors, dense_diag = embed_texts([query], settings)
            if dense_diag.ok and vectors:
                query_vector = vectors[0]
                for index in usable:
                    vector = json.loads(rows[index]["embedding_vector"])
                    if len(vector) == len(query_vector):
                        dense_scores[index] = sum(a * b for a, b in zip(normalize_vector(query_vector), normalize_vector(vector)))
                candidate_indices.update(dense_scores)
        else:
            dense_diag = type("Diag", (), {"ok": False, "error": "索引中没有当前 embedding 模型的向量，请重建知识库"})()

    candidate_indices = sorted(candidate_indices)
    dense_norm = normalize_scores([dense_scores.get(index, 0.0) for index in candidate_indices])
    dense_by_index = {index: value for index, value in zip(candidate_indices, dense_norm)}
    ranked_lexical = sorted(lexical_by_index, key=lambda index: -lexical_by_index[index][0])
    ranked_dense = sorted(dense_scores, key=lambda index: -dense_scores[index])
    rank_lex = {index: rank + 1 for rank, index in enumerate(ranked_lexical)}
    rank_dense = {index: rank + 1 for rank, index in enumerate(ranked_dense)}
    fusion = {
        index: 0.55 * lexical_by_index.get(index, (0.0, 0.0, set()))[1]
        + 0.35 * dense_by_index.get(index, 0.0)
        + 0.10 / (60 + rank_lex.get(index, len(rows)))
        + (0.10 / (60 + rank_dense[index]) if index in rank_dense else 0.0)
        for index in candidate_indices
    }
    ordered = sorted(candidate_indices, key=lambda index: (-fusion[index], rows[index]["document_id"], rows[index]["id"]))
    method = "exact_token_bm25" if exact_tokens else "hybrid_rrf" if dense_scores else "jieba_bm25"
    fallback = None
    if settings is not None and settings.embedding_backend != "none" and not dense_scores:
        fallback = dense_diag.error if dense_diag and not dense_diag.ok else "embedding unavailable"

    rerank_scores: dict[int, float] = {}
    rerank_diag = None
    if not exact_tokens and settings is not None and settings.rerank_enabled and settings.rerank_backend != "none" and len(ordered) > 1:
        selected = ordered[: max(1, min(settings.rerank_top_k, len(ordered)))]
        scores, result = rerank(query, [rows[index].get("retrieval_text") or rows[index]["text"] for index in selected], settings)
        rerank_diag = {"backend": result.backend, "model": result.model, "ok": result.ok, "error": result.error}
        if result.ok and len(scores) == len(selected):
            rerank_scores = dict(zip(selected, normalize_scores(scores)))
            ordered = sorted(ordered, key=lambda index: (-rerank_scores.get(index, -1), -fusion[index]))
        elif not result.ok:
            fallback = fallback or f"rerank fallback: {result.error}"

    hits = []
    for index in ordered[: max(1, min(top_k, 10))]:
        row = rows[index]
        bm25_score, _bm25_norm, overlap = lexical_by_index.get(index, (0.0, 0.0, set()))
        hits.append({
            "chunk_id": row["id"], "document_id": row["document_id"], "title": row["title"],
            "version_date": row["version_date"], "scope": row["scope"],
            "section_path": row.get("section_path", ""), "text": row["text"],
            "score": round(float(rerank_scores.get(index, fusion[index])), 4),
            "bm25_score": round(float(bm25_score), 4), "dense_score": round(float(dense_scores.get(index, 0.0)), 4),
            "fusion_score": round(float(fusion[index]), 4),
            "rerank_score": round(float(rerank_scores[index]), 4) if index in rerank_scores else None,
            "matched_terms": sorted(overlap),
        })
    diagnostic = {"method": method, "fallback": fallback, "dense_available": bool(dense_scores), "rerank": rerank_diag}
    return hits, diagnostic


def search(store, kb_id: str, query: str, top_k: int = 5, document_ids: list[str] | None = None, settings=None):
    hits, _diagnostic = search_with_diagnostics(store, kb_id, query, top_k, document_ids, settings)
    return hits
