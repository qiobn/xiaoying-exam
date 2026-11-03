from __future__ import annotations

import importlib.util
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .config import Settings


@dataclass
class ModelDiagnostic:
    backend: str
    model: str
    ok: bool
    error: str | None = None


_embedding_cache: dict[tuple[str, str, str], Any] = {}
_reranker_cache: dict[tuple[str, str, str], Any] = {}


def _available(module: str) -> bool:
    return importlib.util.find_spec(module) is not None


def cached_model_path(model_name: str, cache_dir: str | None) -> str:
    """Resolve an existing Hugging Face snapshot so loading stays fully offline."""
    if not model_name or not cache_dir:
        return model_name
    direct = Path(model_name).expanduser()
    if direct.exists():
        return str(direct)
    cache_root = Path(cache_dir).expanduser()
    if not cache_root.is_absolute():
        cache_root = Path(__file__).resolve().parents[2] / cache_root
    repo_dir = cache_root / f"models--{model_name.replace('/', '--')}"
    snapshot_root = repo_dir / "snapshots"
    if not snapshot_root.is_dir():
        return model_name
    snapshots = [item for item in snapshot_root.iterdir() if item.is_dir()]
    if not snapshots:
        return model_name
    return str(max(snapshots, key=lambda item: item.stat().st_mtime))


def model_capabilities(settings: Settings) -> dict[str, dict[str, Any]]:
    embedding_module = _available("sentence_transformers")
    rerank_module = embedding_module
    return {
        "embedding": {
            "backend": settings.embedding_backend,
            "model": settings.embedding_model,
            "dependency_installed": embedding_module,
            "configured": settings.embedding_backend != "none",
            "available": settings.embedding_backend != "local" or embedding_module,
            "recommended": "BAAI/bge-small-zh-v1.5",
        },
        "rerank": {
            "backend": settings.rerank_backend,
            "model": settings.rerank_model,
            "dependency_installed": rerank_module,
            "configured": settings.rerank_enabled and settings.rerank_backend != "none",
            "available": not settings.rerank_enabled or settings.rerank_backend != "local" or rerank_module,
            "recommended": "BAAI/bge-reranker-base",
        },
    }


def _local_embedding(texts: list[str], settings: Settings) -> list[list[float]]:
    if not _available("sentence_transformers"):
        raise RuntimeError("未安装 sentence-transformers；请执行 pip install -r requirements-local-rag.txt")
    from sentence_transformers import SentenceTransformer

    key = (settings.embedding_backend, settings.embedding_model, settings.embedding_device)
    model = _embedding_cache.get(key)
    if model is None:
        cache_dir = str(Path(settings.model_cache_dir).expanduser()) if settings.model_cache_dir else None
        if cache_dir and not Path(cache_dir).is_absolute():
            cache_dir = str(Path(__file__).resolve().parents[2] / cache_dir)
        model = SentenceTransformer(
            cached_model_path(settings.embedding_model, cache_dir),
            device=settings.embedding_device,
            cache_folder=cache_dir,
            local_files_only=settings.local_models_only,
        )
        _embedding_cache[key] = model
    values = model.encode(texts, batch_size=settings.embedding_batch_size, normalize_embeddings=True, show_progress_bar=False)
    return values.tolist()


def _openai_embedding(texts: list[str], settings: Settings) -> list[list[float]]:
    from langchain_openai import OpenAIEmbeddings

    model = OpenAIEmbeddings(model=settings.embedding_model, api_key=settings.embedding_api_key or settings.api_key, base_url=settings.embedding_api_url or settings.base_url)
    return model.embed_documents(texts)


def embed_texts(texts: list[str], settings: Settings) -> tuple[list[list[float]], ModelDiagnostic]:
    if settings.embedding_backend == "none":
        raise RuntimeError("embedding backend is none")
    try:
        if settings.embedding_backend == "local":
            values = _local_embedding(texts, settings)
        elif settings.embedding_backend in {"openai", "openai_compatible"}:
            values = _openai_embedding(texts, settings)
        else:
            raise RuntimeError(f"不支持的 embedding backend：{settings.embedding_backend}")
        values = [normalize_vector([float(value) for value in vector]) for vector in values]
        return values, ModelDiagnostic(settings.embedding_backend, settings.embedding_model, True)
    except Exception as exc:  # noqa: BLE001 - retrieval must degrade to lexical search
        return [], ModelDiagnostic(settings.embedding_backend, settings.embedding_model, False, str(exc)[:300])


def _cosine(a: list[float], b: list[float]) -> float:
    if not a or not b or len(a) != len(b):
        return 0.0
    return max(-1.0, min(1.0, sum(x * y for x, y in zip(a, b))))


def _local_rerank(query: str, documents: list[str], settings: Settings) -> list[float]:
    if not _available("sentence_transformers"):
        raise RuntimeError("未安装 sentence-transformers；请执行 pip install -r requirements-local-rag.txt")
    from sentence_transformers import CrossEncoder

    key = (settings.rerank_backend, settings.rerank_model, settings.rerank_device)
    model = _reranker_cache.get(key)
    if model is None:
        cache_dir = str(Path(settings.model_cache_dir).expanduser()) if settings.model_cache_dir else None
        if cache_dir and not Path(cache_dir).is_absolute():
            cache_dir = str(Path(__file__).resolve().parents[2] / cache_dir)
        model = CrossEncoder(
            cached_model_path(settings.rerank_model, cache_dir),
            device=settings.rerank_device,
            max_length=settings.rerank_max_length,
            cache_dir=cache_dir,
            local_files_only=settings.local_models_only,
        )
        _reranker_cache[key] = model
    return [float(value) for value in model.predict([(query, document) for document in documents], show_progress_bar=False)]


def _dashscope_rerank(query: str, documents: list[str], settings: Settings) -> list[float]:
    import dashscope

    response = dashscope.TextReRank.call(api_key=settings.rerank_api_key or settings.api_key, model=settings.rerank_model, query=query, documents=documents, top_n=len(documents), return_documents=False)
    if getattr(response, "status_code", None) != 200 or not getattr(response, "output", None):
        raise RuntimeError(f"DashScope rerank failed: {response}")
    scores = [0.0] * len(documents)
    for item in response.output.get("results", []):
        scores[int(item["index"])] = float(item["relevance_score"])
    return scores


def _http_rerank(query: str, documents: list[str], settings: Settings) -> list[float]:
    import urllib.request

    if not settings.rerank_api_url:
        raise RuntimeError("RERANK_API_URL 未配置")
    body = json.dumps({"query": query, "documents": documents, "top_n": len(documents), "model": settings.rerank_model}).encode()
    request = urllib.request.Request(settings.rerank_api_url, data=body, headers={"Content-Type": "application/json", **({"Authorization": f"Bearer {settings.rerank_api_key}"} if settings.rerank_api_key else {})})
    with urllib.request.urlopen(request, timeout=settings.rerank_timeout_seconds) as response:
        payload = json.loads(response.read().decode())
    scores = [0.0] * len(documents)
    items = payload if isinstance(payload, list) else payload.get("results", [])
    for item in items:
        index = int(item.get("index", 0)); scores[index] = float(item.get("relevance_score", item.get("score", 0.0)))
    return scores


def rerank(query: str, documents: list[str], settings: Settings) -> tuple[list[float], ModelDiagnostic]:
    if not settings.rerank_enabled or settings.rerank_backend == "none":
        return [], ModelDiagnostic("none", settings.rerank_model, True)
    try:
        if settings.rerank_backend == "local":
            scores = _local_rerank(query, documents, settings)
        elif settings.rerank_backend == "dashscope":
            scores = _dashscope_rerank(query, documents, settings)
        elif settings.rerank_backend == "http":
            scores = _http_rerank(query, documents, settings)
        else:
            raise RuntimeError(f"不支持的 rerank backend：{settings.rerank_backend}")
        return scores, ModelDiagnostic(settings.rerank_backend, settings.rerank_model, True)
    except Exception as exc:  # noqa: BLE001 - rerank must fall back to first-stage recall
        return [], ModelDiagnostic(settings.rerank_backend, settings.rerank_model, False, str(exc)[:300])


def normalize_scores(values: list[float]) -> list[float]:
    if not values:
        return []
    low, high = min(values), max(values)
    if math.isclose(low, high):
        return [1.0 if high > 0 else 0.0 for _ in values]
    return [(value - low) / (high - low) for value in values]


def normalize_vector(values: list[float]) -> list[float]:
    norm = math.sqrt(sum(value * value for value in values))
    return [value / norm for value in values] if norm else values
