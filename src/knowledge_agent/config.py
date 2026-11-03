from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from dotenv import dotenv_values
from pydantic import BaseModel


class Settings(BaseModel):
    mode: str = "mock"
    api_key: str | None = None
    base_url: str | None = None
    model: str = "gpt-4o-mini"
    llm_timeout_seconds: int = 45
    query_rewrite_enabled: bool = True
    local_models_only: bool = True
    embedding_backend: str = "none"
    embedding_model: str = "none"
    embedding_device: str = "cpu"
    embedding_batch_size: int = 16
    embedding_api_url: str | None = None
    embedding_api_key: str | None = None
    model_cache_dir: str | None = "runtime/models"
    rerank_enabled: bool = True
    rerank_backend: str = "local"
    rerank_model: str = "BAAI/bge-reranker-base"
    rerank_device: str = "cpu"
    rerank_max_length: int = 512
    rerank_top_k: int = 8
    rerank_api_url: str | None = None
    rerank_api_key: str | None = None
    rerank_timeout_seconds: int = 20
    mineru_command: str | None = None
    mineru_server_url: str | None = None
    mineru_method: str = "auto"
    mineru_timeout_seconds: int = 300
    tesseract_command: str = "tesseract"
    tesseract_lang: str = "chi_sim+eng"
    db_path: Path = Path("runtime/knowledge.db")
    max_upload_bytes: int = 10 * 1024 * 1024
    chunk_size: int = 420
    chunk_overlap: int = 60
    chunk_strategy: str = "auto"
    chunk_llm_model: str = ""
    chunk_llm_min_chars: int = 1600
    chunk_llm_max_chars: int = 12000
    chunk_llm_timeout_seconds: int = 60
    parse_pdf_tables: bool = True
    parse_embedded_images: bool = True
    vision_enabled: bool = False
    vision_model: str = ""
    vision_api_url: str | None = None
    vision_api_key: str | None = None
    vision_timeout_seconds: int = 60
    max_embedded_images: int = 20
    max_steps: int = 6
    max_tool_calls: int = 8


def settings_file_path() -> Path:
    project_root = Path(__file__).resolve().parents[2]
    return Path(os.getenv("KNOWLEDGE_AGENT_SETTINGS_DIR", project_root / "settings")) / ".env"


def _env_value(values: dict[str, Any], name: str, default: Any = None) -> Any:
    value = values.get(name, default)
    return default if value is None or value == "" and default is not None and name not in {"OPENAI_API_KEY", "OPENAI_BASE_URL", "MINERU_COMMAND", "MINERU_SERVER_URL"} else value


def _env_int(values: dict[str, Any], name: str, default: int) -> int:
    value = _env_value(values, name, default)
    try:
        return int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} 必须是整数") from exc


def _path(value: str | None, default: Path, project_root: Path) -> Path:
    candidate = Path(value) if value else default
    return candidate if candidate.is_absolute() else project_root / candidate


def _values() -> dict[str, Any]:
    project_root = Path(__file__).resolve().parents[2]
    values: dict[str, Any] = {}
    # The settings folder is the canonical source; root .env remains compatible.
    root_env = project_root / ".env"
    if root_env.exists(): values.update({k: v for k, v in dotenv_values(root_env).items() if v is not None})
    local_env = settings_file_path()
    if local_env.exists(): values.update({k: v for k, v in dotenv_values(local_env).items() if v is not None})
    values.update({k: v for k, v in os.environ.items() if v is not None})
    return values


def load_settings() -> Settings:
    project_root = Path(__file__).resolve().parents[2]
    values = _values()
    return Settings(
        mode=values.get("KNOWLEDGE_AGENT_MODE", "mock"),
        api_key=values.get("OPENAI_API_KEY") or None,
        base_url=values.get("OPENAI_BASE_URL") or None,
        model=values.get("OPENAI_MODEL") or "gpt-4o-mini",
        llm_timeout_seconds=_env_int(values, "OPENAI_TIMEOUT_SECONDS", 45),
        query_rewrite_enabled=str(values.get("QUERY_REWRITE_ENABLED", "true")).lower() != "false",
        local_models_only=str(values.get("MODEL_LOCAL_FILES_ONLY", "true")).lower() != "false",
        embedding_backend=values.get("EMBEDDING_BACKEND") or ("none" if (values.get("EMBEDDING_MODEL") or "none") == "none" else "local"),
        embedding_model=values.get("EMBEDDING_MODEL") or "none",
        embedding_device=values.get("EMBEDDING_DEVICE") or "cpu",
        embedding_batch_size=_env_int(values, "EMBEDDING_BATCH_SIZE", 16),
        embedding_api_url=values.get("EMBEDDING_API_URL") or None,
        embedding_api_key=values.get("EMBEDDING_API_KEY") or None,
        model_cache_dir=values.get("MODEL_CACHE_DIR") or "runtime/models",
        rerank_enabled=str(values.get("RERANK_ENABLED", "true")).lower() != "false",
        rerank_backend=values.get("RERANK_BACKEND") or "local",
        rerank_model=values.get("RERANK_MODEL") or "BAAI/bge-reranker-base",
        rerank_device=values.get("RERANK_DEVICE") or "cpu",
        rerank_max_length=_env_int(values, "RERANK_MAX_LENGTH", 512),
        rerank_top_k=_env_int(values, "RERANK_TOP_K", 8),
        rerank_api_url=values.get("RERANK_API_URL") or None,
        rerank_api_key=values.get("RERANK_API_KEY") or None,
        rerank_timeout_seconds=_env_int(values, "RERANK_TIMEOUT_SECONDS", 20),
        mineru_command=values.get("MINERU_COMMAND") or None,
        mineru_server_url=values.get("MINERU_SERVER_URL") or None,
        mineru_method=values.get("MINERU_METHOD") or "auto",
        mineru_timeout_seconds=_env_int(values, "MINERU_TIMEOUT_SECONDS", 300),
        tesseract_command=values.get("TESSERACT_COMMAND") or "tesseract",
        tesseract_lang=values.get("TESSERACT_LANG") or "chi_sim+eng",
        db_path=_path(values.get("KNOWLEDGE_AGENT_DB"), Path("runtime/knowledge.db"), project_root),
        max_upload_bytes=_env_int(values, "KNOWLEDGE_AGENT_MAX_UPLOAD_BYTES", 10 * 1024 * 1024),
        chunk_size=_env_int(values, "KNOWLEDGE_AGENT_CHUNK_SIZE", 420),
        chunk_overlap=_env_int(values, "KNOWLEDGE_AGENT_CHUNK_OVERLAP", 60),
        chunk_strategy=values.get("CHUNK_STRATEGY") or "auto",
        chunk_llm_model=values.get("CHUNK_LLM_MODEL") or "",
        chunk_llm_min_chars=_env_int(values, "CHUNK_LLM_MIN_CHARS", 1600),
        chunk_llm_max_chars=_env_int(values, "CHUNK_LLM_MAX_CHARS", 12000),
        chunk_llm_timeout_seconds=_env_int(values, "CHUNK_LLM_TIMEOUT_SECONDS", 60),
        parse_pdf_tables=str(values.get("PARSE_PDF_TABLES", "true")).lower() != "false",
        parse_embedded_images=str(values.get("PARSE_EMBEDDED_IMAGES", "true")).lower() != "false",
        vision_enabled=str(values.get("VISION_ENABLED", "false")).lower() == "true",
        vision_model=values.get("VISION_MODEL") or "",
        vision_api_url=values.get("VISION_API_URL") or None,
        vision_api_key=values.get("VISION_API_KEY") or None,
        vision_timeout_seconds=_env_int(values, "VISION_TIMEOUT_SECONDS", 60),
        max_embedded_images=_env_int(values, "MAX_EMBEDDED_IMAGES", 20),
        max_steps=_env_int(values, "KNOWLEDGE_AGENT_MAX_STEPS", 6),
        max_tool_calls=_env_int(values, "KNOWLEDGE_AGENT_MAX_TOOL_CALLS", 8),
    )


def save_settings(updates: dict[str, Any]) -> Settings:
    path = settings_file_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    current = {k: v for k, v in dotenv_values(path).items() if v is not None} if path.exists() else {}
    key_map = {
        "mode": "KNOWLEDGE_AGENT_MODE", "model": "OPENAI_MODEL", "base_url": "OPENAI_BASE_URL",
        "api_key": "OPENAI_API_KEY", "llm_timeout_seconds": "OPENAI_TIMEOUT_SECONDS",
        "query_rewrite_enabled": "QUERY_REWRITE_ENABLED", "local_models_only": "MODEL_LOCAL_FILES_ONLY",
        "embedding_backend": "EMBEDDING_BACKEND", "embedding_model": "EMBEDDING_MODEL",
        "embedding_device": "EMBEDDING_DEVICE", "embedding_batch_size": "EMBEDDING_BATCH_SIZE", "model_cache_dir": "MODEL_CACHE_DIR",
        "embedding_api_url": "EMBEDDING_API_URL", "embedding_api_key": "EMBEDDING_API_KEY",
        "rerank_enabled": "RERANK_ENABLED", "rerank_backend": "RERANK_BACKEND", "rerank_model": "RERANK_MODEL",
        "rerank_device": "RERANK_DEVICE", "rerank_max_length": "RERANK_MAX_LENGTH", "rerank_top_k": "RERANK_TOP_K",
        "rerank_api_url": "RERANK_API_URL", "rerank_api_key": "RERANK_API_KEY", "rerank_timeout_seconds": "RERANK_TIMEOUT_SECONDS",
        "mineru_command": "MINERU_COMMAND", "mineru_server_url": "MINERU_SERVER_URL",
        "mineru_method": "MINERU_METHOD", "tesseract_command": "TESSERACT_COMMAND",
        "tesseract_lang": "TESSERACT_LANG",
        "chunk_strategy": "CHUNK_STRATEGY", "chunk_llm_model": "CHUNK_LLM_MODEL",
        "chunk_llm_min_chars": "CHUNK_LLM_MIN_CHARS",
        "chunk_llm_max_chars": "CHUNK_LLM_MAX_CHARS",
        "chunk_llm_timeout_seconds": "CHUNK_LLM_TIMEOUT_SECONDS",
        "parse_pdf_tables": "PARSE_PDF_TABLES",
        "parse_embedded_images": "PARSE_EMBEDDED_IMAGES",
        "vision_enabled": "VISION_ENABLED", "vision_model": "VISION_MODEL",
        "vision_api_url": "VISION_API_URL", "vision_api_key": "VISION_API_KEY",
        "vision_timeout_seconds": "VISION_TIMEOUT_SECONDS",
        "max_embedded_images": "MAX_EMBEDDED_IMAGES",
    }
    for field, env_name in key_map.items():
        if field in updates and updates[field] is not None:
            current[env_name] = str(updates[field])
    if updates.get("clear_api_key"):
        current.pop("OPENAI_API_KEY", None)
    if updates.get("clear_embedding_api_key"):
        current.pop("EMBEDDING_API_KEY", None)
    if updates.get("clear_rerank_api_key"):
        current.pop("RERANK_API_KEY", None)
    if updates.get("clear_vision_api_key"):
        current.pop("VISION_API_KEY", None)
    lines = [f"{key}={value}" for key, value in current.items()]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return load_settings()
