from __future__ import annotations

import asyncio
import json
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import date
from pathlib import Path

from fastapi import FastAPI, File, Form, Header, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel, Field

from .agent import run_agent
from .config import load_settings, save_settings
from .ingestion import build_kb, sha256
from .parsers import SUPPORTED_EXTENSIONS, parse_attachment
from .persistence import Store

settings = load_settings()
store = Store(settings.db_path)
executor = ThreadPoolExecutor(max_workers=4, thread_name_prefix="knowledge-agent")
app = FastAPI(title="Xiaoying Knowledge Agent")
app.add_middleware(CORSMiddleware, allow_origins=["http://127.0.0.1:5173", "http://localhost:5173", "http://127.0.0.1:8000"], allow_methods=["*"], allow_headers=["*"])


def seed_demo_if_empty():
    if store.list_kbs():
        return
    seed_path = Path(__file__).resolve().parents[2] / "data" / "documents.jsonl"
    if not seed_path.exists():
        return
    kb = store.create_kb("演示知识库")
    for line in seed_path.read_text(encoding="utf-8-sig").splitlines():
        if not line.strip():
            continue
        item = json.loads(line)
        raw = item["content"].encode("utf-8")
        store.add_document(kb["id"], f"{item['document_id']}.md", item["title"], item["version_date"], item["scope"], item["content"], sha256(raw))
    build_kb(store, kb["id"], settings)


seed_demo_if_empty()


class KnowledgeBaseIn(BaseModel):
    name: str = Field(min_length=1, max_length=80)


class SessionIn(BaseModel):
    kb_id: str
    title: str = "新会话"


class RunIn(BaseModel):
    question: str = Field(min_length=1, max_length=2000)
    history: list[dict] = Field(default_factory=list, max_length=12)


class RetrieveIn(BaseModel):
    query: str = Field(max_length=2000)
    top_k: int = Field(default=5, ge=1, le=10)
    history: list[dict] = Field(default_factory=list, max_length=12)


class SettingsUpdate(BaseModel):
    mode: str | None = None
    model: str | None = None
    base_url: str | None = None
    api_key: str | None = None
    llm_timeout_seconds: int | None = Field(default=None, ge=5, le=180)
    query_rewrite_enabled: bool | None = None
    local_models_only: bool | None = None
    embedding_model: str | None = None
    embedding_backend: str | None = None
    embedding_device: str | None = None
    embedding_batch_size: int | None = Field(default=None, ge=1, le=256)
    embedding_api_url: str | None = None
    embedding_api_key: str | None = None
    model_cache_dir: str | None = None
    rerank_enabled: bool | None = None
    rerank_backend: str | None = None
    rerank_model: str | None = None
    rerank_device: str | None = None
    rerank_top_k: int | None = Field(default=None, ge=1, le=100)
    rerank_api_url: str | None = None
    rerank_api_key: str | None = None
    mineru_command: str | None = None
    mineru_server_url: str | None = None
    mineru_method: str | None = None
    tesseract_command: str | None = None
    tesseract_lang: str | None = None
    chunk_strategy: str | None = None
    chunk_llm_model: str | None = None
    chunk_llm_min_chars: int | None = Field(default=None, ge=300, le=100000)
    chunk_llm_max_chars: int | None = Field(default=None, ge=1000, le=100000)
    chunk_llm_timeout_seconds: int | None = Field(default=None, ge=5, le=300)
    parse_pdf_tables: bool | None = None
    parse_embedded_images: bool | None = None
    vision_enabled: bool | None = None
    vision_model: str | None = None
    vision_api_url: str | None = None
    vision_api_key: str | None = None
    vision_timeout_seconds: int | None = Field(default=None, ge=5, le=300)
    max_embedded_images: int | None = Field(default=None, ge=0, le=200)
    clear_api_key: bool = False
    clear_rerank_api_key: bool = False
    clear_embedding_api_key: bool = False
    clear_vision_api_key: bool = False


def _valid_date(value: str) -> bool:
    try:
        date.fromisoformat(value)
        return True
    except ValueError:
        return False


@app.get("/api/health")
def health(): return {"ok": True, "mode": settings.mode}


@app.get("/api/settings")
def get_settings():
    return {
        "mode": settings.mode,
        "model": settings.model,
        "base_url": settings.base_url or "",
        "api_key_configured": bool(settings.api_key),
        "llm_timeout_seconds": settings.llm_timeout_seconds,
        "query_rewrite_enabled": settings.query_rewrite_enabled,
        "local_models_only": settings.local_models_only,
        "embedding_backend": settings.embedding_backend,
        "embedding_model": settings.embedding_model,
        "embedding_device": settings.embedding_device,
        "embedding_batch_size": settings.embedding_batch_size,
        "embedding_api_url": settings.embedding_api_url or "",
        "embedding_api_key_configured": bool(settings.embedding_api_key),
        "model_cache_dir": settings.model_cache_dir or "",
        "rerank_enabled": settings.rerank_enabled,
        "rerank_backend": settings.rerank_backend,
        "rerank_model": settings.rerank_model,
        "rerank_device": settings.rerank_device,
        "rerank_top_k": settings.rerank_top_k,
        "rerank_api_url": settings.rerank_api_url or "",
        "rerank_api_key_configured": bool(settings.rerank_api_key),
        "mineru_command": settings.mineru_command or "",
        "mineru_server_url": settings.mineru_server_url or "",
        "mineru_method": settings.mineru_method,
        "tesseract_command": settings.tesseract_command,
        "tesseract_lang": settings.tesseract_lang,
        "chunk_strategy": settings.chunk_strategy,
        "chunk_llm_model": settings.chunk_llm_model,
        "chunk_llm_min_chars": settings.chunk_llm_min_chars,
        "chunk_llm_max_chars": settings.chunk_llm_max_chars,
        "chunk_llm_timeout_seconds": settings.chunk_llm_timeout_seconds,
        "parse_pdf_tables": settings.parse_pdf_tables,
        "parse_embedded_images": settings.parse_embedded_images,
        "vision_enabled": settings.vision_enabled,
        "vision_model": settings.vision_model,
        "vision_api_url": settings.vision_api_url or "",
        "vision_api_key_configured": bool(settings.vision_api_key),
        "vision_timeout_seconds": settings.vision_timeout_seconds,
        "max_embedded_images": settings.max_embedded_images,
        "settings_file": "settings/.env",
    }


@app.put("/api/settings")
def update_settings(payload: SettingsUpdate):
    global settings
    updates = payload.model_dump(exclude_unset=True)
    if updates.get("mode") not in {None, "mock", "real"}:
        raise HTTPException(400, "mode 仅支持 mock 或 real")
    choices = {
        "embedding_backend": {"none", "local", "openai", "openai_compatible"},
        "rerank_backend": {"none", "local", "dashscope", "http"},
        "chunk_strategy": {"auto", "structured", "llm"},
    }
    for field, options in choices.items():
        if updates.get(field) is not None and updates[field] not in options:
            raise HTTPException(400, f"{field} 必须是以下值之一：{', '.join(sorted(options))}")
    if not updates.get("api_key"):
        updates.pop("api_key", None)
    if not updates.get("rerank_api_key"):
        updates.pop("rerank_api_key", None)
    if not updates.get("embedding_api_key"):
        updates.pop("embedding_api_key", None)
    if not updates.get("vision_api_key"):
        updates.pop("vision_api_key", None)
    settings = save_settings(updates)
    return get_settings()


@app.post("/api/settings/test-model")
def test_model(payload: SettingsUpdate):
    mode = payload.mode or settings.mode
    model_name = payload.model or settings.model
    api_key = payload.api_key or (None if payload.clear_api_key else settings.api_key)
    base_url = (payload.base_url if payload.base_url is not None else settings.base_url) or None
    if mode != "real":
        return {"ok": True, "mode": "mock", "message": "当前是离线 Mock，不会发起模型请求。"}
    if not api_key:
        return {"ok": False, "mode": "real", "message": "请填写 API Key；测试不会保存该密钥。"}
    started = time.perf_counter()
    try:
        from langchain_openai import ChatOpenAI

        from .agent import chat_model_options
        check_settings = settings.model_copy(update={"model": model_name, "api_key": api_key, "base_url": base_url})
        options = chat_model_options(check_settings)
        options.update(max_retries=0, timeout=20)
        response = ChatOpenAI(**options).invoke("只回复：连接成功")
        return {"ok": True, "mode": "real", "model": model_name, "duration_ms": round((time.perf_counter() - started) * 1000, 2), "preview": str(response.content)[:120]}
    except Exception as exc:  # noqa: BLE001 - surface provider diagnostics to local UI
        return {"ok": False, "mode": "real", "model": model_name, "message": str(exc)[:500]}


@app.get("/api/capabilities")
def capabilities():
    import shutil

    from .retrieval_models import model_capabilities
    model = settings.model or "gpt-4o-mini"
    return {"agent": {"framework": "LangGraph + LangChain", "mode": settings.mode,
                       "model": "离线 Mock（无内置模型权重）" if settings.mode != "real" else model,
                       "provider": "OpenAI-compatible" if settings.mode == "real" else "本地规则 Mock",
                       "api_key_configured": bool(settings.api_key), "base_url_configured": bool(settings.base_url),
                       "query_rewrite": settings.query_rewrite_enabled, "retrieval_on_demand": True},
            "retrieval": {"method": "jieba + BM25" if settings.embedding_backend == "none" else "BM25 + dense + RRF + optional rerank", "embedding_model": settings.embedding_model, "vector_index": settings.embedding_backend != "none", **model_capabilities(settings)},
            "parsers": {"text": {"available": True, "method": "本地文本解码"}, "docx": {"available": True, "method": "python-docx"},
                        "pdf": {"available": True, "method": "MinerU" if settings.mineru_command else "PyMuPDF；扫描件回退 Tesseract"},
                        "mineru": {"available": bool(settings.mineru_command and (shutil.which(settings.mineru_command) or Path(settings.mineru_command).exists()))},
                        "ocr": {"available": bool(shutil.which(settings.tesseract_command) or Path(settings.tesseract_command).exists()), "language": settings.tesseract_lang},
                        "tables": {"available": True, "pdf": settings.parse_pdf_tables, "csv": True},
                        "vision": {"enabled": settings.vision_enabled, "model": settings.vision_model or settings.model},
                        "extensions": sorted(SUPPORTED_EXTENSIONS), "max_upload_bytes": settings.max_upload_bytes},
            "chunking": {"strategy": settings.chunk_strategy, "target_chars": settings.chunk_size,
                         "overlap": settings.chunk_overlap, "llm_model": settings.chunk_llm_model or settings.model}}


@app.get("/api/knowledge-bases")
def list_kbs(): return store.list_kbs()


@app.post("/api/knowledge-bases")
def create_kb(payload: KnowledgeBaseIn):
    try:
        return store.create_kb(payload.name)
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc


@app.get("/api/knowledge-bases/{kb_id}/documents")
def documents(kb_id: str):
    if not store.get_kb(kb_id):
        raise HTTPException(404, "知识库不存在")
    return store.documents(kb_id)


@app.post("/api/knowledge-bases/{kb_id}/documents")
async def upload(kb_id: str, file: UploadFile = File(...), version_date: str = Form("2026-01-01"), scope: str = Form("未指定")):  # noqa: B008 - FastAPI form field declaration
    if not store.get_kb(kb_id): raise HTTPException(404, "知识库不存在")
    data = await file.read(settings.max_upload_bytes + 1)
    if len(data) > settings.max_upload_bytes: raise HTTPException(413, "文件过大")
    if not _valid_date(version_date): raise HTTPException(400, "version_date 必须使用有效的 YYYY-MM-DD 日期")
    try: parsed = await asyncio.to_thread(parse_attachment, file.filename or "upload.txt", data, settings)
    except (ValueError, OSError, RuntimeError) as exc: raise HTTPException(400, f"文件解析失败：{exc}") from exc
    text = parsed.text
    if not text.strip(): raise HTTPException(400, "文件未提取到可索引文本，请检查文档内容或 OCR 语言包")
    doc_id, created = store.add_document(
        kb_id,
        file.filename or "upload.txt",
        Path(file.filename or "upload.txt").stem,
        version_date,
        scope,
        text,
        sha256(data),
        parsed.parser,
        parsed.method,
        parsed.warnings,
        parsed.metadata,
    )
    store.add_build_event(kb_id, "document_parsed", {
        "document_id": doc_id, "filename": file.filename or "upload.txt",
        "parser": parsed.parser, "method": parsed.method, "warnings": parsed.warnings,
        "characters": len(text), "metadata": parsed.metadata, "created": created,
    })
    return {"document_id": doc_id, "created": created, "characters": len(text), "parser": parsed.parser, "method": parsed.method, "warnings": parsed.warnings, "metadata": parsed.metadata}


@app.post("/api/knowledge-bases/{kb_id}/build")
def build(kb_id: str):
    kb = store.get_kb(kb_id)
    if not kb: raise HTTPException(404, "知识库不存在")
    started = time.perf_counter()
    store.add_build_event(kb_id, "build_started", {"documents": len(store.documents(kb_id))})
    try:
        report = build_kb(store, kb_id, settings)
    except Exception as exc:
        store.add_build_event(kb_id, "build_failed", {"error": str(exc)[:240]})
        raise HTTPException(500, "知识库构建失败；原索引保持不变") from exc
    duration_ms = round((time.perf_counter() - started) * 1000, 2)
    store.add_build_event(kb_id, "build_finished", {
        "documents": report.documents,
        "chunks": report.chunks,
        "chunk_methods": report.chunk_methods,
        "diagnostics": report.diagnostics,
        "duration_ms": duration_ms,
    })
    return {"status": "ready", **report.to_dict(), "duration_ms": duration_ms, "index_version": kb["index_version"] + 1}


@app.get("/api/documents/{document_id}/chunks")
def chunks(document_id: str):
    if not store.get_document(document_id):
        raise HTTPException(404, "文档不存在")
    return store.chunks(document_id)


@app.get("/api/knowledge-bases/{kb_id}/build-events")
def build_events(kb_id: str):
    if not store.get_kb(kb_id):
        raise HTTPException(404, "知识库不存在")
    return store.build_events(kb_id)


@app.post("/api/knowledge-bases/{kb_id}/retrieve")
def retrieve(kb_id: str, payload: RetrieveIn):
    from .query_rewrite import ConversationQueryRewriter
    from .rag import KnowledgeRetrievalService

    kb = store.get_kb(kb_id)
    if not kb: raise HTTPException(404, "知识库不存在")
    query = payload.query.strip()
    if not query: raise HTTPException(400, "query 不能为空")
    rewrite = ConversationQueryRewriter().rewrite(query, payload.history) if settings.query_rewrite_enabled else None
    effective_query = rewrite.query if rewrite and rewrite.applied else query
    started = time.perf_counter()
    required_terms = rewrite.focus_terms if rewrite and rewrite.applied else ()
    outcome = KnowledgeRetrievalService(
        store,
        kb_id,
        settings,
        required_terms=required_terms,
    ).retrieve(effective_query, payload.top_k)
    hits, diagnostics = outcome.hits, outcome.diagnostics
    duration_ms = round((time.perf_counter() - started) * 1000, 2)
    store.add_build_event(kb_id, "retrieval", {
        "query": query, "effective_query": effective_query,
        "rewrite_applied": bool(rewrite and rewrite.applied), "returned": len(hits), "duration_ms": duration_ms,
        "hit_document_ids": list(dict.fromkeys(hit["document_id"] for hit in hits)),
    })
    return {
        "query": query, "effective_query": effective_query,
        "rewrite_applied": bool(rewrite and rewrite.applied),
        "rewrite_reason": rewrite.reason if rewrite else None, "hits": hits, "total_hits": len(hits),
        "duration_ms": duration_ms, "index_version": kb["index_version"],
        "retrieval_method": diagnostics["method"], "diagnostics": diagnostics,
    }


@app.get("/api/sessions")
def sessions(): return store.sessions()


@app.post("/api/sessions")
def create_session(payload: SessionIn):
    try:
        return store.create_session(payload.kb_id, payload.title)
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc


async def execute_run(run_id: str, session_id: str, payload: RunIn, attachments: list[tuple[str, bytes]] | None = None):
    session = next((s for s in store.sessions() if s["id"] == session_id), None)
    if not session: return
    store.add_event(run_id, "run_started", {"question": payload.question, "attachments": [name for name, _ in (attachments or [])]})
    attachment_ids = []
    try:
        parsed_attachments = []
        for filename, data in attachments or []:
            store.add_event(run_id, "attachment_received", {"filename": filename, "bytes": len(data)})
            parsed = parse_attachment(filename, data, settings)
            if not parsed.text.strip():
                raise ValueError(f"{filename} 未提取到可索引文本")
            parsed_attachments.append((filename, data, parsed))
        documents = [{"filename": filename, "title": Path(filename).stem, "version_date": date.today().isoformat(),
                      "scope": "当前会话附件", "content": parsed.text, "sha256": sha256(data),
                      "parser_name": parsed.parser, "parser_method": parsed.method,
                      "parse_warnings": parsed.warnings, "parse_metadata": parsed.metadata}
                     for filename, data, parsed in parsed_attachments]
        stored = store.add_documents(session["kb_id"], documents) if documents else []
        for (filename, _data, parsed), (doc_id, created) in zip(parsed_attachments, stored):
            attachment_ids.append(doc_id)
            store.add_event(run_id, "attachment_parsed", {"filename": filename, "document_id": doc_id, "created": created, "parser": parsed.parser, "method": parsed.method, "warnings": parsed.warnings, "metadata": parsed.metadata, "characters": len(parsed.text)})
        if attachment_ids:
            store.add_event(run_id, "attachment_indexing", {"documents": len(attachment_ids)})
            await asyncio.to_thread(build_kb, store, session["kb_id"], settings)
            store.add_event(run_id, "attachment_indexed", {"document_ids": attachment_ids})
        result = await asyncio.to_thread(run_agent, store, session["kb_id"], payload.question, payload.history, settings, lambda t,e: store.add_event(run_id,t,e), attachment_ids or None)
    except Exception as exc:  # noqa: BLE001 - background runs must reach a terminal state
        result = {
            "answer": "Agent 执行失败，未生成可靠答案。",
            "status": "tool_error",
            "citations": [],
            "gaps": [f"{str(exc)[:240]}；请检查运行日志、模型配置或解析器依赖。"],
            "question_to_user": None,
            "trace": [{"step": 1, "action": "error", "error": str(exc)[:240], "mode": settings.mode}],
            "mode": settings.mode,
        }
        store.add_event(run_id, "run_error", {"error": str(exc)[:240]})
        store.finish_run(run_id, result)
        store.add_event(run_id, "run_finished", {"status": result["status"], "result": result})
        return
    store.finish_run(run_id, result.to_dict())
    store.add_event(run_id, "run_finished", {"status": result.status, "result": result.to_dict()})


@app.get("/api/sessions/{session_id}/runs")
def session_runs(session_id: str):
    if not any(item["id"] == session_id for item in store.sessions()):
        raise HTTPException(404, "会话不存在")
    values = store.session_runs(session_id)
    for run in values:
        run["events"] = store.events(run["id"])
    return values


@app.post("/api/sessions/{session_id}/runs")
async def create_run(session_id: str, request: Request):
    session = next((s for s in store.sessions() if s["id"] == session_id), None)
    if not session: raise HTTPException(404, "会话不存在")
    attachments: list[tuple[str, bytes]] = []
    if "multipart/form-data" in request.headers.get("content-type", ""):
        form = await request.form()
        question = str(form.get("question", "")).strip()
        history_raw = str(form.get("history", "[]"))
        try: history = json.loads(history_raw)
        except json.JSONDecodeError: raise HTTPException(400, "history 必须是 JSON 数组")
        file_items = [item for item in form.getlist("files") if hasattr(item, "read") and hasattr(item, "filename")]
        if len(file_items) > 5:
            raise HTTPException(413, "一次最多上传 5 个附件")
        total_bytes = 0
        for item in file_items:
            data = await item.read(settings.max_upload_bytes + 1)
            if len(data) > settings.max_upload_bytes:
                raise HTTPException(413, f"文件过大：{item.filename}")
            total_bytes += len(data)
            if total_bytes > settings.max_upload_bytes * 2:
                raise HTTPException(413, "本次附件总大小不能超过 20 MiB")
            attachments.append((item.filename or "attachment.txt", data))
        if not question and not attachments:
            raise HTTPException(400, "问题和附件不能同时为空")
        payload = RunIn(question=question or "请总结并回答附件中的内容", history=history)
    else:
        try: payload = RunIn.model_validate(await request.json())
        except Exception as exc: raise HTTPException(422, "请求体必须包含 question") from exc
    run_id = store.create_run(session_id, payload.question, [name for name, _ in attachments])
    executor.submit(lambda: asyncio.run(execute_run(run_id, session_id, payload, attachments)))
    return {"run_id": run_id, "status": "running", "attachments": len(attachments)}


@app.get("/api/runs/{run_id}")
def get_run(run_id: str):
    value = store.run(run_id)
    if not value: raise HTTPException(404, "运行不存在")
    return value


@app.get("/api/runs/{run_id}/events")
async def events(run_id: str, request: Request, after_seq: int = 0, last_event_id: str | None = Header(None, alias="Last-Event-ID")):
    if not store.run(run_id):
        raise HTTPException(404, "运行不存在")
    try:
        cursor = max(after_seq, int(last_event_id or 0))
    except ValueError:
        cursor = after_seq
    async def stream():
        last = cursor
        while True:
            values = store.events(run_id, last)
            for event in values:
                last = event["seq"]
                yield f"id: {last}\ndata: {json.dumps(event, ensure_ascii=False)}\n\n"
            run = store.run(run_id)
            if run and run["status"] != "running":
                break
            if await request.is_disconnected():
                break
            yield ": keep-alive\n\n"
            await asyncio.sleep(0.5)
    return StreamingResponse(stream(), media_type="text/event-stream", headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})


@app.get("/")
def index():
    file = Path(__file__).resolve().parents[2] / "frontend" / "dist" / "index.html"
    if file.exists(): return FileResponse(file)
    return {"message": "请通过 python run.py 启动前端开发服务器"}


def main():
    import uvicorn
    uvicorn.run("knowledge_agent.api:app", host="127.0.0.1", port=8000, reload=False)


# Serve Vite production assets when the frontend has been built.
from fastapi.staticfiles import StaticFiles

_frontend_dist = Path(__file__).resolve().parents[2] / "frontend" / "dist"
if (_frontend_dist / "assets").is_dir():
    app.mount("/assets", StaticFiles(directory=_frontend_dist / "assets"), name="frontend-assets")
