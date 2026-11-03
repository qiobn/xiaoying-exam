import asyncio
from pathlib import Path

from fastapi.testclient import TestClient

from knowledge_agent.agent import run_agent
from knowledge_agent.api import app
from knowledge_agent.config import Settings
from knowledge_agent.ingestion import build_kb, sha256
from knowledge_agent.persistence import Store


def make_store(tmp_path):
    return Store(Path(tmp_path) / "db.sqlite")


def add_doc(store, kb_id, name, text):
    return store.add_document(kb_id, name, name, "2026-01-01", "全体员工", text, sha256(text.encode()))[0]


def test_agent_not_found_does_not_invent(tmp_path):
    store = make_store(tmp_path)
    kb = store.create_kb("empty")
    result = run_agent(store, kb["id"], "公司班车几点发车？", [], Settings(db_path=tmp_path / "db.sqlite"))
    assert result.status == "not_found"
    assert result.citations == []


def test_read_tool_cannot_cross_knowledge_bases(tmp_path):
    store = make_store(tmp_path)
    first = store.create_kb("first")
    second = store.create_kb("second")
    doc_id = add_doc(store, first["id"], "first.md", "报销期限为 30 天。")
    build_kb(store, first["id"], Settings(db_path=tmp_path / "db.sqlite"))
    result = run_agent(store, second["id"], "报销期限是多久？", [], Settings(db_path=tmp_path / "db.sqlite"))
    assert result.status == "not_found"
    assert doc_id not in {item["document_id"] for item in result.citations}


def test_step_limit_is_explicit(tmp_path):
    store = make_store(tmp_path)
    kb = store.create_kb("limited")
    add_doc(store, kb["id"], "policy.md", "报销期限为 30 天。")
    build_kb(store, kb["id"], Settings(db_path=tmp_path / "db.sqlite"))
    result = run_agent(store, kb["id"], "报销期限是多久？", [], Settings(db_path=tmp_path / "db.sqlite", max_steps=1))
    assert result.status == "step_limit"
    assert result.gaps


def test_api_rejects_invalid_upload_and_query(tmp_path):
    from knowledge_agent import api

    api.store = Store(tmp_path / "api.db")
    client = TestClient(app)
    kb = client.post("/api/knowledge-bases", json={"name": "api"}).json()
    bad = client.post(f"/api/knowledge-bases/{kb['id']}/documents", files={"file": ("x.exe", b"bad", "application/octet-stream")})
    assert bad.status_code == 400
    assert client.post(f"/api/knowledge-bases/{kb['id']}/retrieve", json={"query": ""}).status_code == 400
    assert client.get("/api/runs/not-found/events").status_code == 404
    assert client.get("/api/knowledge-bases/not-found/documents").status_code == 404


def test_api_retrieve_exposes_diagnostics_and_build_events(tmp_path):
    from knowledge_agent import api

    api.store = Store(tmp_path / "api.db")
    client = TestClient(app)
    kb = client.post("/api/knowledge-bases", json={"name": "diagnostics"}).json()
    uploaded = client.post(
        f"/api/knowledge-bases/{kb['id']}/documents",
        files={"file": ("policy.md", "报销期限为 30 天。".encode(), "text/markdown")},
        data={"version_date": "2026-02-30"},
    )
    assert uploaded.status_code == 400
    uploaded = client.post(
        f"/api/knowledge-bases/{kb['id']}/documents",
        files={"file": ("policy.md", "报销期限为 30 天。".encode(), "text/markdown")},
    )
    assert uploaded.status_code == 200
    built = client.post(f"/api/knowledge-bases/{kb['id']}/build")
    assert built.status_code == 200
    assert built.json()["index_version"] == 1
    result = client.post(f"/api/knowledge-bases/{kb['id']}/retrieve", json={"query": "报销期限"})
    assert result.status_code == 200
    assert result.json()["retrieval_method"] == "jieba_bm25"
    assert result.json()["total_hits"] == 1
    assert isinstance(result.json()["duration_ms"], float)
    event_types = [item["event_type"] for item in client.get(f"/api/knowledge-bases/{kb['id']}/build-events").json()]
    assert event_types[:3] == ["document_parsed", "build_started", "build_finished"]
    assert "retrieval" in event_types


def test_hybrid_retrieval_includes_dense_only_hits_and_reports_rerank_fallback(tmp_path, monkeypatch):
    from knowledge_agent import retrieval

    store = make_store(tmp_path)
    kb = store.create_kb("hybrid")
    first = add_doc(store, kb["id"], "one.md", "员工报销的最晚提交时限为三十天。")
    second = add_doc(store, kb["id"], "two.md", "差旅费用应在出差结束后按流程结算。")
    build_kb(store, kb["id"], Settings(db_path=tmp_path / "db.sqlite"))
    chunks = store.all_chunks(kb["id"])
    store.save_embeddings([(chunks[0]["id"], "fake-embed", [1.0, 0.0]), (chunks[1]["id"], "fake-embed", [0.0, 1.0])])
    settings = Settings(embedding_backend="local", embedding_model="fake-embed", rerank_enabled=True, rerank_backend="local")
    monkeypatch.setattr(retrieval, "embed_texts", lambda _texts, _settings: ([[0.0, 1.0]], type("Diag", (), {"ok": True, "error": None})()))
    monkeypatch.setattr(retrieval, "rerank", lambda *_args: ([], type("Diag", (), {"backend": "local", "model": "fake-rerank", "ok": False, "error": "offline"})()))

    hits, diagnostics = retrieval.search_with_diagnostics(store, kb["id"], "怎样结算差旅费用？", settings=settings)

    assert hits[0]["document_id"] == second
    assert {hit["document_id"] for hit in hits} == {first, second}
    assert diagnostics["method"] == "hybrid_rrf"
    assert "rerank fallback" in diagnostics["fallback"]


def test_exact_code_query_skips_dense_embedding_and_rerank(tmp_path, monkeypatch):
    from knowledge_agent import retrieval

    store = make_store(tmp_path)
    kb = store.create_kb("codes")
    code_doc = add_doc(store, kb["id"], "errors.md", "错误码 E-1042 表示身份校验失败。")
    add_doc(store, kb["id"], "other.md", "账号验证失败时请联系管理员。")
    build_kb(store, kb["id"], Settings(db_path=tmp_path / "db.sqlite"))
    monkeypatch.setattr(retrieval, "embed_texts", lambda *_args: (_ for _ in ()).throw(AssertionError("dense must be skipped")))
    monkeypatch.setattr(retrieval, "rerank", lambda *_args: (_ for _ in ()).throw(AssertionError("rerank must be skipped")))

    hits, diagnostics = retrieval.search_with_diagnostics(store, kb["id"], "E-1042 是什么错误？", settings=Settings(embedding_backend="local", embedding_model="test"))

    assert hits[0]["document_id"] == code_doc
    assert diagnostics["method"] == "exact_token_bm25"


def test_background_agent_error_finishes_run(tmp_path, monkeypatch):
    from knowledge_agent import api

    api.store = Store(tmp_path / "api.db")
    kb = api.store.create_kb("run-errors")
    session = api.store.create_session(kb["id"], "test")
    run_id = api.store.create_run(session["id"], "test question")

    def fail(*_args, **_kwargs):
        raise RuntimeError("simulated agent failure")

    monkeypatch.setattr(api, "run_agent", fail)
    asyncio.run(api.execute_run(run_id, session["id"], api.RunIn(question="test question")))
    assert api.store.run(run_id)["status"] == "tool_error"
    assert api.store.events(run_id)[-1]["event_type"] == "run_finished"
