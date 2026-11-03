import json
from pathlib import Path

from fastapi.testclient import TestClient
from langchain_core.messages import AIMessage, AIMessageChunk

from knowledge_agent.agent import run_agent
from knowledge_agent.api import app
from knowledge_agent.chunking import StructuredChunker
from knowledge_agent.config import Settings
from knowledge_agent.evaluation import EvaluationCase, evaluate
from knowledge_agent.ingestion import build_kb, sha256
from knowledge_agent.outline import OutlineResolver
from knowledge_agent.persistence import Store
from knowledge_agent.query_rewrite import QueryRewriteService


def test_document_reparse_delete_and_kb_delete(tmp_path, monkeypatch):
    from knowledge_agent import api

    api.store = Store(tmp_path / "lifecycle.db")
    api.settings = Settings(mode="mock", db_path=tmp_path / "lifecycle.db")
    client = TestClient(app)
    kb = client.post("/api/knowledge-bases", json={"name": "生命周期"}).json()
    uploaded = client.post(
        f"/api/knowledge-bases/{kb['id']}/documents",
        files={"file": ("policy.md", "报销期限为 30 天。".encode(), "text/markdown")},
    ).json()
    document_id = uploaded["document_id"]
    document = client.get(f"/api/knowledge-bases/{kb['id']}/documents").json()[0]
    source = Path(api._PROJECT_ROOT) / document["source_path"]
    assert source.exists()
    assert client.post(f"/api/knowledge-bases/{kb['id']}/build").status_code == 200

    source.write_text("报销期限为 45 天。", encoding="utf-8")
    reparsed = client.post(f"/api/documents/{document_id}/reparse")

    assert reparsed.status_code == 200
    assert reparsed.json()["rebuild_required"] is True
    assert client.get(f"/api/documents/{document_id}/chunks").json() == []
    deleted = client.delete(f"/api/documents/{document_id}")
    assert deleted.status_code == 200
    assert not source.exists()
    assert client.delete(f"/api/knowledge-bases/{kb['id']}").status_code == 200


def test_retrieval_evaluation_metrics(tmp_path):
    store = Store(tmp_path / "eval.db")
    kb = store.create_kb("eval")
    docs = [
        ("D01", "员工报销制度（大陆）", "员工应在费用发生后 30 天内提交报销申请。"),
        ("D02", "上海地区报销补充规则", "上海员工应在费用发生后 15 天内提交报销申请。"),
        ("D03", "年假申请 FAQ", "员工申请年假应至少提前 3 个工作日提交。"),
    ]
    for name, title, text in docs:
        store.add_document(kb["id"], f"{name}.md", title, "2026-01-01", "全体员工", text, sha256(text.encode()))
    build_kb(store, kb["id"], Settings(db_path=tmp_path / "eval.db"))
    cases = [
        EvaluationCase("expense", "报销期限", ["员工报销制度（大陆）", "上海地区报销补充规则"], "needs_clarification"),
        EvaluationCase("leave", "年假提前几天", ["年假申请 FAQ"], "completed"),
        EvaluationCase("missing", "班车几点发车", [], "not_found"),
    ]

    report = evaluate(store, kb["id"], Settings(db_path=tmp_path / "eval.db"), cases, top_k=3)

    assert report["summary"]["cases"] == 3
    assert report["summary"]["recall_at_k"] >= 0.9
    assert report["summary"]["mrr"] >= 0.9


def test_outline_resolver_rebuilds_flat_numbered_headings(monkeypatch):
    text = "# 总则\n适用范围。\n\n# 报销\n提交票据。\n\n# 审批\n主管审批。"

    class FakeChat:
        def __init__(self, **kwargs):
            pass

        def invoke(self, messages):
            return AIMessage(content=json.dumps({
                "headings": [
                    {"line": 1, "level": 1},
                    {"line": 4, "level": 2},
                    {"line": 7, "level": 2},
                ]
            }, ensure_ascii=False))

    monkeypatch.setattr("langchain_openai.ChatOpenAI", FakeChat)
    settings = Settings(mode="real", api_key="test", outline_llm_min_headings=2)

    outline = OutlineResolver(settings).resolve(text)

    assert outline.source == "llm"
    assert outline.path_for_line(4) == ["总则", "报销"]
    assert outline.path_for_line(7) == ["总则", "审批"]


def test_llm_query_rewrite_has_rule_fallback(monkeypatch):
    class FakeChat:
        def __init__(self, **kwargs):
            pass

        def invoke(self, messages):
            return AIMessage(content='{"query":"国外出差报销期限","keep_terms":["国外","出差"]}')

    monkeypatch.setattr("langchain_openai.ChatOpenAI", FakeChat)
    settings = Settings(mode="real", api_key="test")
    history = [{"role": "user", "content": "上海办公室的报销期限是多久？"}]

    result = QueryRewriteService(settings).rewrite("那国外出差呢？", history)

    assert result.method == "llm"
    assert result.query == "国外出差报销期限"
    assert result.focus_terms == ("国外", "出差")


def test_long_table_is_split_with_retrieval_header():
    rows = ["| 城市 | 额度 |", "| --- | --- |"] + [f"| 城市{i} | {i * 100} |" for i in range(12)]
    text = "\n".join(rows)

    pieces = StructuredChunker(size=30, overlap=0, table_max_chars=80).split(text)

    assert len(pieces) > 1
    assert all(text[piece.start:piece.end] == piece.text for piece in pieces)
    assert pieces[1].retrieval_context == "| 城市 | 额度 |\n| --- | --- |"


def test_run_idempotency_and_cancel(tmp_path):
    from knowledge_agent import api

    api.store = Store(tmp_path / "runs.db")
    api.settings = Settings(mode="mock", db_path=tmp_path / "runs.db")
    client = TestClient(app)
    kb = client.post("/api/knowledge-bases", json={"name": "运行"}).json()
    session = client.post("/api/sessions", json={"kb_id": kb["id"]}).json()
    headers = {"Idempotency-Key": "same-request"}
    first = client.post(f"/api/sessions/{session['id']}/runs", json={"question": "你好"}, headers=headers).json()
    second = client.post(f"/api/sessions/{session['id']}/runs", json={"question": "你好"}, headers=headers).json()
    assert first["run_id"] == second["run_id"]
    assert second["idempotent_replay"] is True

    store = Store(tmp_path / "cancel.db")
    cancelled = run_agent(store, kb["id"], "你好", [], Settings(db_path=tmp_path / "cancel.db"), should_cancel=lambda: True)
    assert cancelled.status == "cancelled"


def test_real_chat_streams_token_events(monkeypatch, tmp_path):
    store = Store(tmp_path / "stream.db")
    kb = store.create_kb("stream")

    class FakeStreamChat:
        def __init__(self, **kwargs):
            pass

        def stream(self, messages):
            yield AIMessageChunk(content="你好")
            yield AIMessageChunk(content="，我是知问。")

    monkeypatch.setattr("langchain_openai.ChatOpenAI", FakeStreamChat)
    events = []
    result = run_agent(
        store,
        kb["id"],
        "你好，你能做什么？",
        [],
        Settings(mode="real", api_key="test", db_path=tmp_path / "stream.db"),
        on_event=lambda event_type, payload: events.append((event_type, payload)),
    )

    tokens = "".join(payload["delta"] for event_type, payload in events if event_type == "token")
    assert result.status == "completed"
    assert tokens == "你好，我是知问。"
    assert any(event_type == "agent_stage" for event_type, _payload in events)
