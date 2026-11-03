import time

from fastapi.testclient import TestClient
from langchain_core.messages import AIMessage


def test_settings_api_persists_and_applies_local_config(tmp_path, monkeypatch):
    from knowledge_agent import api
    from knowledge_agent.persistence import Store

    config_dir = tmp_path / "settings"
    monkeypatch.setenv("KNOWLEDGE_AGENT_SETTINGS_DIR", str(config_dir))
    api.store = Store(tmp_path / "api.db")
    api.settings = api.load_settings()
    client = TestClient(api.app)
    saved = client.put("/api/settings", json={"mode": "real", "model": "test-model", "base_url": "http://localhost:9999/v1"})
    assert saved.status_code == 200
    assert saved.json()["mode"] == "real"
    assert saved.json()["model"] == "test-model"
    assert saved.json()["api_key_configured"] is False
    saved_secret = client.put("/api/settings", json={"api_key": "not-a-real-secret"})
    assert saved_secret.status_code == 200
    assert saved_secret.json()["api_key_configured"] is True
    assert "not-a-real-secret" not in saved_secret.text
    assert "not-a-real-secret" in (config_dir / ".env").read_text(encoding="utf-8")
    assert (config_dir / ".env").exists()
    assert client.get("/api/capabilities").json()["agent"]["model"] == "test-model"
    api.settings = api.load_settings()


def test_real_agent_runs_tool_loop_with_openai_compatible_chat(monkeypatch, tmp_path):
    from knowledge_agent.agent import run_agent
    from knowledge_agent.config import Settings
    from knowledge_agent.ingestion import build_kb, sha256
    from knowledge_agent.persistence import Store

    store = Store(tmp_path / "db.sqlite")
    kb = store.create_kb("real")
    store.add_document(kb["id"], "policy.md", "报销制度", "2026-01-01", "全体员工", "报销期限为 30 天。", sha256("报销期限为 30 天。".encode()))
    build_kb(store, kb["id"], Settings(db_path=tmp_path / "db.sqlite"))

    class FakeChat:
        def __init__(self, **kwargs): pass
        def bind_tools(self, tools): return self
        def invoke(self, messages):
            if any(getattr(message, "name", None) == "read_document" for message in messages):
                return AIMessage(content="根据资料，报销期限是 30 天。")
            if any(getattr(message, "name", None) == "search_documents" for message in messages):
                hit = store.all_chunks(kb["id"])[0]
                return AIMessage(content="", tool_calls=[{"name": "read_document", "args": {"document_id": hit["document_id"], "query": "报销期限"}, "id": "read-1"}])
            return AIMessage(content="", tool_calls=[{"name": "search_documents", "args": {"query": "报销期限是多久？", "top_k": 5}, "id": "search-1"}])

    monkeypatch.setattr("langchain_openai.ChatOpenAI", FakeChat)
    result = run_agent(store, kb["id"], "报销期限是多久？", [], Settings(mode="real", api_key="test-key", model="fake", db_path=tmp_path / "db.sqlite"))
    assert result.status == "completed"
    assert "30" in result.answer
    assert result.citations
    assert [item["action"] for item in result.trace][:2] == ["search_documents", "read_document"]


def test_deepseek_chat_disables_thinking_for_tool_message_round_trip(monkeypatch, tmp_path):
    import json

    from knowledge_agent.agent import chat_model_options, run_agent
    from knowledge_agent.config import Settings
    from knowledge_agent.ingestion import build_kb, sha256
    from knowledge_agent.persistence import Store

    store = Store(tmp_path / "deepseek.sqlite")
    kb = store.create_kb("deepseek")
    text = "报销期限为 30 天。"
    store.add_document(kb["id"], "policy.md", "报销制度", "2026-01-01", "全体员工", text, sha256(text.encode()))
    settings = Settings(mode="real", api_key="test-key", model="deepseek-chat", base_url="https://api.deepseek.com/v1", db_path=tmp_path / "deepseek.sqlite")
    build_kb(store, kb["id"], settings)

    captured = []

    class FakeChat:
        def __init__(self, **kwargs):
            captured.append(kwargs)
        def bind_tools(self, tools): return self
        def invoke(self, messages):
            tool_messages = [message for message in messages if getattr(message, "type", None) == "tool"]
            if not tool_messages:
                return AIMessage(content="", tool_calls=[{"name": "search_documents", "args": {"query": "报销期限", "top_k": 5}, "id": "search"}])
            if tool_messages[-1].name == "search_documents":
                hit = json.loads(tool_messages[-1].content)["hits"][0]
                return AIMessage(content="", tool_calls=[{"name": "read_document", "args": {"document_id": hit["document_id"], "query": "报销期限"}, "id": "read"}])
            return AIMessage(content="报销期限为 30 天。")

    monkeypatch.setattr("langchain_openai.ChatOpenAI", FakeChat)
    assert chat_model_options(settings)["extra_body"] == {"thinking": {"type": "disabled"}}
    result = run_agent(store, kb["id"], "报销期限是多久？", [], settings)
    assert result.status == "completed"
    assert captured[0]["extra_body"] == {"thinking": {"type": "disabled"}}


def test_non_deepseek_provider_options_are_unchanged():
    from knowledge_agent.agent import chat_model_options
    from knowledge_agent.config import Settings

    assert "extra_body" not in chat_model_options(Settings(mode="real", api_key="key", base_url="https://gateway.example/v1"))


def test_real_greeting_skips_tool_binding_and_retrieval(monkeypatch, tmp_path):
    from knowledge_agent.agent import run_agent
    from knowledge_agent.config import Settings
    from knowledge_agent.ingestion import build_kb, sha256
    from knowledge_agent.persistence import Store

    store = Store(tmp_path / "greeting.sqlite")
    kb = store.create_kb("greeting")
    text = "报销期限为 30 天。"
    store.add_document(kb["id"], "policy.md", "报销制度", "2026-01-01", "全体员工", text, sha256(text.encode()))
    settings = Settings(mode="real", api_key="test-key", model="fake", db_path=tmp_path / "greeting.sqlite")
    build_kb(store, kb["id"], settings)
    observed = {"bind_tools": False}

    class FakeChat:
        def __init__(self, **kwargs): pass
        def bind_tools(self, tools):
            observed["bind_tools"] = True
            return self
        def invoke(self, messages):
            return AIMessage(content="你好，我是知问知识助手。")

    monkeypatch.setattr("langchain_openai.ChatOpenAI", FakeChat)
    result = run_agent(store, kb["id"], "你好，请介绍一下你能做什么", [], settings)

    assert observed["bind_tools"] is False
    assert result.status == "completed"
    assert result.citations == []
    assert not any(item.get("action") == "search_documents" for item in result.trace)


def test_local_reranker_uses_installed_sentence_transformers_cache_argument(monkeypatch):
    import sys
    import types

    from knowledge_agent.config import Settings
    from knowledge_agent.retrieval_models import _local_rerank

    calls = {}

    class FakeCrossEncoder:
        def __init__(self, model, **kwargs):
            calls["init"] = (model, kwargs)

        def predict(self, pairs, show_progress_bar=False):
            calls["pairs"] = pairs
            return [0.9 for _ in pairs]

    fake_module = types.ModuleType("sentence_transformers")
    fake_module.CrossEncoder = FakeCrossEncoder
    monkeypatch.setitem(sys.modules, "sentence_transformers", fake_module)
    monkeypatch.setattr("knowledge_agent.retrieval_models._available", lambda _module: True)
    monkeypatch.setattr("knowledge_agent.retrieval_models._reranker_cache", {})

    scores = _local_rerank("query", ["doc"], Settings(model_cache_dir="runtime/models"))

    assert scores == [0.9]
    assert calls["init"][1]["cache_dir"].endswith("runtime\\models") or calls["init"][1]["cache_dir"].endswith("runtime/models")
    assert calls["init"][1]["local_files_only"] is True
    assert calls["pairs"] == [("query", "doc")]


def test_cached_model_path_uses_local_snapshot(tmp_path):
    from knowledge_agent.retrieval_models import cached_model_path

    snapshot = tmp_path / "models--BAAI--example" / "snapshots" / "abc123"
    snapshot.mkdir(parents=True)

    assert cached_model_path("BAAI/example", str(tmp_path)) == str(snapshot)


def test_real_agent_searches_even_when_model_tries_to_finish_immediately(monkeypatch, tmp_path):
    from knowledge_agent.agent import run_agent
    from knowledge_agent.config import Settings
    from knowledge_agent.ingestion import build_kb, sha256
    from knowledge_agent.persistence import Store

    store = Store(tmp_path / "early-finish.sqlite")
    kb = store.create_kb("early-finish")
    text = "报销期限为 30 天。"
    store.add_document(kb["id"], "policy.md", "报销制度", "2026-01-01", "全体员工", text, sha256(text.encode()))
    settings = Settings(mode="real", api_key="test-key", model="fake", db_path=tmp_path / "early-finish.sqlite")
    build_kb(store, kb["id"], settings)

    class FakeChat:
        def __init__(self, **kwargs):
            pass

        def bind_tools(self, tools):
            self.tools = {item.name: item for item in tools}
            return self

        def invoke(self, messages):
            tool_messages = [message for message in messages if getattr(message, "type", None) == "tool"]
            if not tool_messages:
                return AIMessage(content="你好，我可以帮你查制度。")
            if tool_messages[-1].name == "search_documents":
                import json

                hits = json.loads(tool_messages[-1].content)["hits"]
                return AIMessage(content="", tool_calls=[{"name": "read_document", "args": {"document_id": hits[0]["document_id"], "query": "报销期限"}, "id": "read-policy"}])
            return AIMessage(content="报销期限为 30 天。")

    monkeypatch.setattr("langchain_openai.ChatOpenAI", FakeChat)

    result = run_agent(store, kb["id"], "你好，报销期限是多久？", [], settings)

    assert result.status == "completed"
    assert "30" in result.answer
    assert [item["action"] for item in result.trace][:2] == ["search_documents", "read_document"]


def test_attachment_only_run_is_accepted(tmp_path):
    from knowledge_agent import api
    from knowledge_agent.config import Settings
    from knowledge_agent.persistence import Store

    api.store = Store(tmp_path / "attachment-only.db")
    api.settings = Settings(db_path=tmp_path / "attachment-only.db")
    client = TestClient(api.app)
    kb = client.post("/api/knowledge-bases", json={"name": "附件"}).json()
    session = client.post("/api/sessions", json={"kb_id": kb["id"]}).json()
    response = client.post(f"/api/sessions/{session['id']}/runs", data={"question": "", "history": "[]"}, files=[("files", ("note.md", "这是附件内容。".encode(), "text/markdown"))])
    assert response.status_code == 200
    run_id = response.json()["run_id"]
    for _ in range(100):
        run = client.get(f"/api/runs/{run_id}").json()
        if run["status"] != "running": break
        time.sleep(0.05)
    assert run["question"] == "请总结并回答附件中的内容"
