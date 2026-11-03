import time

from fastapi.testclient import TestClient

from knowledge_agent.api import app


def test_chat_attachment_is_parsed_indexed_and_cited(tmp_path):
    from knowledge_agent import api
    from knowledge_agent.config import Settings
    from knowledge_agent.persistence import Store

    api.store = Store(tmp_path / "chat.db")
    api.settings = Settings(mode="mock", db_path=tmp_path / "chat.db")
    client = TestClient(app)
    kb = client.post("/api/knowledge-bases", json={"name": "附件测试"}).json()
    session = client.post("/api/sessions", json={"kb_id": kb["id"]}).json()
    response = client.post(
        f"/api/sessions/{session['id']}/runs",
        data={"question": "报销期限是多久？", "history": "[]"},
        files=[("files", ("expense.md", "报销期限为 45 天。".encode(), "text/markdown"))],
    )
    assert response.status_code == 200, response.text
    run_id = response.json()["run_id"]
    for _ in range(100):
        run = client.get(f"/api/runs/{run_id}").json()
        if run["status"] != "running":
            break
        time.sleep(0.05)
    assert run["result"]["status"] == "completed"
    assert "45" in run["result"]["answer"]
    assert run["result"]["citations"][0]["title"] == "expense"
    events = client.get(f"/api/runs/{run_id}/events").text
    assert "attachment_parsed" in events and "attachment_indexed" in events
    history = client.get(f"/api/sessions/{session['id']}/runs").json()
    assert history[0]["attachments"] == ["expense.md"]
    assert any(event["event_type"] == "attachment_parsed" for event in history[0]["events"])
    assert client.get("/api/capabilities").json()["retrieval"]["embedding_model"] == "none"


def test_image_without_ocr_fails_explicitly(tmp_path, monkeypatch):
    from knowledge_agent.config import Settings
    from knowledge_agent.parsers import parse_attachment

    monkeypatch.setattr("knowledge_agent.document_parsers.shutil.which", lambda _name: None)
    try:
        parse_attachment("scan.png", b"not-a-real-image", Settings(tesseract_command="missing-tesseract"))
    except ValueError as exc:
        assert "Tesseract" in str(exc)
    else:
        raise AssertionError("missing OCR binary must not report success")


def test_multi_attachment_parse_failure_does_not_insert_partial_documents(tmp_path):
    from knowledge_agent import api
    from knowledge_agent.config import Settings
    from knowledge_agent.persistence import Store

    api.store = Store(tmp_path / "atomic.db")
    api.settings = Settings(db_path=tmp_path / "atomic.db", tesseract_command="missing-tesseract")
    client = TestClient(app)
    kb = client.post("/api/knowledge-bases", json={"name": "atomic"}).json()
    session = client.post("/api/sessions", json={"kb_id": kb["id"]}).json()
    response = client.post(
        f"/api/sessions/{session['id']}/runs",
        data={"question": "总结附件", "history": "[]"},
        files=[("files", ("good.md", "有效内容".encode(), "text/markdown")), ("files", ("bad.png", b"bad", "image/png"))],
    )
    run_id = response.json()["run_id"]
    for _ in range(100):
        run = client.get(f"/api/runs/{run_id}").json()
        if run["status"] != "running": break
        time.sleep(0.05)
    assert run["status"] == "tool_error"
    assert client.get(f"/api/knowledge-bases/{kb['id']}/documents").json() == []
    assert "Tesseract" in run["result"]["gaps"][0]
