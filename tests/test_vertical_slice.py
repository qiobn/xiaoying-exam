import time

from fastapi.testclient import TestClient

from knowledge_agent.api import app


def test_web_vertical_slice(tmp_path, monkeypatch):
    monkeypatch.setenv("KNOWLEDGE_AGENT_DB", str(tmp_path / "test.db"))
    # API module has a process store; use an isolated store directly through app globals.
    from knowledge_agent import api
    from knowledge_agent.persistence import Store
    api.store = Store(tmp_path / "test.db")
    client = TestClient(app)
    kb = client.post('/api/knowledge-bases', json={'name': '验收库'}).json()
    uploaded = client.post(f"/api/knowledge-bases/{kb['id']}/documents", files={'file': ('policy.md', '报销期限为 30 天。'.encode(), 'text/markdown')})
    assert uploaded.status_code == 200
    assert client.post(f"/api/knowledge-bases/{kb['id']}/build").json()['status'] == 'ready'
    doc_id = uploaded.json()['document_id']
    assert client.get(f'/api/documents/{doc_id}/chunks').json()[0]['text'] == '报销期限为 30 天。'
    session = client.post('/api/sessions', json={'kb_id': kb['id']}).json()
    run_id = client.post(f"/api/sessions/{session['id']}/runs", json={'question': '报销期限是多久？'}).json()['run_id']
    for _ in range(100):
        run = client.get(f'/api/runs/{run_id}').json()
        if run['status'] != 'running':
            break
        time.sleep(.05)
    assert run['result']['status'] == 'completed'
    assert run['result']['citations']
    event_response = client.get(f'/api/runs/{run_id}/events')
    assert event_response.status_code == 200
    event_text = event_response.text
    assert 'run_started' in event_text
    assert 'search_documents' in event_text
    assert 'read_document' in event_text
    assert 'run_finished' in event_text
    replay = client.get(f'/api/runs/{run_id}/events', headers={'Last-Event-ID': '1'})
    assert 'run_started' not in replay.text
    assert 'search_documents' in replay.text
    duplicate = client.post(f"/api/knowledge-bases/{kb['id']}/documents", files={'file': ('policy.md', '报销期限为 30 天。'.encode(), 'text/markdown')})
    assert duplicate.status_code == 200
    assert duplicate.json()['created'] is False
    invalid_date = client.post(f"/api/knowledge-bases/{kb['id']}/documents", data={'version_date': '2026/01/01'}, files={'file': ('other.md', '新的制度'.encode(), 'text/markdown')})
    assert invalid_date.status_code == 400
