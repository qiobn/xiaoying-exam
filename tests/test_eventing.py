from knowledge_agent.config import Settings
from knowledge_agent.eventing import BufferedRunEventPublisher
from knowledge_agent.persistence import Store


def test_token_events_are_batched_but_audit_events_are_immediate(tmp_path):
    store = Store(tmp_path / "events.db")
    kb = store.create_kb("events")
    session = store.create_session(kb["id"], "events")
    run_id = store.create_run(session["id"], "question")
    publisher = BufferedRunEventPublisher(
        store,
        run_id,
        Settings(stream_token_batch_chars=10, stream_token_batch_ms=10000),
    )

    publisher("agent_stage", {"title": "安全检查", "detail": "通过", "status": "done"})
    publisher("token", {"delta": "你好", "phase": "answer"})
    publisher("token", {"delta": "，这是", "phase": "answer"})
    publisher("token", {"delta": "流式回答", "phase": "answer"})
    publisher.flush()

    events = store.events(run_id)
    tokens = [event for event in events if event["event_type"] == "token"]
    assert events[0]["event_type"] == "agent_stage"
    assert "".join(event["delta"] for event in tokens) == "你好，这是流式回答"
    assert 1 <= len(tokens) < 3
    assert all(event["batched"] is True for event in tokens)
