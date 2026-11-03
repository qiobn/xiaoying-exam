from knowledge_agent.agent import run_agent
from knowledge_agent.config import Settings
from knowledge_agent.ingestion import build_kb, sha256
from knowledge_agent.persistence import Store


def test_conflict_returns_clarification(tmp_path):
    store=Store(tmp_path/'db.sqlite')
    kb=store.create_kb('conflict')
    for item in [('d1.md','30天'),('d2.md','15天')]:
        store.add_document(kb['id'],item[0],'规则', '2026-01-01','未指定',f'报销期限为 {item[1]}。',sha256(item[1].encode()))
    build_kb(store,kb['id'],Settings(db_path=tmp_path/'db.sqlite'))
    result=run_agent(store,kb['id'],'报销期限是多久？',[],Settings(db_path=tmp_path/'db.sqlite'))
    assert result.status == 'needs_clarification'
    assert len(result.citations) >= 2


def test_unrelated_documents_do_not_create_false_conflict(tmp_path):
    store = Store(tmp_path / "db.sqlite")
    kb = store.create_kb("faq")
    store.add_document(kb["id"], "leave.md", "年假 FAQ", "2026-01-01", "全体员工", "员工申请年假应至少提前 3 个工作日提交申请。", sha256(b"leave"))
    store.add_document(kb["id"], "expense.md", "报销制度", "2026-01-01", "全体员工", "员工应在费用发生后 30 天内提交报销申请。", sha256(b"expense"))
    build_kb(store, kb["id"], Settings(db_path=tmp_path / "db.sqlite"))
    result = run_agent(store, kb["id"], "年假需要提前几天申请？", [], Settings(db_path=tmp_path / "db.sqlite"))
    assert result.status == "completed"
    assert len(result.citations) == 1
    assert result.citations[0]["document_id"]


def test_only_same_topic_numeric_evidence_conflicts(tmp_path):
    store = Store(tmp_path / "db.sqlite")
    kb = store.create_kb("conflicts")
    for name, days in [("policy-a.md", 30), ("policy-b.md", 15)]:
        text = f"报销期限为 {days} 天。"
        store.add_document(kb["id"], name, name, "2026-01-01", "未指定", text, sha256(text.encode()))
    build_kb(store, kb["id"], Settings(db_path=tmp_path / "db.sqlite"))
    result = run_agent(store, kb["id"], "报销期限是多久？", [], Settings(db_path=tmp_path / "db.sqlite"))
    assert result.status == "needs_clarification"
    assert len({citation["document_id"] for citation in result.citations}) == 2


def test_context_follow_up_rewrites_query_and_rejects_generic_partial_hit(tmp_path):
    store = Store(tmp_path / "follow-up.sqlite")
    kb = store.create_kb("follow-up")
    text = "员工应在费用发生后 30 天内提交报销申请。"
    store.add_document(
        kb["id"], "expense.md", "上海办公室报销制度", "2026-01-01", "上海办公室",
        text, sha256(text.encode()),
    )
    build_kb(store, kb["id"], Settings(db_path=tmp_path / "follow-up.sqlite"))

    result = run_agent(
        store,
        kb["id"],
        "那国外出差呢？",
        [
            {"role": "user", "content": "报销期限是多久？"},
            {"role": "assistant", "content": "报销期限为 30 天。"},
        ],
        Settings(db_path=tmp_path / "follow-up.sqlite"),
    )

    rewrite = next(item for item in result.trace if item["action"] == "query_rewrite")
    assert rewrite["arguments"]["rewritten"] == "国外出差报销期限"
    assert result.status == "not_found"
    assert result.citations == []
