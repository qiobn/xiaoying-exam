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


def test_explicit_location_scope_resolves_policy_conflict(tmp_path):
    store = Store(tmp_path / "scope.sqlite")
    kb = store.create_kb("scope")
    store.add_document(
        kb["id"], "mainland.md", "员工报销制度（大陆）", "2026-01-01",
        "中国大陆普通员工", "报销期限为 30 天。", sha256(b"mainland"),
    )
    store.add_document(
        kb["id"], "shanghai.md", "上海地区报销补充规则", "2026-05-01",
        "上海地区员工", "报销期限为 15 天。", sha256(b"shanghai"),
    )
    build_kb(store, kb["id"], Settings(db_path=tmp_path / "scope.sqlite"))

    result = run_agent(
        store,
        kb["id"],
        "上海办公室的报销期限是多久？",
        [],
        Settings(db_path=tmp_path / "scope.sqlite"),
    )

    assert result.status == "completed"
    assert {citation["title"] for citation in result.citations} == {"上海地区报销补充规则"}


def test_same_grade_values_from_table_and_prose_are_not_false_conflict(tmp_path):
    store = Store(tmp_path / "same-grade.sqlite")
    kb = store.create_kb("same-grade")
    prose = "深圳 P5 员工住宿上限为 600 元/晚，P6 员工住宿上限为 900 元/晚。"
    table = "| 城市 | 职级 | 住宿上限 |\n| --- | --- | --- |\n| 深圳 | P5 | 600 |\n| 深圳 | P6 | 900 |\n| 上海 | P6 | 800 |"
    store.add_document(kb["id"], "rules.md", "深圳差旅规则", "2026-01-01", "深圳员工", prose, sha256(b"prose"))
    store.add_document(kb["id"], "table.md", "住宿标准表", "2026-01-01", "深圳员工", table, sha256(b"table"))
    build_kb(store, kb["id"], Settings(db_path=tmp_path / "same-grade.sqlite"))

    result = run_agent(
        store,
        kb["id"],
        "深圳P6住宿上限是多少？",
        [],
        Settings(db_path=tmp_path / "same-grade.sqlite"),
    )

    assert result.status == "completed"
    assert "900" in result.answer
