from knowledge_agent.agent import _strip_answer_metadata, make_tools, run_agent
from knowledge_agent.config import Settings
from knowledge_agent.ingestion import build_kb
from knowledge_agent.persistence import Store
from knowledge_agent.query_rewrite import ConversationQueryRewriter
from knowledge_agent.routing import RetrievalGate


def test_standalone_weather_question_is_not_rewritten_from_previous_chat():
    history = [{"role": "user", "content": "你能干什么"}]

    result = ConversationQueryRewriter().rewrite("今天天气怎么样", history)

    assert result.applied is False
    assert result.query == "今天天气怎么样"


def test_weather_is_out_of_scope_instead_of_not_found():
    decision = RetrievalGate().decide(
        "今天天气怎么样",
        [{"role": "user", "content": "你能干什么"}],
    )

    assert decision.needed is False
    assert decision.kind == "out_of_scope"


def test_out_of_scope_answer_is_distinct_from_empty_knowledge_base(tmp_path):
    store = Store(tmp_path / "scope.db")
    kb = store.create_kb("scope")

    result = run_agent(
        store,
        kb["id"],
        "今天天气怎么样",
        [{"role": "user", "content": "你能干什么"}],
        Settings(db_path=tmp_path / "scope.db"),
    )

    assert result.status == "out_of_scope"
    assert result.citations == []
    assert "知识库" in result.answer
    assert "没有找到" not in result.answer


def test_operation_question_uses_hidden_builtin_guide(tmp_path):
    store = Store(tmp_path / "operations.db")
    kb = store.create_kb("operations")

    result = run_agent(
        store,
        kb["id"],
        "怎么上传文件并构建知识库？",
        [],
        Settings(db_path=tmp_path / "operations.db"),
    )

    assert result.status == "completed"
    assert result.citations == []
    assert "上传" in result.answer
    assert "构建" in result.answer
    assert any(item.get("action") == "operation_guide" for item in result.trace)


def test_answer_metadata_is_removed_but_natural_body_is_kept():
    answer = "**第三次讲座**介绍了工程管理研究。\n\n来源：某文档（2026-01-01）"

    assert _strip_answer_metadata(answer) == "第三次讲座介绍了工程管理研究。"


def test_read_document_selects_the_chunk_matching_the_specific_entity(tmp_path):
    store = Store(tmp_path / "read-ranking.db")
    kb = store.create_kb("read-ranking")
    text = (
        "第一次讲座介绍材料研究。\n\n"
        "第二次讲座介绍火灾模拟。\n\n"
        "第三次讲座介绍工程管理与价值共创。"
    )
    doc_id, _created = store.add_document(
        kb["id"], "lectures.md", "讲座记录", "2026-01-01", "全体员工",
        text, "hash-lectures",
    )
    build_kb(store, kb["id"], Settings(db_path=tmp_path / "read-ranking.db", chunk_size=20, chunk_overlap=0))
    tools = make_tools(
        store,
        kb["id"],
        Settings(db_path=tmp_path / "read-ranking.db"),
        "第三次讲座讲了什么？",
        "第三次讲座 工程管理",
    )
    read_tool = next(tool for tool in tools if tool.name == "read_document")

    payload = read_tool.invoke({"document_id": doc_id, "query": "第三次讲座 工程管理"})

    assert "第三次讲座" in payload
    assert "第一次讲座" not in payload
    assert "第二次讲座" not in payload
