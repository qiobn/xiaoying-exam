from knowledge_agent.config import Settings
from knowledge_agent.ingestion import build_kb, sha256, split_chunks
from knowledge_agent.persistence import Store
from knowledge_agent.retrieval import search, terms


def test_chunk_offsets_and_chinese_terms():
    text = "第一段。\n第二段报销期限。"
    chunks = split_chunks(text, 100, 10)
    assert chunks[0][0] == text
    assert chunks[0][1:] == (0, len(text))
    assert "报销" in terms("报销期限")


def test_title_breadcrumb_retrieves_chunk_but_returns_original_text(tmp_path):
    store = Store(tmp_path / "db.sqlite")
    kb = store.create_kb("retrieval")
    original = "员工应至少提前 3 个工作日提交申请。"
    store.add_document(
        kb["id"], "leave.md", "年假申请流程", "2026-01-01", "全体员工", original,
        sha256(original.encode()),
    )
    build_kb(store, kb["id"], Settings(db_path=tmp_path / "db.sqlite"))
    hits = search(store, kb["id"], "年假怎么申请？")
    chunk = store.all_chunks(kb["id"])[0]
    assert hits and hits[0]["text"] == original
    assert "年假申请流程" in chunk["retrieval_text"]
    assert chunk["text"] == original
    assert chunk["start_offset"] == 0
    assert chunk["end_offset"] == len(original)


def test_retrieval_deduplicates_by_chunk_id_before_top_k():
    class DuplicateChunkStore:
        def all_chunks(self, _kb_id):
            row = {
                "id": "stable-chunk", "document_id": "doc", "title": "报销制度",
                "version_date": "2026-01-01", "scope": "全体员工",
                "text": "报销期限为 30 天。", "retrieval_text": "报销制度\n报销期限为 30 天。",
            }
            return [row, dict(row)]

    hits = search(DuplicateChunkStore(), "kb", "报销期限", top_k=1)
    assert len(hits) == 1
    assert hits[0]["chunk_id"] == "stable-chunk"


def test_retrieval_rejects_generic_partial_overlap_for_specific_follow_up(tmp_path):
    from knowledge_agent.rag import KnowledgeRetrievalService

    store = Store(tmp_path / "coverage.sqlite")
    kb = store.create_kb("coverage")
    text = "员工应在费用发生后 30 天内提交报销申请。"
    store.add_document(
        kb["id"], "expense.md", "报销制度", "2026-01-01", "上海办公室", text,
        sha256(text.encode()),
    )
    build_kb(store, kb["id"], Settings(db_path=tmp_path / "coverage.sqlite"))

    settings = Settings(db_path=tmp_path / "coverage.sqlite")
    outcome = KnowledgeRetrievalService(
        store,
        kb["id"],
        settings,
        required_terms=("国外", "出差"),
    ).retrieve("国外出差报销期限")

    assert outcome.hits == []
    assert outcome.diagnostics["required_terms_filtered"] is True


def test_chunk_offsets_preserve_original_crlf_positions():
    text = "  第一段。\r\n第二段报销期限。  "
    chunks = split_chunks(text, 100, 10)
    assert chunks == [("第一段。\r\n第二段报销期限。", 2, len(text) - 2)]
    for piece, start, end in split_chunks(text, 10, 2):
        assert text[start:end] == piece
