from langchain_core.messages import AIMessage

from knowledge_agent.config import Settings
from knowledge_agent.semantic_chunking import LLMSemanticChunker, SmartChunker


def test_llm_chunker_uses_returned_line_boundaries_without_rewriting_text(monkeypatch):
    text = "制度总则。\n\n报销应在 30 天内提交。\n\n年假应提前 3 天申请。"

    class FakeChat:
        def __init__(self, **kwargs):
            pass

        def invoke(self, messages):
            return AIMessage(content='''{
              "segments": [
                {"title": "总则", "start_line": 1, "end_line": 1},
                {"title": "报销期限", "start_line": 3, "end_line": 3},
                {"title": "年假申请", "start_line": 5, "end_line": 5}
              ]
            }''')

    monkeypatch.setattr("langchain_openai.ChatOpenAI", FakeChat)
    settings = Settings(
        mode="real",
        api_key="test-key",
        model="fake",
        chunk_strategy="llm",
        chunk_llm_model="fake",
    )

    pieces = LLMSemanticChunker(settings).split(text)

    assert [piece.text for piece in pieces] == [
        "制度总则。",
        "报销应在 30 天内提交。",
        "年假应提前 3 天申请。",
    ]
    assert [piece.section_path for piece in pieces] == ["总则", "报销期限", "年假申请"]
    assert all(piece.method == "llm" for piece in pieces)


def test_llm_chunker_falls_back_to_structured_when_response_is_invalid(monkeypatch):
    text = "第一段。第二段。"

    class FakeChat:
        def __init__(self, **kwargs):
            pass

        def invoke(self, messages):
            return AIMessage(content="not-json")

    monkeypatch.setattr("langchain_openai.ChatOpenAI", FakeChat)
    settings = Settings(
        mode="real",
        api_key="test-key",
        chunk_strategy="llm",
        chunk_llm_model="fake",
        chunk_size=5,
        chunk_overlap=0,
    )

    chunker = LLMSemanticChunker(settings)
    pieces = chunker.split(text)

    assert pieces
    assert all(piece.method == "structured" for piece in pieces)
    assert chunker.last_diagnostics["used"] == "structured"
    assert chunker.last_diagnostics["fallback"] == "llm_response_invalid"


def test_auto_chunker_keeps_table_documents_structured(monkeypatch):
    text = "制度说明。\n\n| 城市 | 额度 |\n| --- | --- |\n| 上海 | 500 |\n" + "补充说明。" * 400

    class UnexpectedChat:
        def __init__(self, **kwargs):
            raise AssertionError("auto mode must not send table text to the LLM")

    monkeypatch.setattr("langchain_openai.ChatOpenAI", UnexpectedChat)
    settings = Settings(
        mode="real",
        api_key="test-key",
        chunk_strategy="auto",
        chunk_llm_min_chars=100,
    )

    pieces = SmartChunker(settings).split(text)

    assert pieces
    assert all(piece.method == "structured" for piece in pieces)
