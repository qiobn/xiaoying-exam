import io

from docx import Document
from langchain_core.messages import AIMessage

from knowledge_agent.chunking import StructuredChunker
from knowledge_agent.config import Settings
from knowledge_agent.parsers import parse_attachment


def test_docx_parser_preserves_tables_as_markdown():
    document = Document()
    document.add_paragraph("报销标准如下。")
    table = document.add_table(rows=3, cols=3)
    table.cell(0, 0).text = "城市"
    table.cell(0, 1).text = "职级"
    table.cell(0, 2).text = "额度"
    table.cell(1, 0).text = "上海"
    table.cell(1, 1).text = "P5"
    table.cell(1, 2).text = "500"
    table.cell(2, 0).text = "北京"
    table.cell(2, 1).text = "P6"
    table.cell(2, 2).text = "800"
    payload = io.BytesIO()
    document.save(payload)

    result = parse_attachment(
        "policy.docx",
        payload.getvalue(),
        Settings(mode="mock", vision_enabled=False),
    )

    assert "| 城市 | 职级 | 额度 |" in result.text
    assert "| 上海 | P5 | 500 |" in result.text
    assert result.metadata["table_count"] == 1


def test_structured_chunker_keeps_markdown_table_in_one_piece():
    text = (
        "# 差旅标准\n\n"
        "| 城市 | 职级 | 额度 |\n"
        "| --- | --- | --- |\n"
        "| 上海 | P5 | 500 |\n"
        "| 北京 | P6 | 800 |"
    )

    pieces = StructuredChunker(size=20, overlap=0).split(text)

    assert len(pieces) == 1
    assert "| 上海 | P5 | 500 |" in pieces[0].text
    assert "| 北京 | P6 | 800 |" in pieces[0].text


def test_image_parser_combines_vision_description_and_ocr(monkeypatch):
    class FakeChat:
        def __init__(self, **kwargs):
            pass

        def invoke(self, messages):
            return AIMessage(content="柱状图展示上海和北京的报销额度。")

    monkeypatch.setattr("langchain_openai.ChatOpenAI", FakeChat)
    monkeypatch.setattr("knowledge_agent.document_parsers._ocr_image", lambda data, settings: "上海 500")
    settings = Settings(
        mode="real",
        api_key="test-key",
        vision_enabled=True,
        vision_model="vision-test",
    )

    result = parse_attachment("chart.png", b"fake-png", settings)

    assert result.metadata["image_count"] == 1
    assert "柱状图展示上海和北京的报销额度" in result.text
    assert "上海 500" in result.text
