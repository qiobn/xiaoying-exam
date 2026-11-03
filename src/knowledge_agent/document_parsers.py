from __future__ import annotations

import csv
import hashlib
import io
import re
import shutil
import subprocess
import tempfile
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from .config import Settings
from .table_utils import count_markdown_tables, rows_to_markdown
from .vision import ImageUnderstandingService

SUPPORTED_EXTENSIONS = {
    ".txt", ".md", ".markdown", ".csv", ".pdf", ".docx",
    ".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".webp",
}
_IMAGE_REL = "{http://schemas.openxmlformats.org/officeDocument/2006/relationships}embed"
_MARKDOWN_IMAGE = re.compile(r"!\[[^\]]*]\([^)]+\)")


@dataclass
class ParseResult:
    text: str
    parser: str
    method: str
    warnings: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self):
        return asdict(self)


def _ocr_image(image: bytes, settings: Settings) -> str:
    binary = shutil.which(settings.tesseract_command) or (
        settings.tesseract_command if Path(settings.tesseract_command).exists() else None
    )
    if not binary:
        raise ValueError("图片/扫描件需要安装 Tesseract OCR，并配置 TESSERACT_COMMAND")
    proc = subprocess.run(
        [binary, "stdin", "stdout", "-l", settings.tesseract_lang],
        input=image,
        capture_output=True,
        timeout=settings.mineru_timeout_seconds,
        check=False,
    )
    if proc.returncode:
        raise ValueError(f"Tesseract OCR 失败：{proc.stderr.decode(errors='replace')[:240]}")
    return proc.stdout.decode("utf-8", errors="replace").strip()


def _vision_image(data: bytes, filename: str, settings: Settings, warnings: list[str]) -> str:
    result = ImageUnderstandingService().describe(data, filename, settings)
    if result.error:
        warnings.append(f"图片视觉分析失败：{result.error}")
    return result.description


def _render_image(
    data: bytes,
    filename: str,
    settings: Settings,
    warnings: list[str],
    *,
    strict: bool,
) -> str:
    description = _vision_image(data, filename, settings, warnings)
    ocr_text = ""
    try:
        ocr_text = _ocr_image(data, settings)
    except (ValueError, OSError, subprocess.SubprocessError) as exc:
        if strict and not description:
            raise ValueError(str(exc)) from exc
        if not description:
            warnings.append(f"图片 OCR 跳过：{str(exc)[:180]}")
    parts = [f"[图片 {filename}]"]
    if description:
        parts.append(f"图片描述：{description}")
    if ocr_text:
        parts.append(f"图中文字：{ocr_text}")
    if len(parts) == 1:
        if strict:
            raise ValueError("图片未提取到可索引文字或描述，请启用 Tesseract OCR 或视觉模型")
        return ""
    return "\n".join(parts)


def _mineru_pdf(data: bytes, filename: str, settings: Settings) -> str:
    binary = shutil.which(settings.mineru_command or "") or settings.mineru_command
    if not binary:
        return ""
    with tempfile.TemporaryDirectory(prefix="xiaoying-mineru-") as tmp:
        root = Path(tmp)
        source = root / Path(filename).name
        output = root / "output"
        source.write_bytes(data)
        args = [binary, "-p", str(source), "-o", str(output)]
        if settings.mineru_server_url:
            args += [
                "-b", "vlm-http-client",
                "-u", settings.mineru_server_url,
                "-m", settings.mineru_method,
            ]
        proc = subprocess.run(
            args,
            capture_output=True,
            text=True,
            timeout=settings.mineru_timeout_seconds,
            check=False,
        )
        if proc.returncode:
            raise ValueError(f"MinerU 解析失败：{proc.stderr[-500:]}")
        markdown = sorted(output.rglob("*.md"))
        if markdown:
            return "\n\n".join(
                path.read_text(encoding="utf-8", errors="replace")
                for path in markdown
            )
        raise ValueError("MinerU 已运行但没有生成 Markdown 文档")


def _docx_paragraph_images(paragraph, document) -> list[bytes]:
    images: list[bytes] = []
    for blip in paragraph._p.xpath(".//a:blip"):
        rel_id = blip.get(_IMAGE_REL)
        if not rel_id:
            continue
        relationship = document.part.rels.get(rel_id)
        if relationship is None:
            continue
        part = relationship.target_part
        blob = getattr(part, "blob", None)
        if blob:
            images.append(blob)
    return images


def _parse_docx(data: bytes, settings: Settings) -> ParseResult:
    from docx import Document
    from docx.table import Table

    document = Document(io.BytesIO(data))
    paragraphs = iter(document.paragraphs)
    tables = iter(document.tables)
    blocks: list[str] = []
    warnings: list[str] = []
    table_count = 0
    image_count = 0
    seen_images: set[str] = set()

    for element in document.element.body:
        if not isinstance(element.tag, str):
            continue
        if element.tag.endswith("}p"):
            paragraph = next(paragraphs, None)
            if paragraph is None:
                continue
            style_name = getattr(paragraph.style, "name", "") or ""
            heading_match = re.search(r"(?:Heading|标题)\s*([1-6])", style_name, re.IGNORECASE)
            heading_prefix = "#" * int(heading_match.group(1)) + " " if heading_match else ""
            parts = [f"{heading_prefix}{paragraph.text.strip()}".strip()]
            if settings.parse_embedded_images:
                for image in _docx_paragraph_images(paragraph, document):
                    digest = hashlib.sha256(image).hexdigest()
                    if digest in seen_images:
                        continue
                    seen_images.add(digest)
                    if image_count >= settings.max_embedded_images:
                        warnings.append(f"文档内图片超过 {settings.max_embedded_images} 张，其余图片未解析")
                        break
                    image_count += 1
                    rendered = _render_image(
                        image,
                        f"image-{image_count}",
                        settings,
                        warnings,
                        strict=False,
                    )
                    if rendered:
                        parts.append(rendered)
            content = "\n\n".join(part for part in parts if part)
            if content:
                blocks.append(content)
        elif element.tag.endswith("}tbl"):
            table = next(tables, None)
            if not isinstance(table, Table):
                continue
            rows = [[cell.text for cell in row.cells] for row in table.rows]
            markdown = rows_to_markdown(rows)
            if markdown:
                table_count += 1
                blocks.append(f"### 表格 {table_count}\n\n{markdown}")

    return ParseResult(
        text="\n\n".join(blocks).strip(),
        parser="python-docx",
        method="local",
        warnings=warnings,
        metadata={"table_count": table_count, "image_count": image_count},
    )


def _pdf_images(
    page,
    document,
    settings: Settings,
    warnings: list[str],
    remaining: int,
) -> tuple[list[str], int]:
    if not settings.parse_embedded_images or remaining <= 0:
        return [], 0
    rendered: list[str] = []
    seen: set[int] = set()
    count = 0
    for item in page.get_images(full=True):
        xref = int(item[0])
        if xref in seen:
            continue
        seen.add(xref)
        if count >= remaining:
            warnings.append(f"PDF 内图片超过 {settings.max_embedded_images} 张，其余图片未解析")
            break
        try:
            image = document.extract_image(xref)
            content = _render_image(
                image["image"],
                f"page-{page.number + 1}-image-{xref}.{image.get('ext', 'png')}",
                settings,
                warnings,
                strict=False,
            )
            if content:
                rendered.append(content)
            count += 1
        except Exception as exc:  # noqa: BLE001 - one image must not fail the PDF
            warnings.append(f"PDF 图片 {xref} 解析失败：{str(exc)[:160]}")
    return rendered, count


def _parse_pdf(data: bytes, filename: str, settings: Settings) -> ParseResult:
    import pymupdf

    if settings.mineru_command:
        text = _mineru_pdf(data, filename, settings).strip()
        return ParseResult(
            text=text,
            parser="MinerU",
            method="mineru",
            metadata={
                "table_count": count_markdown_tables(text),
                "image_count": len(_MARKDOWN_IMAGE.findall(text)),
            },
        )

    document = pymupdf.open(stream=data, filetype="pdf")
    pages: list[str] = []
    warnings: list[str] = []
    table_count = 0
    image_count = 0
    page_count = 0
    try:
        for page in document:
            page_count += 1
            blocks = [page.get_text().strip()]
            if settings.parse_pdf_tables:
                try:
                    for index, table in enumerate(page.find_tables().tables, start=1):
                        markdown = table.to_markdown().strip()
                        if markdown:
                            table_count += 1
                            blocks.append(
                                f"### 第 {page.number + 1} 页表格 {index}\n\n{markdown}"
                            )
                except Exception as exc:  # noqa: BLE001 - text extraction remains available
                    warnings.append(
                        f"第 {page.number + 1} 页表格识别失败：{str(exc)[:160]}"
                    )
            if settings.vision_enabled:
                rendered, count = _pdf_images(
                    page,
                    document,
                    settings,
                    warnings,
                    settings.max_embedded_images - image_count,
                )
                blocks.extend(rendered)
                image_count += count
            content = "\n\n".join(block for block in blocks if block)
            if content:
                pages.append(content)
    finally:
        document.close()

    text = "\n\n".join(pages).strip()
    if text:
        return ParseResult(
            text=text,
            parser="PyMuPDF",
            method="pdf_structured",
            warnings=warnings,
            metadata={
                "table_count": table_count,
                "image_count": image_count,
                "page_count": page_count,
            },
        )

    pieces = []
    ocr_document = pymupdf.open(stream=data, filetype="pdf")
    try:
        for page in ocr_document:
            pixmap = page.get_pixmap(matrix=pymupdf.Matrix(2, 2), alpha=False)
            pieces.append(_ocr_image(pixmap.tobytes("png"), settings))
    finally:
        ocr_document.close()
    return ParseResult(
        text="\n".join(pieces).strip(),
        parser="PyMuPDF + Tesseract",
        method="pdf_ocr",
        warnings=[*warnings, "PDF 未检测到可提取文本，已按页面执行 OCR"],
        metadata={"table_count": 0, "image_count": 0, "page_count": page_count},
    )


def _parse_csv(data: bytes) -> ParseResult:
    text = data.decode("utf-8-sig", errors="replace")
    rows = list(csv.reader(io.StringIO(text)))
    markdown = rows_to_markdown(rows)
    if not markdown:
        raise ValueError("CSV 中没有可索引表格")
    return ParseResult(
        text=markdown,
        parser="csv",
        method="table",
        metadata={"table_count": 1, "image_count": 0},
    )


def parse_attachment(filename: str, data: bytes, settings: Settings) -> ParseResult:
    suffix = Path(filename).suffix.lower()
    if suffix not in SUPPORTED_EXTENSIONS:
        raise ValueError("支持 TXT、Markdown、CSV、PDF、DOCX 和常见图片格式")
    if not data:
        raise ValueError("文件为空")
    if suffix in {".txt", ".md", ".markdown"}:
        text = data.decode("utf-8-sig", errors="replace")
        return ParseResult(
            text=text,
            parser="text",
            method="local",
            metadata={
                "table_count": count_markdown_tables(text),
                "image_count": len(_MARKDOWN_IMAGE.findall(text)),
            },
        )
    if suffix == ".csv":
        return _parse_csv(data)
    if suffix == ".docx":
        return _parse_docx(data, settings)
    if suffix == ".pdf":
        return _parse_pdf(data, filename, settings)
    warnings: list[str] = []
    text = _render_image(data, Path(filename).name, settings, warnings, strict=True)
    vision_used = "图片描述：" in text
    ocr_used = "图中文字：" in text
    if vision_used and ocr_used:
        parser, method = "Vision + Tesseract", "vision_ocr"
    elif vision_used:
        parser, method = "Vision", "vision"
    else:
        parser, method = "Tesseract", "ocr"
    return ParseResult(
        text=text,
        parser=parser,
        method=method,
        warnings=warnings,
        metadata={
            "table_count": count_markdown_tables(text),
            "image_count": 1,
            "vision_used": vision_used,
            "ocr_used": ocr_used,
        },
    )
