from __future__ import annotations

import io
import shutil
import subprocess
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path

from .config import Settings

SUPPORTED_EXTENSIONS = {".txt", ".md", ".markdown", ".pdf", ".docx", ".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".webp"}

@dataclass
class ParseResult:
    text: str
    parser: str
    method: str
    warnings: list[str]
    def to_dict(self): return asdict(self)

def _ocr_image(image: bytes, settings: Settings) -> str:
    binary = shutil.which(settings.tesseract_command) or (settings.tesseract_command if Path(settings.tesseract_command).exists() else None)
    if not binary: raise ValueError("图片/扫描件需要安装 Tesseract OCR，并配置 TESSERACT_COMMAND")
    proc = subprocess.run([binary, "stdin", "stdout", "-l", settings.tesseract_lang], input=image, capture_output=True, timeout=settings.mineru_timeout_seconds, check=False)
    if proc.returncode: raise ValueError(f"Tesseract OCR 失败：{proc.stderr.decode(errors='replace')[:240]}")
    return proc.stdout.decode("utf-8", errors="replace")

def _mineru_pdf(data: bytes, filename: str, settings: Settings) -> str:
    binary = shutil.which(settings.mineru_command or "") or settings.mineru_command
    if not binary: return ""
    with tempfile.TemporaryDirectory(prefix="xiaoying-mineru-") as tmp:
        root = Path(tmp); source = root / Path(filename).name; output = root / "output"; source.write_bytes(data)
        args = [binary, "-p", str(source), "-o", str(output)]
        if settings.mineru_server_url: args += ["-b", "vlm-http-client", "-u", settings.mineru_server_url, "-m", settings.mineru_method]
        proc = subprocess.run(args, capture_output=True, text=True, timeout=settings.mineru_timeout_seconds, check=False)
        if proc.returncode: raise ValueError(f"MinerU 解析失败：{proc.stderr[-500:]}")
        markdown = sorted(output.rglob("*.md"))
        if markdown: return "\n\n".join(path.read_text(encoding="utf-8", errors="replace") for path in markdown)
        raise ValueError("MinerU 已运行但没有生成 Markdown 文档")

def parse_attachment(filename: str, data: bytes, settings: Settings) -> ParseResult:
    suffix = Path(filename).suffix.lower()
    if suffix not in SUPPORTED_EXTENSIONS: raise ValueError("支持 TXT、Markdown、PDF、DOCX 和常见图片格式")
    if not data: raise ValueError("文件为空")
    if suffix in {".txt", ".md", ".markdown"}: return ParseResult(data.decode("utf-8-sig", errors="replace"), "text", "local", [])
    if suffix == ".docx":
        from docx import Document
        return ParseResult("\n".join(p.text for p in Document(io.BytesIO(data)).paragraphs), "python-docx", "local", [])
    if suffix == ".pdf":
        if settings.mineru_command: return ParseResult(_mineru_pdf(data, filename, settings).strip(), "MinerU", "mineru", [])
        import pymupdf
        pdf = pymupdf.open(stream=data, filetype="pdf"); text = "\n".join(page.get_text() for page in pdf).strip()
        if text: return ParseResult(text, "PyMuPDF", "pdf_text", [])
        pieces = []
        for page in pdf:
            pixmap = page.get_pixmap(matrix=pymupdf.Matrix(2, 2), alpha=False); pieces.append(_ocr_image(pixmap.tobytes("png"), settings))
        return ParseResult("\n".join(pieces).strip(), "PyMuPDF + Tesseract", "pdf_ocr", ["PDF 未检测到可提取文本，已按页面执行 OCR"])
    return ParseResult(_ocr_image(data, settings).strip(), "Tesseract", "ocr", [])
