from io import BytesIO
from pathlib import Path


SUPPORTED_EXTENSIONS = {".pdf", ".md", ".markdown", ".txt"}


class DocumentParseError(ValueError):
    pass


def parse_document(filename: str, content: bytes) -> str:
    extension = Path(filename).suffix.lower()
    if extension not in SUPPORTED_EXTENSIONS:
        raise DocumentParseError("仅支持 PDF、Markdown 和 TXT 文件")
    if not content:
        raise DocumentParseError("文件内容为空")

    if extension == ".pdf":
        try:
            from pypdf import PdfReader
        except ImportError as exc:
            raise DocumentParseError("PDF 解析依赖未安装，请先执行 pip install -r requirements.txt") from exc
        try:
            reader = PdfReader(BytesIO(content))
            text = "\n\n".join(page.extract_text() or "" for page in reader.pages)
        except Exception as exc:
            raise DocumentParseError(f"PDF 解析失败: {exc}") from exc
    else:
        try:
            text = content.decode("utf-8-sig")
        except UnicodeDecodeError as exc:
            raise DocumentParseError("文本文件必须使用 UTF-8 编码") from exc

    normalized = "\n".join(line.rstrip() for line in text.replace("\r\n", "\n").splitlines())
    normalized = "\n".join(line for line in normalized.split("\n") if line.strip())
    if not normalized.strip():
        raise DocumentParseError("文件中没有可提取的文字")
    return normalized.strip()
