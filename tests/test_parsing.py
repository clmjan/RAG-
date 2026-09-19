import pytest

from app.parsing import DocumentParseError, parse_document


def test_markdown_is_normalized():
    assert parse_document("note.md", b"# Title\r\n\r\nhello  \r\n") == "# Title\nhello"


def test_unsupported_file_is_rejected():
    with pytest.raises(DocumentParseError):
        parse_document("note.docx", b"content")

