import re
import uuid
from typing import Any


_HEADING = re.compile(r"^\s{0,3}(#{1,6})\s+(.+?)\s*#*\s*$")
_FENCE = re.compile(r"^\s{0,3}(`{3,}|~{3,})")
_LIST = re.compile(r"^\s*(?:[-+*]|\d+[.)])\s+")
_TABLE_SEPARATOR = re.compile(r"^\s*\|?\s*:?-{3,}:?\s*(?:\|\s*:?-{3,}:?\s*)+\|?\s*$")


def _line_offsets(text: str) -> list[tuple[int, int, str]]:
    lines = []
    offset = 0
    for line in text.splitlines(keepends=True):
        lines.append((offset, offset + len(line), line))
        offset += len(line)
    if offset < len(text) or not lines:
        lines.append((offset, len(text), text[offset:]))
    return lines


def _is_table_start(lines: list[tuple[int, int, str]], index: int) -> bool:
    if "|" not in lines[index][2]:
        return False
    next_index = index + 1
    return next_index < len(lines) and bool(_TABLE_SEPARATOR.match(lines[next_index][2].rstrip()))


def _markdown_blocks(text: str) -> list[dict[str, Any]]:
    lines = _line_offsets(text)
    blocks: list[dict[str, Any]] = []
    titles: list[str] = []
    index = 0
    while index < len(lines):
        start, end, line = lines[index]
        stripped = line.rstrip("\r\n")
        if not stripped.strip():
            index += 1
            continue

        heading = _HEADING.match(stripped)
        if heading:
            level = len(heading.group(1))
            title = heading.group(2).strip()
            titles = titles[:level - 1]
            titles.append(title)
            blocks.append({"start": start, "end": end, "kind": "heading", "title_path": list(titles)})
            index += 1
            continue

        fence = _FENCE.match(stripped)
        if fence:
            marker = fence.group(1)
            index += 1
            while index < len(lines):
                end = lines[index][1]
                closing = _FENCE.match(lines[index][2].rstrip("\r\n"))
                index += 1
                if closing and closing.group(1)[0] == marker[0] and len(closing.group(1)) >= len(marker):
                    break
            blocks.append({"start": start, "end": end, "kind": "code", "title_path": list(titles)})
            continue

        if _is_table_start(lines, index):
            index += 2
            end = lines[index - 1][1]
            while index < len(lines) and lines[index][2].strip() and "|" in lines[index][2]:
                end = lines[index][1]
                index += 1
            blocks.append({"start": start, "end": end, "kind": "table", "title_path": list(titles)})
            continue

        if _LIST.match(stripped):
            index += 1
            while index < len(lines):
                candidate = lines[index][2].rstrip("\r\n")
                if not candidate.strip() or _HEADING.match(candidate) or _FENCE.match(candidate):
                    break
                if _is_table_start(lines, index):
                    break
                if not _LIST.match(candidate) and not candidate[:1].isspace():
                    break
                end = lines[index][1]
                index += 1
            blocks.append({"start": start, "end": end, "kind": "list", "title_path": list(titles)})
            continue

        index += 1
        while index < len(lines):
            candidate = lines[index][2].rstrip("\r\n")
            if (not candidate.strip() or _HEADING.match(candidate) or _FENCE.match(candidate)
                    or _LIST.match(candidate) or _is_table_start(lines, index)):
                break
            end = lines[index][1]
            index += 1
        blocks.append({"start": start, "end": end, "kind": "paragraph", "title_path": list(titles)})
    return blocks


def _token_count(content: str) -> int:
    return len(re.findall(r"[\u4e00-\u9fff]|[A-Za-z_][A-Za-z0-9_]*|\d+(?:\.\d+)?|[^\s]", content))


def _chunk_record(
    text: str, start: int, end: int, index: int, title_path: list[str], document_id: str | None
) -> dict[str, Any]:
    raw = text[start:end]
    content = raw.strip()
    leading = len(raw) - len(raw.lstrip())
    trailing = len(raw) - len(raw.rstrip())
    chunk_id = str(uuid.uuid4())
    record: dict[str, Any] = {
        "id": chunk_id, "chunk_id": chunk_id, "chunk_index": index,
        "title_path": list(title_path), "content": content,
        "start_char": start + leading, "end_char": end - trailing,
        "token_count": _token_count(content),
    }
    if document_id is not None:
        record["document_id"] = document_id
    return record


def split_text(
    text: str,
    chunk_size: int = 700,
    overlap: int = 100,
    document_id: str | None = None,
) -> list[dict[str, Any]]:
    if chunk_size <= 0 or overlap < 0 or overlap >= chunk_size:
        raise ValueError("chunk_size 必须大于 0，overlap 必须在 0 和 chunk_size 之间")
    if not text.strip():
        return []

    blocks = _markdown_blocks(text)
    chunks: list[dict[str, Any]] = []
    current_start: int | None = None
    current_end = 0
    current_path: list[str] = []

    def emit(start: int, end: int, title_path: list[str]) -> None:
        if text[start:end].strip():
            chunks.append(_chunk_record(text, start, end, len(chunks), title_path, document_id))

    for block in blocks:
        block_start, block_end = block["start"], block["end"]
        if block["kind"] == "heading" and current_start is not None:
            emit(current_start, current_end, current_path)
            current_start, current_end = block_start, block_end
            current_path = block["title_path"]
            continue
        if block_end - block_start > chunk_size and block["kind"] == "paragraph":
            if current_start is not None:
                emit(current_start, current_end, current_path)
                current_start = None
            step = chunk_size - overlap
            position = block_start
            while position < block_end:
                piece_end = min(position + chunk_size, block_end)
                emit(position, piece_end, block["title_path"])
                if piece_end == block_end:
                    break
                position += step
            continue

        if current_start is None:
            current_start, current_end = block_start, block_end
            current_path = block["title_path"]
        elif block_end - current_start <= chunk_size:
            current_end = block_end
        else:
            emit(current_start, current_end, current_path)
            current_start, current_end = block_start, block_end
            current_path = block["title_path"]

    if current_start is not None:
        emit(current_start, current_end, current_path)
    return chunks
