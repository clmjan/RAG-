from app.chunking import split_text


def test_split_text_has_overlap_and_preserves_content():
    text = "第一段内容。" * 80
    chunks = split_text(text, chunk_size=100, overlap=20)
    assert len(chunks) > 1
    assert all(chunk["content"] for chunk in chunks)
    assert chunks[0]["content"][-20:] in chunks[1]["content"]


def test_invalid_chunking_parameters_are_rejected():
    try:
        split_text("text", chunk_size=10, overlap=10)
    except ValueError as exc:
        assert "overlap" in str(exc)
    else:
        raise AssertionError("expected ValueError")


def test_markdown_chunks_track_heading_hierarchy_and_metadata():
    text = (
        "# Course\n\nOverview.\n\n## Retrieval\n\n"
        "RAG combines 中文检索 with English ranking.\n\n"
        "### Notes\n\nFinal paragraph."
    )
    chunks = split_text(text, chunk_size=65, overlap=10, document_id="doc-1")

    assert len(chunks) >= 2
    assert chunks[0]["title_path"] == ["Course"]
    assert any(chunk["title_path"] == ["Course", "Retrieval"] for chunk in chunks)
    assert any(chunk["title_path"] == ["Course", "Retrieval", "Notes"] for chunk in chunks)
    assert all(chunk["chunk_id"] == chunk["id"] for chunk in chunks)
    assert all(chunk["document_id"] == "doc-1" for chunk in chunks)
    assert all(chunk["token_count"] > 0 for chunk in chunks)
    assert all(text[chunk["start_char"]:chunk["end_char"]] == chunk["content"] for chunk in chunks)


def test_code_block_and_table_are_never_split_mid_block():
    code = "```python\ndef retrieve(query):\n    return [\"中文\", query]\n```"
    table = "| Name | Value |\n| --- | --- |\n| RAG | 检索 |"
    text = f"# Mixed\n\nIntro.\n\n{code}\n\n{table}\n\nEnd."
    chunks = split_text(text, chunk_size=45, overlap=5)

    code_chunks = [chunk for chunk in chunks if "```" in chunk["content"]]
    table_chunks = [chunk for chunk in chunks if "| --- | --- |" in chunk["content"]]
    assert len(code_chunks) == 1
    assert code in code_chunks[0]["content"]
    assert len(table_chunks) == 1
    assert table in table_chunks[0]["content"]
