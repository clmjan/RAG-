import uuid

from fastapi.testclient import TestClient

from app import main
from app.config import Settings
from app.db import Database


def observable_client(tmp_path, monkeypatch, **overrides):
    config = Settings(
        database_path=tmp_path / "observability.db",
        upload_dir=tmp_path / "uploads",
        **overrides,
    )
    database = Database(config.database_path)
    monkeypatch.setattr(main, "settings", config)
    monkeypatch.setattr(main, "database", database)
    return config, database, TestClient(main.app)


def test_every_request_gets_correlated_request_id(tmp_path, monkeypatch):
    _, database, test_client = observable_client(tmp_path, monkeypatch)
    with test_client:
        response = test_client.get("/health")

    request_id = response.headers["x-request-id"]
    assert str(uuid.UUID(request_id)) == request_id
    request_log = next(item for item in database.list_request_logs() if item["request_id"] == request_id)
    assert request_log["method"] == "GET"
    assert request_log["path"] == "/health"
    assert request_log["status_code"] == 200
    assert request_log["duration_ms"] >= 0


def test_document_processing_log_contains_timings_and_upload_correlation(tmp_path, monkeypatch):
    _, database, test_client = observable_client(tmp_path, monkeypatch)
    with test_client:
        response = test_client.post(
            "/documents/upload",
            files={"file": ("lesson.md", b"# Lesson\n\nRAG uses retrieval.", "text/markdown")},
        )
        upload = response.json()

    processing_log = database.list_processing_logs()[0]
    assert processing_log["request_id"] == response.headers["x-request-id"]
    assert processing_log["task_id"] == upload["task_id"]
    assert processing_log["document_id"] == upload["document_id"]
    assert processing_log["filename"] == "lesson.md"
    assert processing_log["size_bytes"] > 0
    assert processing_log["status"] == "succeeded"
    assert processing_log["chunk_count"] > 0
    assert processing_log["duration_ms"] >= processing_log["index_duration_ms"] >= 0
    assert processing_log["embedding_duration_ms"] >= 0
    assert processing_log["failure_reason"] is None


def test_search_log_records_method_latency_result_count_and_fallback(tmp_path, monkeypatch):
    _, database, test_client = observable_client(tmp_path, monkeypatch)
    with test_client:
        test_client.post(
            "/documents/upload",
            files={"file": ("lesson.md", b"# Lesson\n\nRAG uses retrieval.", "text/markdown")},
        )
        response = test_client.post("/search", json={
            "query": "retrieval", "retrieval_method": "embedding", "top_k": 3,
        })

    search_log = next(
        item for item in database.list_search_logs()
        if item["request_id"] == response.headers["x-request-id"]
    )
    assert search_log["query"] == "retrieval"
    assert search_log["retrieval_method"] == "embedding"
    assert search_log["effective_method"] == "bm25"
    assert search_log["top_k"] == 3
    assert search_log["result_count"] == len(response.json()["results"])
    assert search_log["fallback_reason"] == "provider_not_configured"
    assert search_log["duration_ms"] >= 0


def test_qa_log_is_bounded_and_does_not_store_full_retrieved_content(tmp_path, monkeypatch):
    _, database, test_client = observable_client(
        tmp_path, monkeypatch, log_text_max_chars=24,
    )
    source = b"# RAG\n\nRAG retrieves context from a private course knowledge base."
    with test_client:
        test_client.post(
            "/documents/upload", files={"file": ("private.md", source, "text/markdown")}
        )
        response = test_client.post("/questions", json={"question": "What does RAG retrieve?"})

    qa_log = database.list_qa_records()[0]
    assert qa_log["request_id"] == response.headers["x-request-id"]
    assert len(qa_log["question"]) <= 24
    assert len(qa_log["answer"]) <= 24
    assert qa_log["retrieval_method"] == "bm25"
    assert qa_log["effective_method"] == "bm25"
    assert qa_log["duration_ms"] >= 0
    assert qa_log["generation_method"] == "extractive"
    assert qa_log["retrieval"]
    assert "content" not in qa_log["retrieval"][0]
    assert set(qa_log["retrieval"][0]) == {
        "chunk_id", "document_id", "filename", "chunk_index", "score",
    }


def test_failed_processing_is_observable_without_file_content(tmp_path, monkeypatch):
    _, database, test_client = observable_client(tmp_path, monkeypatch, log_text_max_chars=30)
    with test_client:
        response = test_client.post(
            "/documents/upload", files={"file": ("broken.txt", b"\xff\xfe", "text/plain")}
        )

    processing_log = database.list_processing_logs()[0]
    assert processing_log["status"] == "failed"
    assert processing_log["failure_reason"]
    assert len(processing_log["failure_reason"]) <= 30
    assert processing_log["chunk_count"] == 0
    assert processing_log["request_id"] == response.headers["x-request-id"]
