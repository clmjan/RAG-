from fastapi.testclient import TestClient

from app import main
from app.config import Settings
from app.db import Database


def client(tmp_path, monkeypatch):
    test_settings = Settings(database_path=tmp_path / "test.db", upload_dir=tmp_path / "uploads")
    monkeypatch.setattr(main, "settings", test_settings)
    monkeypatch.setattr(main, "database", Database(test_settings.database_path))
    with TestClient(main.app) as test_client:
        yield test_client


def test_upload_search_ask_and_refusal(tmp_path, monkeypatch):
    test_client = next(client(tmp_path, monkeypatch))
    content = b"# Python\n\nPython is a programming language.\n\n# Retrieval\n\nRAG uses retrieval and citations."
    uploaded = test_client.post(
        "/documents/upload", files={"file": ("lesson.md", content, "text/markdown")}
    )
    assert uploaded.status_code == 202
    upload = uploaded.json()
    assert upload["status"] == "queued"
    task = test_client.get(f"/processing-tasks/{upload['task_id']}")
    assert task.status_code == 200
    assert task.json()["status"] == "succeeded"
    document = test_client.get(f"/documents/{upload['document_id']}").json()
    assert document["status"] == "ready"
    assert document["chunk_count"] >= 1

    search = test_client.post("/search", json={"query": "What does RAG use?"})
    assert search.status_code == 200
    assert "retrieval" in search.json()["results"][0]["content"]

    answer = test_client.post("/questions", json={"question": "What does RAG use?"})
    assert answer.status_code == 200
    assert answer.json()["refused"] is False
    assert answer.json()["citations"]

    refusal = test_client.post("/questions", json={"question": "What is the capital of Mars?"})
    assert refusal.status_code == 200
    assert refusal.json()["refused"] is True

    assert test_client.get("/health").json() == {"status": "ok"}
    assert test_client.get("/ready").json() == {"status": "ready"}
    assert "X-Request-ID" in test_client.get("/health").headers
    assert test_client.options("/health", headers={"Origin": "http://127.0.0.1:5173"}).headers.get(
        "access-control-allow-origin"
    ) == "http://127.0.0.1:5173"


def test_duplicate_upload_is_rejected(tmp_path, monkeypatch):
    test_client = next(client(tmp_path, monkeypatch))
    content = b"# Lesson\n\nA short lesson."
    payload = {"file": ("lesson.md", content, "text/markdown")}
    assert test_client.post("/documents/upload", files=payload).status_code == 202
    assert test_client.post("/documents/upload", files=payload).status_code == 409


def test_search_methods_document_management_and_evaluation_history(tmp_path, monkeypatch):
    test_client = next(client(tmp_path, monkeypatch))
    content = b"# RAG\n\nRAG uses retrieval and citations."
    uploaded = test_client.post(
        "/documents/upload", files={"file": ("rag.md", content, "text/markdown")}
    )
    assert uploaded.status_code == 202
    document_id = uploaded.json()["document_id"]

    for method in ("bm25", "tfidf"):
        response = test_client.post(
            "/search", json={"query": "What does RAG use?", "retrieval_method": method}
        )
        assert response.status_code == 200
        assert response.json()["retrieval_method"] == method
        assert response.json()["results"]
        assert "title_path" in response.json()["results"][0]
    assert test_client.post("/search", json={"query": "RAG", "retrieval_method": "bad"}).status_code == 422

    assert test_client.get(f"/documents/{document_id}").status_code == 200
    assert test_client.get(f"/documents/{document_id}/chunks").json()
    assert test_client.post("/evaluations/run").status_code == 200
    comparison = test_client.post("/evaluations/compare")
    assert comparison.status_code == 200
    assert {run["retrieval_method"] for run in comparison.json()["runs"]} == {
        "tfidf", "bm25", "embedding", "hybrid",
    }
    history = test_client.get("/evaluations")
    assert history.status_code == 200
    assert len(history.json()) >= 3

    assert test_client.delete(f"/documents/{document_id}").status_code == 200
    assert test_client.post("/search", json={"query": "RAG", "retrieval_method": "bm25"}).json()["results"] == []
    deleted = test_client.get(f"/documents/{document_id}")
    assert deleted.status_code == 200
    assert deleted.json()["status"] == "deleted"
    assert deleted.json()["chunk_count"] == 0
    assert test_client.delete(f"/documents/{document_id}").status_code == 404


def test_new_document_is_searchable_without_restart(tmp_path, monkeypatch):
    test_client = next(client(tmp_path, monkeypatch))
    assert test_client.post("/search", json={"query": "稀有词", "retrieval_method": "bm25"}).json()["results"] == []
    uploaded = test_client.post(
        "/documents/upload",
        files={"file": ("new.md", "# New\n\n稀有词出现在新文档中。".encode(), "text/markdown")},
    )
    assert uploaded.status_code == 202
    results = test_client.post("/search", json={"query": "稀有词", "retrieval_method": "bm25"}).json()["results"]
    assert results and results[0]["filename"] == "new.md"


def test_llm_mode_without_provider_falls_back_to_extractive(tmp_path, monkeypatch):
    test_settings = Settings(
        database_path=tmp_path / "test.db", upload_dir=tmp_path / "uploads",
        generation_method="llm",
    )
    monkeypatch.setattr(main, "settings", test_settings)
    monkeypatch.setattr(main, "database", Database(test_settings.database_path))
    with TestClient(main.app) as test_client:
        uploaded = test_client.post(
            "/documents/upload",
            files={"file": ("lesson.md", b"# RAG\n\nRAG uses retrieval.", "text/markdown")},
        )
        assert uploaded.status_code == 202
        response = test_client.post("/questions", json={"question": "What does RAG use?"})
        assert response.status_code == 200
        body = response.json()
        assert body["generation_method"] == "extractive"
        assert "not configured" in body["generation_error"]
        assert body["refused"] is False


def test_invalid_llm_result_falls_back_to_extractive_with_error_code(tmp_path, monkeypatch):
    test_settings = Settings(
        database_path=tmp_path / "test.db", upload_dir=tmp_path / "uploads",
        generation_method="llm", llm_api_base="https://chat.example/v1", llm_model="chat",
    )
    monkeypatch.setattr(main, "settings", test_settings)
    monkeypatch.setattr(main, "database", Database(test_settings.database_path))

    class InvalidGenerator:
        def generate(self, question, contexts):
            from app.generation import GeneratedResponse
            return GeneratedResponse("unsupported", False, [{"chunk_id": "bad", "quote": "nope"}])

    monkeypatch.setattr(main, "create_generator", lambda method, config: InvalidGenerator())
    with TestClient(main.app) as test_client:
        uploaded = test_client.post(
            "/documents/upload", files={"file": ("lesson.md", b"# RAG\n\nRAG uses retrieval.", "text/markdown")}
        )
        assert uploaded.status_code == 202
        response = test_client.post("/questions", json={"question": "What does RAG use?"})
        body = response.json()
        assert body["generation_method"] == "extractive"
        assert body["generation_error_code"] == "citation_validation_failed"
        assert body["refused"] is False


def test_runtime_settings_are_forwarded_to_optional_provider_factories(tmp_path, monkeypatch):
    test_settings = Settings(
        database_path=tmp_path / "test.db",
        upload_dir=tmp_path / "uploads",
        retrieval_method="embedding",
        embedding_api_base="https://embedding.example/v1",
        embedding_api_key="embedding-key",
        embedding_model="embedding-model",
        generation_method="llm",
        llm_api_base="https://chat.example/v1",
        llm_api_key="chat-key",
        llm_model="chat-model",
    )
    monkeypatch.setattr(main, "settings", test_settings)
    monkeypatch.setattr(main, "database", Database(test_settings.database_path))
    embedding_configs = []
    chat_configs = []

    class FakeEmbeddingClient:
        model = "fake-embedding"

        def embed_texts(self, texts):
            return [[1.0, 0.0] for _ in texts]

    class FakeChatClient:
        def complete_json(self, messages):
            return {"answer": "Supported.", "refused": True, "citations": []}

    monkeypatch.setattr(
        "app.retrieval.create_embedding_client",
        lambda config: embedding_configs.append(config) or FakeEmbeddingClient(),
    )
    monkeypatch.setattr(
        "app.processing.create_embedding_client",
        lambda config: embedding_configs.append(config) or FakeEmbeddingClient(),
    )
    monkeypatch.setattr(
        "app.generation.create_chat_client",
        lambda config: chat_configs.append(config) or FakeChatClient(),
    )

    with TestClient(main.app) as test_client:
        uploaded = test_client.post(
            "/documents/upload",
            files={"file": ("lesson.md", b"# RAG\n\nRAG uses retrieval.", "text/markdown")},
        )
        assert uploaded.status_code == 202

        search = test_client.post("/search", json={"query": "retrieval"})
        assert search.status_code == 200
        answer = test_client.post("/questions", json={"question": "What does RAG use?"})
        assert answer.status_code == 200

    assert len(embedding_configs) == 3
    assert all(config is test_settings for config in embedding_configs)
    assert chat_configs == [test_settings]


def test_upload_validation_happens_before_task_creation(tmp_path, monkeypatch):
    test_settings = Settings(
        database_path=tmp_path / "test.db", upload_dir=tmp_path / "uploads",
        max_upload_bytes=4,
    )
    monkeypatch.setattr(main, "settings", test_settings)
    monkeypatch.setattr(main, "database", Database(test_settings.database_path))
    with TestClient(main.app) as test_client:
        unsupported = test_client.post(
            "/documents/upload", files={"file": ("lesson.docx", b"abc", "application/octet-stream")}
        )
        oversized = test_client.post(
            "/documents/upload", files={"file": ("lesson.txt", b"12345", "text/plain")}
        )
        assert unsupported.status_code == 400
        assert oversized.status_code == 413
        assert test_client.get("/documents").json() == []
        assert test_client.get("/processing-tasks").json() == []


def test_parse_failure_is_persisted_on_document_and_task(tmp_path, monkeypatch):
    test_client = next(client(tmp_path, monkeypatch))
    uploaded = test_client.post(
        "/documents/upload", files={"file": ("broken.txt", b"\xff\xfe", "text/plain")}
    )
    assert uploaded.status_code == 202
    upload = uploaded.json()

    task = test_client.get(f"/processing-tasks/{upload['task_id']}").json()
    document = test_client.get(f"/documents/{upload['document_id']}").json()
    assert task["status"] == "failed"
    assert document["status"] == "failed"
    assert "UTF-8" in task["error_message"]
    assert document["error_message"] == task["error_message"]
    assert document["chunk_count"] == 0
    assert test_client.get(f"/documents/{upload['document_id']}/chunks").json() == []


def test_upload_creates_pending_records_before_worker_runs(tmp_path, monkeypatch):
    test_settings = Settings(database_path=tmp_path / "test.db", upload_dir=tmp_path / "uploads")
    test_database = Database(test_settings.database_path)
    monkeypatch.setattr(main, "settings", test_settings)
    monkeypatch.setattr(main, "database", test_database)
    scheduled = []
    monkeypatch.setattr(
        main, "process_document_task",
        lambda *args: scheduled.append(args),
    )
    with TestClient(main.app) as test_client:
        uploaded = test_client.post(
            "/documents/upload", files={"file": ("lesson.md", b"# Lesson", "text/markdown")}
        )
        upload = uploaded.json()
        assert uploaded.status_code == 202
        assert scheduled
        assert test_client.get(f"/documents/{upload['document_id']}").json()["status"] == "pending"
        assert test_client.get(f"/processing-tasks/{upload['task_id']}").json()["status"] == "queued"
        assert test_client.get(f"/documents/{upload['document_id']}/chunks").json() == []


def test_status_api_reports_background_failure_without_partial_index(tmp_path, monkeypatch):
    test_settings = Settings(
        database_path=tmp_path / "test.db", upload_dir=tmp_path / "uploads",
        embedding_api_base="https://embedding.example/v1", embedding_model="model",
    )
    test_database = Database(test_settings.database_path)
    monkeypatch.setattr(main, "settings", test_settings)
    monkeypatch.setattr(main, "database", test_database)

    class FailingEmbeddingClient:
        model = "model"

        def embed_texts(self, texts):
            raise RuntimeError("embedding unavailable")

    monkeypatch.setattr(
        "app.processing.create_embedding_client", lambda config: FailingEmbeddingClient()
    )
    with TestClient(main.app) as test_client:
        uploaded = test_client.post(
            "/documents/upload",
            files={"file": ("lesson.md", b"# Lesson\n\nSearchable text.", "text/markdown")},
        ).json()
        status = test_client.get(f"/documents/{uploaded['document_id']}/status")
        task = test_client.get(f"/tasks/{uploaded['task_id']}")

        assert status.status_code == 200
        assert status.json()["status"] == "failed"
        assert status.json()["task_status"] == "failed"
        assert "embedding unavailable" in status.json()["error_message"]
        assert task.json()["status"] == "failed"
        assert test_database.get_document_chunks(uploaded["document_id"]) == []
        assert test_database.search_bm25("Searchable") == []
