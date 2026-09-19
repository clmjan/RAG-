import hashlib
import uuid

from fastapi.testclient import TestClient

from app import main
from app.chunking import split_text
from app.config import Settings
from app.db import Database, utc_now
from app.retrieval import (
    Bm25Retriever, EmbeddingRetriever, HybridRetriever, TfidfRetriever,
    create_retriever,
)
from app.providers import ProviderError


def make_database(tmp_path):
    database = Database(tmp_path / "retrieval.db")
    database.init()
    document_ids = {}
    for filename, content in (
        ("alpha.md", "# Alpha\n\nAlpha covers retrieval and indexing."),
        ("beta.md", "# Beta\n\nBeta covers deployment and operations."),
    ):
        document_id = str(uuid.uuid4())
        document_ids[filename] = document_id
        raw = content.encode()
        chunks = [dict(chunk, document_id=document_id) for chunk in split_text(content)]
        database.create_document({
            "id": document_id, "filename": filename, "content_type": "text/markdown",
            "size_bytes": len(raw), "sha256": hashlib.sha256(raw).hexdigest(),
            "created_at": utc_now(),
        }, chunks)
    return database, document_ids


class FakeEmbeddingClient:
    model = "fake-v1"

    def __init__(self, dimensions=2):
        self.dimensions = dimensions
        self.calls = []

    def embed_texts(self, texts):
        self.calls.append(list(texts))
        return [
            ([1.0, 0.0] if "retrieval" in text.lower() or "alpha" in text.lower() else [0.0, 1.0])
            + [0.0] * (self.dimensions - 2)
            for text in texts
        ]


def test_all_retrievers_return_common_status_and_highlights(tmp_path):
    database, _ = make_database(tmp_path)
    methods = [
        Bm25Retriever(database),
        TfidfRetriever(database.get_chunks()),
        EmbeddingRetriever(database, FakeEmbeddingClient()),
        HybridRetriever(database, Settings()),
    ]
    # Hybrid is tested with an explicitly injected embedding client to avoid a provider dependency.
    methods[-1].embedding = EmbeddingRetriever(database, FakeEmbeddingClient())

    for retriever in methods:
        outcome = retriever.search_with_status("retrieval", top_k=1, max_candidates=5)
        assert hasattr(outcome, "status")
        assert len(outcome.results) == 1
        assert outcome.results[0].matched_terms
        assert "<mark>" in outcome.results[0].highlight


def test_filters_min_score_and_max_candidates(tmp_path):
    database, document_ids = make_database(tmp_path)
    retriever = Bm25Retriever(database)

    outcome = retriever.search_with_status(
        "deployment", top_k=5, min_score=0.5, max_candidates=1,
        document_id=document_ids["beta.md"], filename="beta.md",
    )
    assert len(outcome.results) == 1
    assert outcome.results[0].filename == "beta.md"
    assert outcome.results[0].document_id == document_ids["beta.md"]

    assert retriever.search_with_status("retrieval", filename="beta.md").results == []


def test_embedding_provider_fallback_exposes_status(tmp_path):
    database, _ = make_database(tmp_path)

    class FailingClient(FakeEmbeddingClient):
        def embed_texts(self, texts):
            raise ProviderError("offline", "provider_timeout")

    outcome = EmbeddingRetriever(database, FailingClient()).search_with_status("retrieval")
    assert outcome.results
    assert outcome.status.as_dict() == {
        "requested_method": "embedding", "effective_method": "bm25",
        "fallback_reason": "provider_timeout", "embedding_used": False,
    }


def test_hybrid_uses_configured_weights_and_reports_embedding(tmp_path):
    database, _ = make_database(tmp_path)
    config = Settings(hybrid_lexical_weight=0.2, hybrid_semantic_weight=0.8)
    retriever = HybridRetriever(database, config)
    retriever.embedding = EmbeddingRetriever(database, FakeEmbeddingClient())

    outcome = retriever.search_with_status("retrieval", top_k=2)
    assert outcome.status.effective_method == "hybrid"
    assert outcome.status.embedding_used is True
    assert retriever.lexical_weight == 0.2
    assert retriever.semantic_weight == 0.8


def test_embedding_cache_invalidates_when_vector_dimensions_change(tmp_path):
    database, _ = make_database(tmp_path)
    client = FakeEmbeddingClient(dimensions=2)
    retriever = EmbeddingRetriever(database, client)
    retriever.search("retrieval")
    calls_before = len(client.calls)

    client.dimensions = 3
    retriever.search("retrieval")
    assert len(client.calls) >= calls_before + 2
    assert {row["dimensions"] for row in database.get_embeddings(client.model)} == {3}


def test_empty_database_and_empty_query_are_safe(tmp_path):
    database = Database(tmp_path / "empty.db")
    database.init()
    for method in ("bm25", "tfidf", "embedding", "hybrid"):
        retriever = create_retriever(method, database, config=Settings())
        outcome = retriever.search_with_status("")
        assert outcome.results == []
        assert outcome.status.requested_method == method


def test_search_api_returns_status_and_applies_filename_filter(tmp_path, monkeypatch):
    settings = Settings(database_path=tmp_path / "api.db", upload_dir=tmp_path / "uploads")
    monkeypatch.setattr(main, "settings", settings)
    monkeypatch.setattr(main, "database", Database(settings.database_path))
    with TestClient(main.app) as client:
        first = client.post(
            "/documents/upload", files={"file": ("alpha.md", b"# Alpha\n\nretrieval", "text/markdown")}
        ).json()
        second = client.post(
            "/documents/upload", files={"file": ("beta.md", b"# Beta\n\ndeployment", "text/markdown")}
        ).json()
        assert client.get(f"/tasks/{first['task_id']}").json()["status"] == "succeeded"
        assert client.get(f"/tasks/{second['task_id']}").json()["status"] == "succeeded"

        response = client.post("/search", json={
            "query": "deployment", "filename": "beta.md", "min_score": 0.1,
            "max_candidates": 2,
        })
        body = response.json()
        assert response.status_code == 200
        assert set(body["status"]) == {
            "requested_method", "effective_method", "fallback_reason", "embedding_used",
        }
        assert body["results"] and all(item["filename"] == "beta.md" for item in body["results"])
