import hashlib
import uuid

import pytest

from app.answering import answer_question
from app.chunking import split_text
from app.config import Settings
from app.db import Database, utc_now
from app.generation import GeneratedResponse, LLMGenerator, create_generator, validate_generated_response
from app.providers import (
    ProviderError,
    create_chat_client,
    create_embedding_client,
)
from app.retrieval import EmbeddingRetriever, SearchResult, create_retriever


def make_database(tmp_path):
    database = Database(tmp_path / "adapters.db")
    database.init()
    documents = [
        ("alpha.md", b"# Alpha\n\nAlpha discusses retrieval."),
        ("beta.md", b"# Beta\n\nBeta discusses deployment."),
    ]
    for filename, content in documents:
        document_id = str(uuid.uuid4())
        chunks = [dict(chunk, document_id=document_id) for chunk in split_text(content.decode())]
        database.create_document({
            "id": document_id, "filename": filename, "content_type": "text/markdown",
            "size_bytes": len(content), "sha256": hashlib.sha256(content).hexdigest(),
            "created_at": utc_now(),
        }, chunks)
    return database


def test_provider_factories_use_explicit_settings():
    config = Settings(
        embedding_api_base="https://embedding.example/v1/",
        embedding_api_key="embedding-key",
        embedding_model="embedding-model",
        embedding_timeout_seconds=12,
        llm_api_base="https://chat.example/v1/",
        llm_api_key="chat-key",
        llm_model="chat-model",
        llm_timeout_seconds=34,
    )

    embedding = create_embedding_client(config)
    chat = create_chat_client(config)

    assert embedding is not None
    assert (embedding.api_base, embedding.api_key, embedding.model, embedding.timeout) == (
        "https://embedding.example/v1", "embedding-key", "embedding-model", 12,
    )
    assert chat is not None
    assert (chat.api_base, chat.api_key, chat.model, chat.timeout) == (
        "https://chat.example/v1", "chat-key", "chat-model", 34,
    )


def test_provider_factories_return_none_when_provider_is_incomplete():
    config = Settings(
        embedding_api_base="https://embedding.example/v1",
        llm_model="chat-model",
    )

    assert create_embedding_client(config) is None
    assert create_chat_client(config) is None


def test_retriever_and_generator_forward_explicit_provider_config(monkeypatch, tmp_path):
    database = make_database(tmp_path)
    config = Settings(
        embedding_api_base="https://embedding.example/v1",
        embedding_model="embedding-model",
        llm_api_base="https://chat.example/v1",
        llm_model="chat-model",
    )
    embedding_configs = []
    chat_configs = []

    class FakeEmbeddingClient:
        model = "fake-embedding"

    class FakeChatClient:
        pass

    monkeypatch.setattr(
        "app.retrieval.create_embedding_client",
        lambda received: embedding_configs.append(received) or FakeEmbeddingClient(),
    )
    monkeypatch.setattr(
        "app.generation.create_chat_client",
        lambda received: chat_configs.append(received) or FakeChatClient(),
    )

    retriever = create_retriever("embedding", database, config=config)
    generator = create_generator("llm", config=config)

    assert retriever.client.model == "fake-embedding"
    assert generator.client.__class__ is FakeChatClient
    assert embedding_configs == [config]
    assert chat_configs == [config]


class FakeEmbeddingClient:
    model = "fake-embedding"

    def __init__(self):
        self.calls = []

    def embed_texts(self, texts):
        self.calls.append(list(texts))
        vectors = []
        for text in texts:
            lowered = text.lower()
            vectors.append([1.0, 0.0] if "alpha" in lowered or "retrieval" in lowered else [0.0, 1.0])
        return vectors


def test_embedding_retriever_caches_document_vectors(tmp_path):
    database = make_database(tmp_path)
    client = FakeEmbeddingClient()
    retriever = EmbeddingRetriever(database, client)

    first = retriever.search("retrieval", top_k=1)
    second = retriever.search("retrieval", top_k=1)

    assert first[0].filename == "alpha.md"
    assert second[0].filename == "alpha.md"
    assert len(client.calls) == 3
    assert len(client.calls[0]) == 2
    assert client.calls[1] == ["retrieval"]
    assert client.calls[2] == ["retrieval"]
    assert len(database.get_embeddings("fake-embedding")) == 2


def test_llm_citation_validation_requires_exact_retrieved_quote():
    result = SearchResult("chunk-1", "doc-1", "note.md", 0, "A supported statement.", 0.9)
    valid = GeneratedResponse(
        answer="A supported statement.", refused=False,
        citations=[{"chunk_id": "chunk-1", "quote": "A supported statement."}],
    )
    citations = validate_generated_response(valid, [result])
    assert citations[0]["filename"] == "note.md"

    invalid = GeneratedResponse(
        answer="An unsupported statement.", refused=False,
        citations=[{"chunk_id": "chunk-1", "quote": "Not in the source."}],
    )
    with pytest.raises(ProviderError):
        validate_generated_response(invalid, [result])


def test_llm_generator_normalizes_structured_response():
    result = SearchResult("chunk-1", "doc-1", "note.md", 0, "A supported statement.", 0.9)

    class FakeChatClient:
        def complete_json(self, messages):
            assert "exact chunk_id" in messages[0]["content"]
            return {
                "answer": "A supported statement.", "refused": False,
                "citations": [{"chunk_id": "chunk-1", "quote": "A supported statement."}],
            }

    generated = LLMGenerator(FakeChatClient()).generate("What is supported?", [result])
    assert generated.answer == "A supported statement."
    assert generated.refused is False
    assert generated.citations == [{"chunk_id": "chunk-1", "quote": "A supported statement."}]


def test_llm_refusal_does_not_need_citations():
    result = SearchResult("chunk-1", "doc-1", "note.md", 0, "A statement.", 0.9)
    generated = GeneratedResponse(answer="", refused=True, citations=[])
    assert validate_generated_response(generated, [result]) == []


def test_embedding_provider_failure_falls_back_to_bm25(tmp_path):
    database = make_database(tmp_path)

    class FailingClient(FakeEmbeddingClient):
        def embed_texts(self, texts):
            raise ProviderError("offline")

    results = EmbeddingRetriever(database, FailingClient()).search("retrieval", top_k=1)
    assert results and results[0].filename == "alpha.md"
