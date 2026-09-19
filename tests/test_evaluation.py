import hashlib
import uuid

from app.chunking import split_text
from app.config import settings
from app.db import Database, utc_now
from app.evaluation import run_evaluation
from app.parsing import parse_document


def test_evaluation_reports_split_metrics(tmp_path):
    database = Database(tmp_path / "evaluation.db")
    database.init()
    content = b"# RAG\n\nRAG retrieves context from a knowledge base.\n\nThe system should refuse unsupported questions."
    document_id = str(uuid.uuid4())
    chunks = [dict(chunk, document_id=document_id) for chunk in split_text(parse_document("rag.md", content))]
    database.create_document({
        "id": document_id, "filename": "rag.md", "content_type": "text/markdown",
        "size_bytes": len(content), "sha256": hashlib.sha256(content).hexdigest(), "created_at": utc_now(),
    }, chunks)
    cases = [
        {"id": "a", "split": "validation", "question": "What does RAG retrieve?", "source": ["rag.md"], "answer_terms": ["context"]},
        {"id": "b", "split": "validation", "question": "What is the capital of Mars?", "source": [], "should_refuse": True},
        {"id": "c", "split": "test", "question": "What is RAG?", "source": ["rag.md"], "answer_terms": ["knowledge base"]},
        {"id": "d", "split": "test", "question": "Explain quantum computing.", "source": [], "should_refuse": True},
    ]
    result = run_evaluation(database, cases)
    assert result["metrics"]["answerable_count"] == 1
    assert result["metrics"]["refusal_count"] == 1
    assert "recall_at_1" in result["metrics"]
    assert "p95_latency_ms" in result["metrics"]
    assert result["metrics"]["evaluation_split"] == "test"


def test_evaluation_uses_requested_method(tmp_path):
    database = Database(tmp_path / "evaluation.db")
    database.init()
    content = b"# RAG\n\nRAG retrieves context from a knowledge base."
    document_id = str(uuid.uuid4())
    chunks = [dict(chunk, document_id=document_id) for chunk in split_text(parse_document("rag.md", content))]
    database.create_document({
        "id": document_id, "filename": "rag.md", "content_type": "text/markdown",
        "size_bytes": len(content), "sha256": hashlib.sha256(content).hexdigest(), "created_at": utc_now(),
    }, chunks)
    cases = [
        {"id": "v1", "split": "validation", "question": "What does RAG retrieve?", "source": ["rag.md"], "answer_terms": ["context"]},
        {"id": "v2", "split": "validation", "question": "What is Mars?", "source": [], "should_refuse": True},
        {"id": "t1", "split": "test", "question": "What is RAG?", "source": ["rag.md"], "answer_terms": ["knowledge base"]},
        {"id": "t2", "split": "test", "question": "What is Java?", "source": [], "should_refuse": True},
    ]
    for method in ("tfidf", "bm25"):
        result = run_evaluation(database, cases, method=method)
        assert result["retrieval_method"] == method
        assert result["metrics"]["retrieval_method"] == method
        assert all(error["retrieval_method"] == method for error in result["errors"])
        history = database.list_evaluation_runs()
        assert history[0]["retrieval_method"] == method
