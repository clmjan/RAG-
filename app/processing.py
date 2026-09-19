"""Persistent document-processing tasks."""

from pathlib import Path
import logging
import time
import uuid

from .chunking import split_text
from .config import Settings
from .db import Database, utc_now
from .parsing import parse_document
from .providers import create_embedding_client
from .observability import log_event, reset_request_id, safe_text, set_request_id


def process_document_task(
    database: Database,
    config: Settings,
    document_id: str,
    task_id: str,
    source_path: Path,
    request_id: str | None = None,
) -> None:
    """Parse, chunk, and index one document after its upload response is sent."""
    token = set_request_id(request_id or f"task-{task_id}")
    started = time.perf_counter()
    embedding_duration_ms = 0.0
    index_duration_ms = 0.0
    chunk_count = 0
    failure_reason = None
    status = "failed"
    document = database.get_document(document_id) or {}
    filename = document.get("filename", source_path.name)
    size_bytes = int(document.get("size_bytes", 0))
    if not database.start_processing_task(task_id):
        reset_request_id(token)
        return
    try:
        content = source_path.read_bytes()
        size_bytes = len(content)
        text = parse_document(source_path.name, content)
        chunks = split_text(
            text, config.chunk_size, config.chunk_overlap, document_id=document_id
        )
        if not chunks:
            raise ValueError("文档未生成任何可索引片段")
        chunk_count = len(chunks)
        embeddings = []
        embedding_client = create_embedding_client(config)
        if embedding_client is not None:
            embedding_started = time.perf_counter()
            vectors = embedding_client.embed_texts([chunk["content"] for chunk in chunks])
            embedding_duration_ms = (time.perf_counter() - embedding_started) * 1000
            if len(vectors) != len(chunks):
                raise ValueError("Embedding 返回数量与文档片段数量不一致")
            embeddings = [
                {
                    "chunk_id": chunk["id"], "model": embedding_client.model,
                    "dimensions": len(vector), "vector": vector,
                }
                for chunk, vector in zip(chunks, vectors)
            ]
        index_started = time.perf_counter()
        completed = database.complete_processing_task(task_id, document_id, chunks, embeddings)
        index_duration_ms = (time.perf_counter() - index_started) * 1000
        if not completed:
            raise RuntimeError("处理任务在索引写入前已失效")
        status = "succeeded"
    except Exception as exc:
        failure_reason = safe_text(str(exc) or type(exc).__name__, config.log_text_max_chars)
        database.fail_processing_task(task_id, failure_reason or type(exc).__name__)
    finally:
        duration_ms = (time.perf_counter() - started) * 1000
        record = {
            "id": str(uuid.uuid4()), "request_id": request_id, "task_id": task_id,
            "document_id": document_id, "filename": safe_text(filename, 255) or "unknown",
            "size_bytes": size_bytes, "status": status, "duration_ms": round(duration_ms, 3),
            "chunk_count": chunk_count, "index_duration_ms": round(index_duration_ms, 3),
            "embedding_duration_ms": round(embedding_duration_ms, 3),
            "failure_reason": failure_reason, "created_at": utc_now(),
        }
        try:
            database.create_processing_log(record)
        except Exception as log_error:
            log_event("processing_log_write_failed", logging.ERROR, error=type(log_error).__name__)
        log_event(
            "document_processing", status=status, task_id=task_id, document_id=document_id,
            filename=record["filename"], size_bytes=size_bytes, duration_ms=record["duration_ms"],
            chunk_count=chunk_count, index_duration_ms=record["index_duration_ms"],
            embedding_duration_ms=record["embedding_duration_ms"], failure_reason=failure_reason,
        )
        reset_request_id(token)
