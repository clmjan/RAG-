import json
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator


SCHEMA = """
CREATE TABLE IF NOT EXISTS documents (
    id TEXT PRIMARY KEY,
    filename TEXT NOT NULL,
    content_type TEXT,
    size_bytes INTEGER NOT NULL,
    sha256 TEXT NOT NULL UNIQUE,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    status TEXT NOT NULL DEFAULT 'ready',
    error_message TEXT,
    chunk_count INTEGER NOT NULL DEFAULT 0,
    CHECK(status IN ('pending', 'processing', 'ready', 'failed', 'deleted'))
);

CREATE TABLE IF NOT EXISTS processing_tasks (
    id TEXT PRIMARY KEY,
    document_id TEXT NOT NULL REFERENCES documents(id),
    status TEXT NOT NULL DEFAULT 'queued',
    error_message TEXT,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    started_at TEXT,
    completed_at TEXT,
    CHECK(status IN ('queued', 'running', 'succeeded', 'failed'))
);

CREATE INDEX IF NOT EXISTS idx_processing_tasks_document_id
ON processing_tasks(document_id);

CREATE TABLE IF NOT EXISTS chunks (
    id TEXT PRIMARY KEY,
    document_id TEXT NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
    chunk_index INTEGER NOT NULL,
    content TEXT NOT NULL,
    start_char INTEGER NOT NULL,
    end_char INTEGER NOT NULL,
    title_path TEXT NOT NULL DEFAULT '[]',
    token_count INTEGER NOT NULL DEFAULT 0,
    UNIQUE(document_id, chunk_index)
);

CREATE TABLE IF NOT EXISTS qa_records (
    id TEXT PRIMARY KEY,
    question TEXT NOT NULL,
    answer TEXT NOT NULL,
    refused INTEGER NOT NULL,
    retrieval_json TEXT NOT NULL,
    citations_json TEXT NOT NULL,
    generation_method TEXT NOT NULL DEFAULT 'extractive',
    generation_error TEXT,
    request_id TEXT,
    retrieval_method TEXT,
    effective_method TEXT,
    refusal_reason TEXT,
    duration_ms REAL,
    created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS request_logs (
    request_id TEXT PRIMARY KEY,
    method TEXT NOT NULL,
    path TEXT NOT NULL,
    status_code INTEGER NOT NULL,
    duration_ms REAL NOT NULL,
    error_message TEXT,
    created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS processing_logs (
    id TEXT PRIMARY KEY,
    request_id TEXT,
    task_id TEXT NOT NULL,
    document_id TEXT NOT NULL,
    filename TEXT NOT NULL,
    size_bytes INTEGER NOT NULL,
    status TEXT NOT NULL,
    duration_ms REAL NOT NULL,
    chunk_count INTEGER NOT NULL,
    index_duration_ms REAL NOT NULL,
    embedding_duration_ms REAL NOT NULL,
    failure_reason TEXT,
    created_at TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_processing_logs_task_id ON processing_logs(task_id);

CREATE TABLE IF NOT EXISTS search_logs (
    id TEXT PRIMARY KEY,
    request_id TEXT,
    query TEXT NOT NULL,
    retrieval_method TEXT NOT NULL,
    effective_method TEXT NOT NULL,
    top_k INTEGER NOT NULL,
    result_count INTEGER NOT NULL,
    duration_ms REAL NOT NULL,
    fallback_reason TEXT,
    created_at TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_search_logs_request_id ON search_logs(request_id);

CREATE TABLE IF NOT EXISTS chunk_embeddings (
    chunk_id TEXT PRIMARY KEY REFERENCES chunks(id) ON DELETE CASCADE,
    model TEXT NOT NULL,
    dimensions INTEGER NOT NULL,
    vector_json TEXT NOT NULL,
    created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS evaluation_runs (
    id TEXT PRIMARY KEY,
    retrieval_method TEXT NOT NULL DEFAULT 'bm25',
    total INTEGER NOT NULL,
    retrieval_hit_rate REAL NOT NULL,
    answer_accuracy REAL NOT NULL,
    error_count INTEGER NOT NULL,
    errors_json TEXT NOT NULL,
    metrics_json TEXT NOT NULL DEFAULT '{}',
    config_json TEXT NOT NULL DEFAULT '{}',
    created_at TEXT NOT NULL
);

CREATE VIRTUAL TABLE IF NOT EXISTS chunk_fts USING fts5(
    chunk_id UNINDEXED,
    search_text,
    tokenize = 'unicode61'
);
"""


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


class Database:
    def __init__(self, path: Path):
        self.path = Path(path)

    @contextmanager
    def connection(self) -> Iterator[sqlite3.Connection]:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        connection = sqlite3.connect(self.path, timeout=30)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys = ON")
        connection.execute("PRAGMA busy_timeout = 30000")
        try:
            yield connection
            connection.commit()
        finally:
            connection.close()

    def init(self) -> None:
        with self.connection() as connection:
            connection.executescript(SCHEMA)
            document_columns = {
                row["name"] for row in connection.execute("PRAGMA table_info(documents)")
            }
            if "updated_at" not in document_columns:
                connection.execute("ALTER TABLE documents ADD COLUMN updated_at TEXT")
                connection.execute("UPDATE documents SET updated_at = created_at")
            if "status" not in document_columns:
                connection.execute(
                    "ALTER TABLE documents ADD COLUMN status TEXT NOT NULL DEFAULT 'ready'"
                )
            if "error_message" not in document_columns:
                connection.execute("ALTER TABLE documents ADD COLUMN error_message TEXT")
            connection.execute("UPDATE documents SET status = 'ready' WHERE status = 'succeeded'")
            chunk_columns = {row["name"] for row in connection.execute("PRAGMA table_info(chunks)")}
            if "title_path" not in chunk_columns:
                connection.execute("ALTER TABLE chunks ADD COLUMN title_path TEXT NOT NULL DEFAULT '[]'")
            if "token_count" not in chunk_columns:
                connection.execute("ALTER TABLE chunks ADD COLUMN token_count INTEGER NOT NULL DEFAULT 0")
            task_sql = connection.execute(
                "SELECT sql FROM sqlite_master WHERE type = 'table' AND name = 'processing_tasks'"
            ).fetchone()
            if task_sql and "'queued'" not in (task_sql["sql"] or ""):
                connection.execute("ALTER TABLE processing_tasks RENAME TO processing_tasks_legacy")
                connection.execute(
                    """CREATE TABLE processing_tasks (
                    id TEXT PRIMARY KEY,
                    document_id TEXT NOT NULL REFERENCES documents(id),
                    status TEXT NOT NULL DEFAULT 'queued', error_message TEXT,
                    created_at TEXT NOT NULL, updated_at TEXT NOT NULL,
                    started_at TEXT, completed_at TEXT,
                    CHECK(status IN ('queued', 'running', 'succeeded', 'failed'))
                    )"""
                )
                connection.execute(
                    """INSERT INTO processing_tasks
                    SELECT id, document_id,
                    CASE status WHEN 'pending' THEN 'queued' WHEN 'processing' THEN 'running'
                    WHEN 'ready' THEN 'succeeded' ELSE 'failed' END,
                    error_message, created_at, updated_at, started_at, completed_at
                    FROM processing_tasks_legacy"""
                )
                connection.execute("DROP TABLE processing_tasks_legacy")
                connection.execute("DROP INDEX IF EXISTS idx_processing_tasks_document_id")
                connection.execute("CREATE INDEX IF NOT EXISTS idx_processing_tasks_document_id ON processing_tasks(document_id)")
            # Migrate task names from the first async implementation.
            connection.execute("UPDATE processing_tasks SET status = 'queued' WHERE status = 'pending'")
            connection.execute("UPDATE processing_tasks SET status = 'running' WHERE status = 'processing'")
            connection.execute("UPDATE processing_tasks SET status = 'succeeded' WHERE status = 'ready'")
            connection.execute("UPDATE processing_tasks SET status = 'failed', error_message = COALESCE(error_message, 'document deleted') WHERE status = 'deleted'")
            connection.execute(
                """UPDATE documents SET status = 'processing'
                WHERE id IN (SELECT document_id FROM processing_tasks WHERE status = 'running')
                AND status = 'ready'"""
            )
            connection.execute(
                """UPDATE documents SET status = 'pending'
                WHERE id IN (SELECT document_id FROM processing_tasks WHERE status = 'queued')
                AND status = 'ready'"""
            )
            columns = {row["name"] for row in connection.execute("PRAGMA table_info(evaluation_runs)")}
            if "metrics_json" not in columns:
                connection.execute(
                    "ALTER TABLE evaluation_runs ADD COLUMN metrics_json TEXT NOT NULL DEFAULT '{}'"
                )
            if "config_json" not in columns:
                connection.execute(
                    "ALTER TABLE evaluation_runs ADD COLUMN config_json TEXT NOT NULL DEFAULT '{}'"
                )
            qa_columns = {row["name"] for row in connection.execute("PRAGMA table_info(qa_records)")}
            if "generation_method" not in qa_columns:
                connection.execute(
                    "ALTER TABLE qa_records ADD COLUMN generation_method TEXT NOT NULL DEFAULT 'extractive'"
                )
            if "generation_error" not in qa_columns:
                connection.execute("ALTER TABLE qa_records ADD COLUMN generation_error TEXT")
            for column, definition in (
                ("request_id", "TEXT"), ("retrieval_method", "TEXT"),
                ("effective_method", "TEXT"), ("refusal_reason", "TEXT"),
                ("duration_ms", "REAL"),
            ):
                if column not in qa_columns:
                    connection.execute(f"ALTER TABLE qa_records ADD COLUMN {column} {definition}")
            evaluation_columns = {
                row["name"] for row in connection.execute("PRAGMA table_info(evaluation_runs)")
            }
            if "retrieval_method" not in evaluation_columns:
                connection.execute(
                    "ALTER TABLE evaluation_runs ADD COLUMN retrieval_method TEXT NOT NULL DEFAULT 'bm25'"
                )
            chunk_ids = {row[0] for row in connection.execute("SELECT id FROM chunks")}
            indexed_ids = {row[0] for row in connection.execute("SELECT chunk_id FROM chunk_fts")}
            if chunk_ids != indexed_ids:
                self._rebuild_fts_index(connection)

    @staticmethod
    def _rebuild_fts_index(connection: sqlite3.Connection) -> None:
        from .retrieval import build_index_text

        connection.execute("DELETE FROM chunk_fts")
        rows = connection.execute("SELECT id, content FROM chunks").fetchall()
        connection.executemany(
            "INSERT INTO chunk_fts (chunk_id, search_text) VALUES (?, ?)",
            [(row["id"], build_index_text(row["content"])) for row in rows],
        )

    def rebuild_fts_index(self) -> None:
        with self.connection() as connection:
            self._rebuild_fts_index(connection)

    @staticmethod
    def _insert_chunks(
        connection: sqlite3.Connection, chunks: list[dict[str, Any]]
    ) -> None:
        if not chunks:
            return
        connection.executemany(
            """INSERT INTO chunks
                (id, document_id, chunk_index, content, start_char, end_char, title_path, token_count)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
            [
                (chunk["id"], chunk["document_id"], chunk["chunk_index"], chunk["content"],
                 chunk["start_char"], chunk["end_char"], json.dumps(chunk.get("title_path", []), ensure_ascii=False),
                 int(chunk.get("token_count", len(chunk["content"]))))
                for chunk in chunks
            ],
        )
        from .retrieval import build_index_text

        connection.executemany(
            "INSERT INTO chunk_fts (chunk_id, search_text) VALUES (?, ?)",
            [(chunk["id"], build_index_text(chunk["content"])) for chunk in chunks],
        )

    def create_document(self, document: dict[str, Any], chunks: list[dict[str, Any]]) -> None:
        """Create an already processed document (used by imports and fixtures)."""
        timestamp = document.get("updated_at", document["created_at"])
        with self.connection() as connection:
            connection.execute(
                """INSERT INTO documents
                (id, filename, content_type, size_bytes, sha256, created_at, updated_at,
                 status, error_message, chunk_count)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    document["id"], document["filename"], document.get("content_type"),
                    document["size_bytes"], document["sha256"], document["created_at"], timestamp,
                    document.get("status", "ready"), document.get("error_message"), len(chunks),
                ),
            )
            self._insert_chunks(connection, chunks)

    def create_document_task(
        self, document: dict[str, Any], task: dict[str, Any]
    ) -> None:
        """Atomically create a pending document and its processing task."""
        with self.connection() as connection:
            connection.execute(
                """INSERT INTO documents
                (id, filename, content_type, size_bytes, sha256, created_at, updated_at,
                 status, error_message, chunk_count)
                VALUES (?, ?, ?, ?, ?, ?, ?, 'pending', NULL, 0)""",
                (
                    document["id"], document["filename"], document.get("content_type"),
                    document["size_bytes"], document["sha256"], document["created_at"],
                    document.get("updated_at", document["created_at"]),
                ),
            )
            connection.execute(
                """INSERT INTO processing_tasks
                (id, document_id, status, error_message, created_at, updated_at,
                 started_at, completed_at)
                VALUES (?, ?, 'queued', NULL, ?, ?, NULL, NULL)""",
                (
                    task["id"], document["id"], task["created_at"],
                    task.get("updated_at", task["created_at"]),
                ),
            )

    def start_processing_task(self, task_id: str) -> bool:
        timestamp = utc_now()
        with self.connection() as connection:
            task = connection.execute(
                "SELECT document_id FROM processing_tasks WHERE id = ? AND status = 'queued'",
                (task_id,),
            ).fetchone()
            if not task:
                return False
            updated = connection.execute(
                """UPDATE documents
                SET status = 'processing', error_message = NULL, updated_at = ?
                WHERE id = ? AND status = 'pending'""",
                (timestamp, task["document_id"]),
            )
            if updated.rowcount != 1:
                return False
            connection.execute(
                """UPDATE processing_tasks
                SET status = 'running', error_message = NULL, started_at = ?, updated_at = ?
                WHERE id = ?""",
                (timestamp, timestamp, task_id),
            )
            return True

    def complete_processing_task(
        self, task_id: str, document_id: str, chunks: list[dict[str, Any]],
        embeddings: list[dict[str, Any]] | None = None,
    ) -> bool:
        timestamp = utc_now()
        with self.connection() as connection:
            task = connection.execute(
                """SELECT id FROM processing_tasks
                WHERE id = ? AND document_id = ? AND status = 'running'""",
                (task_id, document_id),
            ).fetchone()
            document = connection.execute(
                "SELECT id FROM documents WHERE id = ? AND status = 'processing'",
                (document_id,),
            ).fetchone()
            if not task or not document:
                return False
            self._insert_chunks(connection, chunks)
            if embeddings:
                connection.executemany(
                    """INSERT INTO chunk_embeddings
                    (chunk_id, model, dimensions, vector_json, created_at)
                    VALUES (?, ?, ?, ?, ?)
                    ON CONFLICT(chunk_id) DO UPDATE SET model = excluded.model,
                    dimensions = excluded.dimensions, vector_json = excluded.vector_json,
                    created_at = excluded.created_at""",
                    [
                        (item["chunk_id"], item["model"], item["dimensions"],
                         json.dumps(item["vector"]), item.get("created_at", timestamp))
                        for item in embeddings
                    ],
                )
            connection.execute(
                """UPDATE documents
                SET status = 'ready', chunk_count = ?, error_message = NULL, updated_at = ?
                WHERE id = ?""",
                (len(chunks), timestamp, document_id),
            )
            connection.execute(
                """UPDATE processing_tasks
                SET status = 'succeeded', error_message = NULL, completed_at = ?, updated_at = ?
                WHERE id = ?""",
                (timestamp, timestamp, task_id),
            )
            return True

    def fail_processing_task(self, task_id: str, error_message: str) -> bool:
        timestamp = utc_now()
        message = error_message[:2000]
        with self.connection() as connection:
            task = connection.execute(
                """SELECT document_id FROM processing_tasks
                WHERE id = ? AND status IN ('queued', 'running')""",
                (task_id,),
            ).fetchone()
            if not task:
                return False
            connection.execute(
                """UPDATE documents
                SET status = 'failed', error_message = ?, updated_at = ?
                WHERE id = ? AND status IN ('pending', 'processing')""",
                (message, timestamp, task["document_id"]),
            )
            connection.execute(
                """UPDATE processing_tasks
                SET status = 'failed', error_message = ?, completed_at = ?, updated_at = ?
                WHERE id = ?""",
                (message, timestamp, timestamp, task_id),
            )
            return True

    def find_document_by_hash(self, sha256: str) -> dict[str, Any] | None:
        with self.connection() as connection:
            row = connection.execute("SELECT * FROM documents WHERE sha256 = ?", (sha256,)).fetchone()
            return dict(row) if row else None

    def delete_document(self, document_id: str) -> bool:
        timestamp = utc_now()
        with self.connection() as connection:
            document = connection.execute(
                "SELECT status FROM documents WHERE id = ?", (document_id,)
            ).fetchone()
            if not document or document["status"] == "deleted":
                return False
            connection.execute(
                "DELETE FROM chunk_fts WHERE chunk_id IN (SELECT id FROM chunks WHERE document_id = ?)",
                (document_id,),
            )
            connection.execute("DELETE FROM chunks WHERE document_id = ?", (document_id,))
            connection.execute(
                """UPDATE documents
                SET status = 'deleted', chunk_count = 0, error_message = NULL, updated_at = ?
                WHERE id = ?""",
                (timestamp, document_id),
            )
            connection.execute(
                """UPDATE processing_tasks
                SET status = 'failed', error_message = 'document deleted',
                    completed_at = COALESCE(completed_at, ?), updated_at = ?
                WHERE document_id = ? AND status IN ('queued', 'running')""",
                (timestamp, timestamp, document_id),
            )
            return True

    def get_index_signature(self) -> tuple[str, int, int, str | None]:
        with self.connection() as connection:
            row = connection.execute(
                "SELECT COUNT(*) AS chunk_count, COALESCE(SUM(LENGTH(content)), 0) AS content_size, "
                "MAX(id) AS last_chunk_id FROM chunks"
            ).fetchone()
            return str(self.path), int(row["chunk_count"]), int(row["content_size"]), row["last_chunk_id"]

    def get_document(self, document_id: str) -> dict[str, Any] | None:
        with self.connection() as connection:
            row = connection.execute("SELECT * FROM documents WHERE id = ?", (document_id,)).fetchone()
            return dict(row) if row else None

    def get_processing_task(self, task_id: str) -> dict[str, Any] | None:
        with self.connection() as connection:
            row = connection.execute(
                "SELECT * FROM processing_tasks WHERE id = ?", (task_id,)
            ).fetchone()
            return dict(row) if row else None

    def list_processing_tasks(
        self, limit: int = 50, document_id: str | None = None
    ) -> list[dict[str, Any]]:
        with self.connection() as connection:
            if document_id:
                rows = connection.execute(
                    """SELECT * FROM processing_tasks WHERE document_id = ?
                    ORDER BY created_at DESC LIMIT ?""",
                    (document_id, limit),
                ).fetchall()
            else:
                rows = connection.execute(
                    "SELECT * FROM processing_tasks ORDER BY created_at DESC LIMIT ?", (limit,)
                ).fetchall()
            result = [dict(row) for row in rows]
            for row in result:
                try:
                    row["title_path"] = json.loads(row.get("title_path") or "[]")
                except (TypeError, json.JSONDecodeError):
                    row["title_path"] = []
            return result

    def recover_processing_tasks(self) -> list[dict[str, Any]]:
        """Re-queue tasks interrupted by a process restart and return queued work."""
        timestamp = utc_now()
        with self.connection() as connection:
            connection.execute(
                """UPDATE processing_tasks SET status = 'queued', updated_at = ?,
                started_at = NULL, error_message = NULL
                WHERE status = 'running'""",
                (timestamp,),
            )
            connection.execute(
                """UPDATE documents SET status = 'pending', updated_at = ?, error_message = NULL
                WHERE status = 'processing'""",
                (timestamp,),
            )
            rows = connection.execute(
                "SELECT * FROM processing_tasks WHERE status = 'queued' ORDER BY created_at"
            ).fetchall()
            result = [dict(row) for row in rows]
            for row in result:
                try:
                    row["title_path"] = json.loads(row.get("title_path") or "[]")
                except (TypeError, json.JSONDecodeError):
                    row["title_path"] = []
            return result

    def get_document_chunks(self, document_id: str) -> list[dict[str, Any]]:
        with self.connection() as connection:
            rows = [dict(row) for row in connection.execute(
                "SELECT * FROM chunks WHERE document_id = ? ORDER BY chunk_index", (document_id,)
            ).fetchall()]
            for row in rows:
                try:
                    row["title_path"] = json.loads(row.get("title_path") or "[]")
                except (TypeError, json.JSONDecodeError):
                    row["title_path"] = []
            return rows

    def get_embeddings(self, model: str) -> list[dict[str, Any]]:
        with self.connection() as connection:
            return [dict(row) for row in connection.execute(
                "SELECT chunk_id, model, dimensions, vector_json, created_at "
                "FROM chunk_embeddings WHERE model = ?", (model,)
            ).fetchall()]

    def upsert_embeddings(self, records: list[dict[str, Any]]) -> None:
        if not records:
            return
        with self.connection() as connection:
            connection.executemany(
                """INSERT INTO chunk_embeddings
                (chunk_id, model, dimensions, vector_json, created_at)
                VALUES (?, ?, ?, ?, ?)
                ON CONFLICT(chunk_id) DO UPDATE SET
                    model = excluded.model,
                    dimensions = excluded.dimensions,
                    vector_json = excluded.vector_json,
                    created_at = excluded.created_at""",
                [
                    (
                        record["chunk_id"], record["model"], record["dimensions"],
                        json.dumps(record["vector"]), record.get("created_at", utc_now()),
                    )
                    for record in records
                ],
            )

    def delete_embeddings(self, model: str) -> None:
        with self.connection() as connection:
            connection.execute("DELETE FROM chunk_embeddings WHERE model = ?", (model,))

    def search_bm25(
        self, query: str, top_k: int = 5, document_id: str | None = None,
        filename: str | None = None,
    ) -> list[dict[str, Any]]:
        from .retrieval import build_fts_query

        match_query = build_fts_query(query)
        if not match_query:
            return []
        with self.connection() as connection:
            clauses = ["chunk_fts MATCH ?", "d.status = 'ready'"]
            parameters: list[Any] = [match_query]
            if document_id:
                clauses.append("c.document_id = ?")
                parameters.append(document_id)
            if filename:
                clauses.append("d.filename = ?")
                parameters.append(filename)
            parameters.append(max(1, min(top_k, 200)))
            rows = connection.execute(
                f"""SELECT c.*, d.filename AS filename, bm25(chunk_fts) AS bm25_rank
                FROM chunk_fts
                JOIN chunks c ON c.id = chunk_fts.chunk_id
                JOIN documents d ON d.id = c.document_id
                WHERE {' AND '.join(clauses)}
                ORDER BY bm25_rank ASC, c.chunk_index ASC
                LIMIT ?""",
                parameters,
            ).fetchall()
            result = [dict(row) for row in rows]
            for row in result:
                try:
                    row["title_path"] = json.loads(row.get("title_path") or "[]")
                except (TypeError, json.JSONDecodeError):
                    row["title_path"] = []
            return result

    def list_documents(self) -> list[dict[str, Any]]:
        with self.connection() as connection:
            return [dict(row) for row in connection.execute(
                "SELECT * FROM documents ORDER BY created_at DESC"
            ).fetchall()]

    def get_chunks(self) -> list[dict[str, Any]]:
        with self.connection() as connection:
            result = [dict(row) for row in connection.execute(
                """SELECT c.*, d.filename AS filename
                FROM chunks c JOIN documents d ON d.id = c.document_id
                WHERE d.status = 'ready'
                ORDER BY c.document_id, c.chunk_index"""
            ).fetchall()]
            for row in result:
                try:
                    row["title_path"] = json.loads(row.get("title_path") or "[]")
                except (TypeError, json.JSONDecodeError):
                    row["title_path"] = []
            return result

    def create_qa_record(self, record: dict[str, Any]) -> None:
        with self.connection() as connection:
            connection.execute(
                """INSERT INTO qa_records
                (id, question, answer, refused, retrieval_json, citations_json,
                 generation_method, generation_error, request_id, retrieval_method,
                 effective_method, refusal_reason, duration_ms, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    record["id"], record["question"], record["answer"], int(record["refused"]),
                    json.dumps(record["retrieval"], ensure_ascii=False),
                    json.dumps(record["citations"], ensure_ascii=False),
                    record.get("generation_method", "extractive"), record.get("generation_error"),
                    record.get("request_id"), record.get("retrieval_method"),
                    record.get("effective_method"), record.get("refusal_reason"),
                    record.get("duration_ms"),
                    record["created_at"],
                ),
            )

    def create_request_log(self, record: dict[str, Any]) -> None:
        with self.connection() as connection:
            connection.execute(
                """INSERT INTO request_logs
                (request_id, method, path, status_code, duration_ms, error_message, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?)""",
                (record["request_id"], record["method"], record["path"], record["status_code"],
                 record["duration_ms"], record.get("error_message"), record["created_at"]),
            )

    def create_processing_log(self, record: dict[str, Any]) -> None:
        with self.connection() as connection:
            connection.execute(
                """INSERT INTO processing_logs
                (id, request_id, task_id, document_id, filename, size_bytes, status,
                 duration_ms, chunk_count, index_duration_ms, embedding_duration_ms,
                 failure_reason, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (record["id"], record.get("request_id"), record["task_id"],
                 record["document_id"], record["filename"], record["size_bytes"],
                 record["status"], record["duration_ms"], record["chunk_count"],
                 record["index_duration_ms"], record["embedding_duration_ms"],
                 record.get("failure_reason"), record["created_at"]),
            )

    def create_search_log(self, record: dict[str, Any]) -> None:
        with self.connection() as connection:
            connection.execute(
                """INSERT INTO search_logs
                (id, request_id, query, retrieval_method, effective_method, top_k,
                 result_count, duration_ms, fallback_reason, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (record["id"], record.get("request_id"), record["query"],
                 record["retrieval_method"], record["effective_method"], record["top_k"],
                 record["result_count"], record["duration_ms"], record.get("fallback_reason"),
                 record["created_at"]),
            )

    def list_request_logs(self, limit: int = 100) -> list[dict[str, Any]]:
        with self.connection() as connection:
            return [dict(row) for row in connection.execute(
                "SELECT * FROM request_logs ORDER BY created_at DESC LIMIT ?", (limit,)
            ).fetchall()]

    def list_processing_logs(self, limit: int = 100) -> list[dict[str, Any]]:
        with self.connection() as connection:
            return [dict(row) for row in connection.execute(
                "SELECT * FROM processing_logs ORDER BY created_at DESC LIMIT ?", (limit,)
            ).fetchall()]

    def list_search_logs(self, limit: int = 100) -> list[dict[str, Any]]:
        with self.connection() as connection:
            return [dict(row) for row in connection.execute(
                "SELECT * FROM search_logs ORDER BY created_at DESC LIMIT ?", (limit,)
            ).fetchall()]

    def list_qa_records(self, limit: int = 50) -> list[dict[str, Any]]:
        with self.connection() as connection:
            rows = connection.execute(
                "SELECT * FROM qa_records ORDER BY created_at DESC LIMIT ?", (limit,)
            ).fetchall()
            results = []
            for row in rows:
                item = dict(row)
                item["refused"] = bool(item["refused"])
                item["retrieval"] = json.loads(item.pop("retrieval_json"))
                item["citations"] = json.loads(item.pop("citations_json"))
                results.append(item)
            return results

    def create_evaluation_run(self, run: dict[str, Any]) -> None:
        with self.connection() as connection:
            connection.execute(
                """INSERT INTO evaluation_runs
                (id, retrieval_method, total, retrieval_hit_rate, answer_accuracy, error_count,
                 errors_json, metrics_json, config_json, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    run["id"], run.get("retrieval_method", "bm25"), run["total"],
                    run["retrieval_hit_rate"], run["answer_accuracy"],
                    len(run["errors"]), json.dumps(run["errors"], ensure_ascii=False),
                    json.dumps(run.get("metrics", {}), ensure_ascii=False),
                    json.dumps(run.get("config", {}), ensure_ascii=False), run["created_at"],
                ),
            )

    def list_evaluation_runs(self, limit: int = 20) -> list[dict[str, Any]]:
        with self.connection() as connection:
            rows = connection.execute(
                "SELECT * FROM evaluation_runs ORDER BY created_at DESC LIMIT ?", (limit,)
            ).fetchall()
            results = []
            for row in rows:
                item = dict(row)
                item["errors"] = json.loads(item.pop("errors_json"))
                item["metrics"] = json.loads(item.pop("metrics_json", "{}"))
                item["config"] = json.loads(item.pop("config_json", "{}"))
                results.append(item)
            return results
