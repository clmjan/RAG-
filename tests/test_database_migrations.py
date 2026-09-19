import sqlite3

from app.db import Database


def test_existing_documents_are_migrated_as_ready(tmp_path):
    path = tmp_path / "legacy.db"
    with sqlite3.connect(path) as connection:
        connection.execute(
            """CREATE TABLE documents (
                id TEXT PRIMARY KEY,
                filename TEXT NOT NULL,
                content_type TEXT,
                size_bytes INTEGER NOT NULL,
                sha256 TEXT NOT NULL UNIQUE,
                created_at TEXT NOT NULL,
                chunk_count INTEGER NOT NULL DEFAULT 0
            )"""
        )
        connection.execute(
            """INSERT INTO documents
            (id, filename, content_type, size_bytes, sha256, created_at, chunk_count)
            VALUES ('doc-1', 'legacy.md', 'text/markdown', 1, 'hash',
                    '2026-01-01T00:00:00+00:00', 0)"""
        )

    database = Database(path)
    database.init()
    document = database.get_document("doc-1")

    assert document["status"] == "ready"
    assert document["updated_at"] == document["created_at"]
    assert database.list_processing_tasks() == []


def test_processing_task_states_migrate_and_running_work_is_requeued(tmp_path):
    path = tmp_path / "legacy_tasks.db"
    with sqlite3.connect(path) as connection:
        connection.executescript(
            """CREATE TABLE documents (
                id TEXT PRIMARY KEY, filename TEXT NOT NULL, content_type TEXT,
                size_bytes INTEGER NOT NULL, sha256 TEXT NOT NULL UNIQUE,
                created_at TEXT NOT NULL, chunk_count INTEGER NOT NULL DEFAULT 0
            );
            CREATE TABLE processing_tasks (
                id TEXT PRIMARY KEY, document_id TEXT NOT NULL, status TEXT NOT NULL,
                error_message TEXT, created_at TEXT NOT NULL, updated_at TEXT NOT NULL,
                started_at TEXT, completed_at TEXT,
                CHECK(status IN ('pending', 'processing', 'ready', 'failed', 'deleted'))
            );
            INSERT INTO documents VALUES
                ('doc-1', 'lesson.md', 'text/markdown', 1, 'hash-1',
                 '2026-01-01T00:00:00+00:00', 0);
            INSERT INTO processing_tasks VALUES
                ('task-1', 'doc-1', 'processing', NULL,
                 '2026-01-01T00:00:00+00:00', '2026-01-01T00:00:00+00:00',
                 '2026-01-01T00:00:00+00:00', NULL);"""
        )

    database = Database(path)
    database.init()
    task = database.get_processing_task("task-1")
    assert task["status"] == "running"

    queued = database.recover_processing_tasks()
    assert [item["id"] for item in queued] == ["task-1"]
    assert database.get_processing_task("task-1")["status"] == "queued"
    assert database.get_document("doc-1")["status"] == "pending"
