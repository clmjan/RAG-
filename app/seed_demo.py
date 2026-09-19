"""Load the five demo documents into the local SQLite database."""

import hashlib
from pathlib import Path
import uuid

from .chunking import split_text
from .config import ROOT_DIR, settings
from .db import Database, utc_now
from .parsing import parse_document


def seed_demo_documents(database: Database | None = None) -> int:
    database = database or Database(settings.database_path)
    database.init()
    count = 0
    for path in sorted((ROOT_DIR / "data").glob("*_note.md")):
        content = path.read_bytes()
        digest = hashlib.sha256(content).hexdigest()
        if database.find_document_by_hash(digest):
            continue
        document_id = str(uuid.uuid4())
        text = parse_document(path.name, content)
        chunks = [dict(chunk, document_id=document_id) for chunk in split_text(
            text, settings.chunk_size, settings.chunk_overlap
        )]
        database.create_document({
            "id": document_id,
            "filename": path.name,
            "content_type": "text/markdown",
            "size_bytes": len(content),
            "sha256": digest,
            "created_at": utc_now(),
        }, chunks)
        count += 1
    return count


if __name__ == "__main__":
    print(f"seeded {seed_demo_documents()} document(s)")
