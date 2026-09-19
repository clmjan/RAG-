"""Structured application logging and request correlation."""

from contextvars import ContextVar
import json
import logging
import re
import sys
from typing import Any


_request_id: ContextVar[str] = ContextVar("request_id", default="-")
_configured = False


def configure_logging(level: str = "INFO") -> None:
    global _configured
    if _configured:
        return
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(logging.Formatter("%(message)s"))
    logger = logging.getLogger("rag")
    logger.handlers.clear()
    logger.addHandler(handler)
    logger.setLevel(getattr(logging, level.upper(), logging.INFO))
    logger.propagate = False
    _configured = True


def set_request_id(value: str):
    return _request_id.set(value)


def reset_request_id(token) -> None:
    _request_id.reset(token)


def current_request_id() -> str:
    return _request_id.get()


def safe_text(value: Any, max_chars: int = 1000) -> str | None:
    if value is None:
        return None
    text = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]", " ", str(value))
    return text[:max(0, max_chars)]


def log_event(event: str, level: int = logging.INFO, **fields: Any) -> None:
    payload = {"event": event, "request_id": current_request_id(), **fields}
    logging.getLogger("rag").log(level, json.dumps(payload, ensure_ascii=False, default=str))
