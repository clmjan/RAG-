from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field


DocumentStatus = Literal["pending", "processing", "ready", "failed", "deleted"]
TaskStatus = Literal["queued", "running", "succeeded", "failed"]


class DocumentUploadResponse(BaseModel):
    document_id: str
    task_id: str
    status: TaskStatus


class DocumentResponse(BaseModel):
    id: str
    filename: str
    content_type: str | None
    size_bytes: int
    created_at: datetime
    updated_at: datetime
    status: DocumentStatus
    error_message: str | None = None
    chunk_count: int


class ProcessingTaskResponse(BaseModel):
    id: str
    document_id: str
    status: TaskStatus
    error_message: str | None = None
    created_at: datetime
    updated_at: datetime
    started_at: datetime | None = None
    completed_at: datetime | None = None


class DocumentStatusResponse(BaseModel):
    document_id: str
    status: DocumentStatus
    error_message: str | None = None
    task_id: str | None = None
    task_status: TaskStatus | None = None


class SearchRequest(BaseModel):
    query: str = Field(min_length=1, max_length=500)
    top_k: int | None = Field(default=None, ge=1, le=20)
    retrieval_method: str | None = Field(default=None, pattern="^(bm25|tfidf|embedding|hybrid)$")
    document_id: str | None = None
    filename: str | None = None
    min_score: float | None = Field(default=None, ge=0.0, le=1.0)
    max_candidates: int | None = Field(default=None, ge=1, le=200)


class SearchResultResponse(BaseModel):
    chunk_id: str
    document_id: str
    filename: str
    chunk_index: int
    content: str
    score: float
    matched_terms: list[str] = []
    highlight: str = ""
    title_path: list[str] = []


class SearchResponse(BaseModel):
    query: str
    retrieval_method: str
    results: list[SearchResultResponse]
    status: dict[str, str | bool | None]


class AskRequest(BaseModel):
    question: str = Field(min_length=1, max_length=500)
    top_k: int | None = Field(default=None, ge=1, le=20)
    retrieval_method: str | None = Field(default=None, pattern="^(bm25|tfidf|embedding|hybrid)$")
    document_id: str | None = None
    filename: str | None = None
    min_score: float | None = Field(default=None, ge=0.0, le=1.0)
    max_candidates: int | None = Field(default=None, ge=1, le=200)


class CitationResponse(BaseModel):
    index: int
    chunk_id: str
    document_id: str
    filename: str
    chunk_index: int
    score: float
    quote: str
    title_path: list[str] = []


class AskResponse(BaseModel):
    id: str
    question: str
    answer: str
    refused: bool
    refusal_reason: str | None = None
    refusal: dict[str, str] | None = None
    retrieval_method: str
    generation_method: str
    generation_error: str | None = None
    generation_error_code: str | None = None
    citations: list[CitationResponse]
    retrieved: list[SearchResultResponse]
    status: dict[str, str | bool | None]


class EvaluationError(BaseModel):
    id: str
    question: str
    expected: str
    actual: str
    retrieval_hit: bool
    answer_correct: bool
    category: str
    expected_sources: list[str]
    retrieval_method: str
    selected_threshold: float
    refusal_reason: str | None = None
    retrieved: list[SearchResultResponse]


class EvaluationResponse(BaseModel):
    id: str
    retrieval_method: str
    total: int
    retrieval_hit_rate: float
    answer_accuracy: float
    error_count: int
    errors: list[EvaluationError]
    metrics: dict[str, float | int | str | bool]
    config: dict[str, float | int | str | bool | None]
    created_at: datetime
