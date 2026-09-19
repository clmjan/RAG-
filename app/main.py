import hashlib
import asyncio
import logging
import os
from pathlib import Path
import sqlite3
import time
import uuid
from contextlib import asynccontextmanager

from fastapi import BackgroundTasks, FastAPI, File, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from .answering import REFUSAL, answer_question
from .config import settings
from .db import Database, utc_now
from .evaluation import run_ablation, run_evaluation
from .generation import create_generator, validate_generated_response
from .providers import ProviderError
from .parsing import SUPPORTED_EXTENSIONS
from .processing import process_document_task
from .observability import (
    configure_logging, current_request_id, log_event, reset_request_id, safe_text,
    set_request_id,
)
from .schemas import (
    AskRequest, AskResponse, DocumentResponse, DocumentStatusResponse, DocumentUploadResponse, EvaluationResponse,
    ProcessingTaskResponse, SearchRequest, SearchResponse,
)
from .retrieval import create_retriever


database = Database(settings.database_path)


@asynccontextmanager
async def lifespan(_: FastAPI):
    configure_logging(settings.log_level)
    database.init()
    settings.upload_dir.mkdir(parents=True, exist_ok=True)
    recovery_jobs = []
    for task in database.recover_processing_tasks():
        document = database.get_document(task["document_id"])
        if not document:
            database.fail_processing_task(task["id"], "恢复任务时找不到文档记录")
            continue
        source_path = settings.upload_dir / (
            f"{document['id']}{Path(document['filename']).suffix.lower()}"
        )
        recovery_request_id = f"recovery-{uuid.uuid4()}"
        recovery_jobs.append(asyncio.create_task(asyncio.to_thread(
            process_document_task, database, settings, document["id"], task["id"], source_path,
            recovery_request_id,
        )))
    try:
        yield
    finally:
        if recovery_jobs:
            await asyncio.gather(*recovery_jobs, return_exceptions=True)


app = FastAPI(title="课程资料知识库与学习助理", version="0.1.0", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.middleware("http")
async def request_observability(request: Request, call_next):
    request_id = str(uuid.uuid4())
    token = set_request_id(request_id)
    request.state.request_id = request_id
    started = time.perf_counter()
    status_code = 500
    error_message = None
    try:
        response = await call_next(request)
        status_code = response.status_code
        response.headers["X-Request-ID"] = request_id
        return response
    except Exception as exc:
        error_message = type(exc).__name__
        raise
    finally:
        duration_ms = round((time.perf_counter() - started) * 1000, 3)
        record = {
            "request_id": request_id, "method": request.method,
            "path": safe_text(request.url.path, 500) or "", "status_code": status_code,
            "duration_ms": duration_ms, "error_message": error_message,
            "created_at": utc_now(),
        }
        try:
            database.create_request_log(record)
        except Exception as log_error:
            log_event("request_log_write_failed", logging.ERROR, error=type(log_error).__name__)
        log_event("http_request", **record)
        reset_request_id(token)


def _retriever(method: str | None = None):
    selected = method or settings.retrieval_method
    try:
        return create_retriever(selected, database, database.get_chunks(), config=settings)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


def _record_search(
    query: str, requested_method: str, outcome, top_k: int, duration_ms: float
) -> None:
    record = {
        "id": str(uuid.uuid4()), "request_id": current_request_id(),
        "query": safe_text(query, settings.log_text_max_chars) or "",
        "retrieval_method": requested_method,
        "effective_method": outcome.status.effective_method, "top_k": top_k,
        "result_count": len(outcome.results), "duration_ms": round(duration_ms, 3),
        "fallback_reason": outcome.status.fallback_reason, "created_at": utc_now(),
    }
    try:
        database.create_search_log(record)
    except Exception as exc:
        log_event("search_log_write_failed", logging.ERROR, error=type(exc).__name__)
    log_event("search", **record)


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/ready")
def ready() -> dict[str, str]:
    try:
        database.init()
        database.get_chunks()
    except Exception as exc:
        raise HTTPException(status_code=503, detail=f"数据库未就绪: {exc}") from exc
    return {"status": "ready"}


@app.post("/documents/upload", response_model=DocumentUploadResponse, status_code=202)
async def upload_document(
    background_tasks: BackgroundTasks, file: UploadFile = File(...)
) -> dict:
    filename = file.filename or "unnamed"
    suffix = Path(filename).suffix.lower()
    if suffix not in SUPPORTED_EXTENSIONS:
        raise HTTPException(status_code=400, detail="仅支持 PDF、Markdown 和 TXT 文件")
    content = await file.read(settings.max_upload_bytes + 1)
    if len(content) > settings.max_upload_bytes:
        raise HTTPException(status_code=413, detail="文件大小超过限制")
    digest = hashlib.sha256(content).hexdigest()
    existing = database.find_document_by_hash(digest)
    if existing:
        raise HTTPException(status_code=409, detail="相同文件已经上传")

    document_id = str(uuid.uuid4())
    task_id = str(uuid.uuid4())
    created_at = utc_now()
    document = {
        "id": document_id, "filename": filename, "content_type": file.content_type,
        "size_bytes": len(content), "sha256": digest, "created_at": created_at,
    }
    task = {"id": task_id, "document_id": document_id, "created_at": created_at}
    settings.upload_dir.mkdir(parents=True, exist_ok=True)
    final_path = settings.upload_dir / f"{document_id}{suffix}"
    temporary_path = settings.upload_dir / f".{document_id}.uploading"
    try:
        temporary_path.write_bytes(content)
        os.replace(temporary_path, final_path)
        database.create_document_task(document, task)
    except sqlite3.IntegrityError as exc:
        temporary_path.unlink(missing_ok=True)
        final_path.unlink(missing_ok=True)
        raise HTTPException(status_code=409, detail="相同文件已经上传") from exc
    except Exception as exc:
        temporary_path.unlink(missing_ok=True)
        final_path.unlink(missing_ok=True)
        raise HTTPException(status_code=500, detail=f"保存文档失败: {exc}") from exc
    background_tasks.add_task(
        process_document_task, database, settings, document_id, task_id, final_path,
        current_request_id(),
    )
    return {"document_id": document_id, "task_id": task_id, "status": "queued"}


@app.get("/documents", response_model=list[DocumentResponse])
def list_documents() -> list[dict]:
    return database.list_documents()


@app.get("/documents/{document_id}", response_model=DocumentResponse)
def get_document(document_id: str) -> dict:
    document = database.get_document(document_id)
    if not document:
        raise HTTPException(status_code=404, detail="文档不存在")
    return document


@app.get("/documents/{document_id}/status", response_model=DocumentStatusResponse)
def get_document_status(document_id: str) -> dict:
    document = database.get_document(document_id)
    if not document:
        raise HTTPException(status_code=404, detail="文档不存在")
    tasks = database.list_processing_tasks(limit=1, document_id=document_id)
    task = tasks[0] if tasks else None
    return {
        "document_id": document_id,
        "status": document["status"],
        "error_message": document.get("error_message"),
        "task_id": task["id"] if task else None,
        "task_status": task["status"] if task else None,
    }


@app.get("/processing-tasks/{task_id}", response_model=ProcessingTaskResponse)
@app.get("/tasks/{task_id}", response_model=ProcessingTaskResponse)
def get_processing_task(task_id: str) -> dict:
    task = database.get_processing_task(task_id)
    if not task:
        raise HTTPException(status_code=404, detail="处理任务不存在")
    return task


@app.get("/processing-tasks", response_model=list[ProcessingTaskResponse])
def list_processing_tasks(
    limit: int = 50, document_id: str | None = None
) -> list[dict]:
    if limit < 1 or limit > 200:
        raise HTTPException(status_code=400, detail="limit 必须在 1 到 200 之间")
    return database.list_processing_tasks(limit, document_id)


@app.get("/documents/{document_id}/chunks")
def document_chunks(document_id: str) -> list[dict]:
    if not database.get_document(document_id):
        raise HTTPException(status_code=404, detail="文档不存在")
    return database.get_document_chunks(document_id)


@app.delete("/documents/{document_id}")
def delete_document(document_id: str) -> dict[str, str]:
    document = database.get_document(document_id)
    if not document or document["status"] == "deleted":
        raise HTTPException(status_code=404, detail="文档不存在")
    document_path = settings.upload_dir / f"{document_id}{Path(document['filename']).suffix.lower()}"
    try:
        database.delete_document(document_id)
        document_path.unlink(missing_ok=True)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"删除文档失败: {exc}") from exc
    return {"id": document_id, "status": "deleted"}


@app.post("/search", response_model=SearchResponse)
def search(request: SearchRequest) -> dict:
    method = request.retrieval_method or settings.retrieval_method
    top_k = request.top_k or settings.retrieval_top_k
    min_score = settings.retrieval_min_score if request.min_score is None else request.min_score
    max_candidates = request.max_candidates or settings.retrieval_max_candidates
    started = time.perf_counter()
    outcome = _retriever(method).search_with_status(
        request.query, top_k, min_score=min_score,
        max_candidates=max_candidates, document_id=request.document_id,
        filename=request.filename,
    )
    _record_search(request.query, method, outcome, top_k, (time.perf_counter() - started) * 1000)
    return {"query": request.query, "retrieval_method": method,
            "results": [result.as_dict() for result in outcome.results],
            "status": outcome.status.as_dict()}


@app.post("/questions", response_model=AskResponse)
def ask(request: AskRequest) -> dict:
    total_started = time.perf_counter()
    top_k = request.top_k or settings.retrieval_top_k
    min_score = settings.retrieval_min_score if request.min_score is None else request.min_score
    max_candidates = request.max_candidates or settings.retrieval_max_candidates
    requested_method = request.retrieval_method or settings.retrieval_method
    retrieval_started = time.perf_counter()
    outcome = _retriever(request.retrieval_method).search_with_status(
        request.question, top_k, min_score=min_score,
        max_candidates=max_candidates, document_id=request.document_id,
        filename=request.filename,
    )
    _record_search(
        request.question, requested_method, outcome, top_k,
        (time.perf_counter() - retrieval_started) * 1000,
    )
    results = outcome.results
    answer = answer_question(request.question, results, settings.retrieval_threshold, config=settings)
    generation_method = "extractive"
    generation_error = None
    generation_error_code = None
    if not answer.refused and settings.generation_method == "llm":
        generator = create_generator("llm", config=settings)
        if generator is None:
            generation_error = "LLM provider is not configured; used extractive answering"
            generation_error_code = "provider_not_configured"
        else:
            try:
                generated = generator.generate(request.question, results)
                citations = validate_generated_response(
                    generated, results, max_citations=settings.llm_max_citations,
                    max_answer_chars=settings.llm_max_answer_chars,
                )
                if generated.refused:
                    answer.refused = True
                    answer.citations = []
                    answer.text = REFUSAL
                    answer.refusal_reason = "llm_refused"
                    answer.refusal = {"code": "llm_refused", "message": "LLM 判断上下文不足"}
                else:
                    answer.text = generated.answer
                    answer.citations = citations
                generation_method = "llm"
            except ProviderError as exc:
                generation_error = str(exc)
                generation_error_code = exc.reason
    response_retrieval = [result.as_dict() for result in results]
    retrieval_log = [
        {
            "chunk_id": result.chunk_id, "document_id": result.document_id,
            "filename": safe_text(result.filename, 255), "chunk_index": result.chunk_index,
            "score": round(result.score, 4),
        }
        for result in results
    ]
    citation_log = [
        citation | {"quote": safe_text(citation.get("quote"), settings.log_text_max_chars) or ""}
        for citation in answer.citations
    ]
    duration_ms = round((time.perf_counter() - total_started) * 1000, 3)
    record = {
        "id": str(uuid.uuid4()),
        "request_id": current_request_id(),
        "question": safe_text(request.question, settings.log_text_max_chars) or "",
        "answer": safe_text(answer.text, settings.log_text_max_chars) or "",
        "refused": answer.refused, "refusal_reason": answer.refusal_reason,
        "retrieval": retrieval_log, "citations": citation_log,
        "retrieval_method": requested_method,
        "effective_method": outcome.status.effective_method,
        "generation_method": generation_method,
        "generation_error": safe_text(generation_error, settings.log_text_max_chars),
        "duration_ms": duration_ms, "created_at": utc_now(),
    }
    database.create_qa_record(record)
    log_event(
        "question_answer", request_id=record["request_id"], question=record["question"],
        retrieval=record["retrieval"], answer=record["answer"], refused=answer.refused,
        refusal_reason=answer.refusal_reason, citations=record["citations"],
        generation_method=generation_method, generation_error=record["generation_error"],
        duration_ms=duration_ms,
    )
    return {
        "id": record["id"], "question": request.question, "answer": answer.text,
        "refused": record["refused"], "refusal_reason": answer.refusal_reason,
        "retrieval_method": request.retrieval_method or settings.retrieval_method,
        "generation_method": generation_method, "generation_error": generation_error,
        "generation_error_code": generation_error_code,
        "refusal": answer.refusal,
        "citations": answer.citations,
        "retrieved": response_retrieval,
        "status": outcome.status.as_dict(),
    }


@app.get("/questions")
def question_history(limit: int = 50) -> list[dict]:
    if limit < 1 or limit > 200:
        raise HTTPException(status_code=400, detail="limit 必须在 1 到 200 之间")
    return database.list_qa_records(limit)


@app.post("/evaluations/run", response_model=EvaluationResponse)
def evaluate() -> dict:
    return run_evaluation(database, method=settings.retrieval_method, config=settings)


@app.post("/evaluations/compare")
def compare_evaluations() -> dict:
    return {
        "runs": [
            run_evaluation(database, method="tfidf", config=settings),
            run_evaluation(database, method="bm25", config=settings),
            run_evaluation(database, method="embedding", config=settings),
            run_evaluation(database, method="hybrid", config=settings),
        ]
    }


@app.post("/evaluations/ablation")
def ablation_evaluations(method: str | None = None) -> dict:
    selected = (method or settings.retrieval_method).lower().strip()
    if selected not in {"tfidf", "bm25", "embedding", "hybrid"}:
        raise HTTPException(status_code=400, detail="不支持的评测检索方式")
    return {"method": selected, "runs": run_ablation(database, selected, config=settings)}


@app.get("/evaluations")
def evaluation_history(limit: int = 20) -> list[dict]:
    if limit < 1 or limit > 100:
        raise HTTPException(status_code=400, detail="limit 必须在 1 到 100 之间")
    return database.list_evaluation_runs(limit)


# The production image includes the Vite build. Keeping this conditional makes
# the API-only development and test environment work without frontend assets.
_frontend_dist = Path(__file__).resolve().parent.parent / "frontend" / "dist"
if _frontend_dist.exists():
    app.mount("/", StaticFiles(directory=_frontend_dist, html=True), name="frontend")
