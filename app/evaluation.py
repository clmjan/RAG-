import json
import hashlib
from pathlib import Path
import time
import uuid
from dataclasses import asdict
from tempfile import TemporaryDirectory

from .answering import answer_question
from .config import ROOT_DIR, Settings
from .db import Database, utc_now
from .retrieval import EmbeddingRetriever, HybridRetriever, create_retriever
from .chunking import split_text
from .parsing import parse_document


def load_test_cases(path: Path | None = None) -> list[dict]:
    test_path = path or ROOT_DIR / "data" / "evaluation_questions.json"
    payload = json.loads(test_path.read_text(encoding="utf-8"))
    if isinstance(payload, list):
        return payload
    return [case | {"category": category} for category in ("answerable", "unanswerable")
            for case in payload.get(category, [])]


def _percentile(values: list[float], percentile: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, round((len(ordered) - 1) * percentile)))
    return ordered[index]


def _expected_sources(case: dict) -> set[str]:
    source = case.get("source", [])
    return {source} if isinstance(source, str) else set(source)


def _is_source_hit(case: dict, results: list) -> bool:
    expected_sources = _expected_sources(case)
    return bool(expected_sources) and any(result.filename in expected_sources for result in results)


def _retrieval_metrics(cases: list[dict], ranked_results: dict[str, list], k: int) -> tuple[float, float, float]:
    answerable = [case for case in cases if not case.get("should_refuse") and _expected_sources(case)]
    if not answerable:
        return 0.0, 0.0, 0.0
    hits_at_1 = 0
    hits_at_k = 0
    reciprocal_ranks = []
    for case in answerable:
        expected_sources = _expected_sources(case)
        results = ranked_results[case["id"]]
        ranks = [
            position for position, result in enumerate(results[:k], start=1)
            if result.filename in expected_sources
        ]
        hits_at_1 += int(bool(ranks) and ranks[0] == 1)
        hits_at_k += int(bool(ranks))
        reciprocal_ranks.append(1 / ranks[0] if ranks else 0.0)
    return (
        round(hits_at_1 / len(answerable), 4),
        round(hits_at_k / len(answerable), 4),
        round(sum(reciprocal_ranks) / len(answerable), 4),
    )


def _answer_is_correct(case: dict, answer) -> bool:
    expected_refusal = bool(case.get("should_refuse"))
    if answer.refused != expected_refusal:
        return False
    if expected_refusal:
        return True
    return all(term.lower() in answer.text.lower() for term in case.get("answer_terms", []))


def _select_threshold(
    cases: list[dict], ranked_results: dict[str, list], config: Settings
) -> float:
    candidates = [float(item.strip()) for item in config.threshold_candidates.split(",") if item.strip()]
    answerable = [case for case in cases if not case.get("should_refuse")]
    refusal_cases = [case for case in cases if case.get("should_refuse")]
    if not answerable or not refusal_cases:
        return config.retrieval_threshold
    best_threshold = config.retrieval_threshold
    best_score = -1.0
    for threshold in candidates:
        answerable_accuracy = sum(
            _answer_is_correct(case, answer_question(
                case["question"], ranked_results[case["id"]], threshold, config=config
            ))
            for case in answerable
        ) / len(answerable)
        refusal_accuracy = sum(
            _answer_is_correct(case, answer_question(
                case["question"], ranked_results[case["id"]], threshold, config=config
            ))
            for case in refusal_cases
        ) / len(refusal_cases)
        score = (answerable_accuracy + refusal_accuracy) / 2
        if score > best_score or (score == best_score and threshold <= best_threshold):
            best_score = score
            best_threshold = threshold
    return best_threshold


def run_evaluation(
    database: Database,
    test_cases: list[dict] | None = None,
    method: str | None = None,
    config: Settings | None = None,
    persist: bool = True,
) -> dict:
    effective_config = config or Settings()
    selected_method = (method or effective_config.retrieval_method).lower().strip()
    retriever = create_retriever(
        selected_method, database, database.get_chunks(), config=effective_config
    )
    all_cases = test_cases if test_cases is not None else load_test_cases()
    validation_cases = [case for case in all_cases if case.get("split") == "validation"]
    test_cases_only = [case for case in all_cases if case.get("split", "test") == "test"]
    if not test_cases_only:
        test_cases_only = all_cases
    if not validation_cases:
        validation_cases = test_cases_only

    ranked_results: dict[str, list] = {}
    for case in all_cases:
        ranked_results[case["id"]] = retriever.search(case["question"], top_k=5)
    selected_threshold = _select_threshold(validation_cases, ranked_results, effective_config)

    errors = []
    latencies: list[float] = []
    answerable = [case for case in test_cases_only if not case.get("should_refuse")]
    refusal_cases = [case for case in test_cases_only if case.get("should_refuse")]
    answerable_correct = 0
    refusal_correct = 0
    citation_covered = 0
    predicted_refusals = 0
    true_refusals = 0

    for case in test_cases_only:
        started = time.perf_counter()
        results = retriever.search(case["question"], top_k=5)
        answer = answer_question(case["question"], results, selected_threshold, config=effective_config)
        latencies.append((time.perf_counter() - started) * 1000)
        correct = _answer_is_correct(case, answer)
        predicted_refusals += int(answer.refused)
        if case.get("should_refuse"):
            refusal_correct += int(correct)
            true_refusals += int(answer.refused)
        else:
            answerable_correct += int(correct)
            citation_covered += int(bool(answer.citations) and all(
                any(
                    result.chunk_id == citation["chunk_id"] and citation["quote"] in result.content
                    for result in results
                )
                for citation in answer.citations
            ))

        retrieval_hit = _is_source_hit(case, results)
        if (not case.get("should_refuse") and not retrieval_hit) or not correct:
            errors.append({
                "id": case["id"], "question": case["question"],
                "expected": "拒答" if case.get("should_refuse") else "、".join(case.get("answer_terms", [])),
                "actual": answer.text, "retrieval_hit": retrieval_hit, "answer_correct": correct,
                "category": "unanswerable" if case.get("should_refuse") else "answerable",
                "expected_sources": sorted(_expected_sources(case)), "retrieval_method": selected_method,
                "selected_threshold": selected_threshold, "refusal_reason": answer.refusal_reason,
                "retrieved": [result.as_dict() for result in results],
            })

    recall_at_1, recall_at_5, mrr = _retrieval_metrics(test_cases_only, ranked_results, 5)
    total = len(test_cases_only)
    refusal_total = len(refusal_cases)
    provider_configured = None
    if isinstance(retriever, EmbeddingRetriever):
        provider_configured = retriever.provider_configured
    elif isinstance(retriever, HybridRetriever):
        provider_configured = retriever.embedding.provider_configured
    metrics = {
        "recall_at_1": recall_at_1, "recall_at_5": recall_at_5, "mrr": mrr,
        "answerable_accuracy": round(answerable_correct / len(answerable), 4) if answerable else 0.0,
        "refusal_accuracy": round(refusal_correct / refusal_total, 4) if refusal_total else 0.0,
        "refusal_precision": round(true_refusals / predicted_refusals, 4) if predicted_refusals else 0.0,
        "refusal_recall": round(true_refusals / refusal_total, 4) if refusal_total else 0.0,
        "citation_coverage": round(citation_covered / len(answerable), 4) if answerable else 0.0,
        "average_latency_ms": round(sum(latencies) / len(latencies), 2) if latencies else 0.0,
        "p95_latency_ms": round(_percentile(latencies, 0.95), 2), "selected_threshold": selected_threshold,
        "retrieval_method": selected_method, "chunk_size": effective_config.chunk_size,
        "chunk_overlap": effective_config.chunk_overlap, "evaluation_split": "test",
        "validation_count": len(validation_cases), "answerable_count": len(answerable),
        "refusal_count": len(refusal_cases),
        "answer_accuracy": round((answerable_correct + refusal_correct) / total, 4) if total else 0.0,
        "Recall@1": recall_at_1, "Recall@5": recall_at_5, "MRR": mrr,
    }
    if provider_configured is not None:
        metrics["embedding_provider_configured"] = provider_configured
    run = {
        "id": str(uuid.uuid4()), "retrieval_method": selected_method, "total": total,
        "retrieval_hit_rate": recall_at_5,
        "answer_accuracy": round((answerable_correct + refusal_correct) / total, 4) if total else 0.0,
        "error_count": len(errors), "errors": errors, "metrics": metrics,
        "config": evaluation_config_snapshot(
            effective_config, selected_threshold, selected_method, all_cases
        ),
        "created_at": utc_now(),
    }
    if persist:
        database.create_evaluation_run(run)
    return run


def evaluation_config_snapshot(
    config: Settings, selected_threshold: float | None = None,
    method: str | None = None, test_cases: list[dict] | None = None,
) -> dict:
    """Return only reproducibility settings, never provider secrets."""
    cases = test_cases if test_cases is not None else load_test_cases()
    serialized_cases = json.dumps(cases, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return {
        "chunk_size": config.chunk_size,
        "chunk_overlap": config.chunk_overlap,
        "retrieval_threshold": config.retrieval_threshold,
        "selected_threshold": selected_threshold,
        "threshold_candidates": config.threshold_candidates,
        "retrieval_method": method or config.retrieval_method,
        "evidence_min_score_gap": config.evidence_min_score_gap,
        "hybrid_lexical_weight": config.hybrid_lexical_weight,
        "hybrid_semantic_weight": config.hybrid_semantic_weight,
        "retrieval_top_k": config.retrieval_top_k,
        "retrieval_min_score": config.retrieval_min_score,
        "retrieval_max_candidates": config.retrieval_max_candidates,
        "evaluation_dataset": str(ROOT_DIR / "data" / "evaluation_questions.json"),
        "evaluation_case_count": len(cases),
        "evaluation_dataset_sha256": hashlib.sha256(serialized_cases.encode("utf-8")).hexdigest(),
        "embedding_model": config.embedding_model or None,
    }


def run_ablation(
    database: Database,
    method: str,
    config: Settings | None = None,
    test_cases: list[dict] | None = None,
) -> list[dict]:
    """Run real evaluations for configured chunk, threshold and hybrid variants."""
    base = config or Settings()
    variants = [{"name": "baseline", "config": base, "requires_rechunk": False}]
    for size in sorted({max(100, base.chunk_size // 2), base.chunk_size, base.chunk_size * 2}):
        variants.append({"name": f"chunk_size_{size}", "config": Settings(**{
            **asdict(base), "chunk_size": size,
        }), "requires_rechunk": True})
    for overlap in sorted({0, min(base.chunk_overlap, max(0, base.chunk_size // 2)), base.chunk_overlap}):
        variants.append({"name": f"chunk_overlap_{overlap}", "config": Settings(**{
            **asdict(base), "chunk_overlap": overlap,
        }), "requires_rechunk": True})
    for threshold in sorted({base.retrieval_threshold, *[float(item.strip()) for item in base.threshold_candidates.split(",") if item.strip()]}):
        variants.append({"name": f"threshold_{threshold:g}", "config": Settings(**{
            **asdict(base), "retrieval_threshold": threshold,
            "threshold_candidates": str(threshold),
        }), "requires_rechunk": False})
    if method == "hybrid":
        for lexical, semantic in ((0.2, 0.8), (0.5, 0.5), (0.8, 0.2)):
            variants.append({"name": f"hybrid_{lexical:g}_{semantic:g}", "config": Settings(**{
                **asdict(base), "hybrid_lexical_weight": lexical,
                "hybrid_semantic_weight": semantic,
            }), "requires_rechunk": False})
    results = []
    for variant in variants:
        variant_database = database
        rebuilt = False
        with TemporaryDirectory(prefix="rag-eval-") as temporary_dir:
            if variant["requires_rechunk"]:
                candidate = Database(Path(temporary_dir) / "variant.db")
                candidate.init()
                rebuilt = _rebuild_evaluation_index(database, candidate, variant["config"])
                if rebuilt:
                    variant_database = candidate
            run = run_evaluation(
                variant_database, test_cases, method=method, config=variant["config"], persist=False
            )
            run["config"]["chunking_rebuilt"] = rebuilt
            run["ablation"] = variant["name"]
            database.create_evaluation_run(run)
        results.append(run)
    return results


def _rebuild_evaluation_index(source: Database, target: Database, config: Settings) -> bool:
    documents = [document for document in source.list_documents() if document.get("status") != "deleted"]
    if not documents:
        return False
    rebuilt = []
    for document in documents:
        path = config.upload_dir / f"{document['id']}{Path(document['filename']).suffix.lower()}"
        bundled_path = ROOT_DIR / "data" / document["filename"]
        try:
            if path.exists():
                text = parse_document(document["filename"], path.read_bytes())
            elif bundled_path.exists():
                text = parse_document(document["filename"], bundled_path.read_bytes())
            else:
                # Fall back to the persisted source chunks. This still evaluates a real
                # alternate chunking configuration without inventing document content.
                text = "\n\n".join(
                    chunk["content"] for chunk in source.get_document_chunks(document["id"])
                )
                if not text.strip():
                    return False
            chunks = split_text(text, config.chunk_size, config.chunk_overlap,
                                document_id=document["id"])
            rebuilt.append((document, chunks))
        except Exception:
            return False
    for document, chunks in rebuilt:
        target.create_document(document, chunks)
    return True
