import re
from dataclasses import dataclass

from .config import settings
from .retrieval import SearchResult, query_terms


REFUSAL = "我在当前课程资料中没有找到足够依据来回答这个问题。"


@dataclass
class Answer:
    text: str
    refused: bool
    citations: list[dict]
    refusal_reason: str | None = None
    refusal: dict | None = None


def _query_terms(question: str) -> set[str]:
    return set(query_terms(question))


def _term_hits(question: str, content: str) -> int:
    lowered = content.lower()
    return sum(1 for term in _query_terms(question) if term in lowered)


def _best_sentences(question: str, result: SearchResult, limit: int = 2) -> list[str]:
    terms = _query_terms(question)
    sentences = [sentence.strip() for sentence in re.split(r"(?<=[。！？.!?])\s*|\n+", result.content) if sentence.strip()]
    scored = []
    for position, sentence in enumerate(sentences):
        overlap = sum(1 for term in terms if term in sentence.lower())
        scored.append((overlap, -position, sentence))
    selected = [item[2] for item in sorted(scored, reverse=True)[:limit] if item[0] > 0]
    if not selected:
        return []
    return selected


def _refused(reason: str, detail: str | None = None) -> Answer:
    return Answer(
        text=REFUSAL, refused=True, citations=[], refusal_reason=reason,
        refusal={"code": reason, "message": detail or reason},
    )


def answer_question(question: str, results: list[SearchResult], threshold: float, config=None) -> Answer:
    effective_config = config or settings
    if not results:
        return _refused("no_results", "没有检索到相关证据")
    if results[0].score < threshold:
        return _refused("below_threshold", "最高检索分数低于证据阈值")
    if len(results) > 1 and results[0].score - results[1].score < effective_config.evidence_min_score_gap:
        return _refused("insufficient_score_gap", "第一名与第二名证据分差不足")

    terms = _query_terms(question)
    usable = [result for result in results if result.score >= threshold]
    usable = [result for result in usable if terms and _term_hits(question, result.content) / len(terms) >= 0.15]
    if not usable:
        return _refused("insufficient_term_overlap", "查询词与证据重合度不足")

    parts = []
    citations = []
    for citation_index, result in enumerate(usable[:effective_config.extractive_max_citations], start=1):
        sentences = _best_sentences(question, result)
        if sentences:
            quote = sentences[0]
            parts.append(f"{''.join(sentences)} [{citation_index}]")
            citations.append({
                "index": citation_index, "chunk_id": result.chunk_id, "document_id": result.document_id,
                "filename": result.filename, "chunk_index": result.chunk_index,
                "score": round(result.score, 4), "quote": quote,
                "title_path": result.title_path or [],
            })
    if not parts:
        return _refused("no_citable_sentence", "检索证据中没有可引用句子")
    answer_text = "根据课程资料：" + " ".join(parts)
    return Answer(
        text=answer_text[:effective_config.extractive_max_answer_chars],
        refused=False, citations=citations,
    )
