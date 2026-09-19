from dataclasses import dataclass
import json
import re

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from .config import Settings
from .providers import OpenAICompatibleEmbeddingClient, ProviderError, create_embedding_client


_TFIDF_CACHE: dict[tuple, "TfidfRetriever"] = {}


@dataclass
class SearchResult:
    chunk_id: str
    document_id: str
    filename: str
    chunk_index: int
    content: str
    score: float
    matched_terms: list[str] | None = None
    highlight: str = ""
    title_path: list[str] | None = None

    def as_dict(self) -> dict[str, str | int | float | list[str]]:
        return {
            "chunk_id": self.chunk_id, "document_id": self.document_id,
            "filename": self.filename, "chunk_index": self.chunk_index,
            "content": self.content, "score": round(self.score, 4),
            "matched_terms": self.matched_terms or [],
            "highlight": self.highlight or self.content,
            "title_path": self.title_path or [],
        }


@dataclass
class RetrievalStatus:
    requested_method: str
    effective_method: str
    fallback_reason: str | None
    embedding_used: bool

    def as_dict(self) -> dict[str, str | bool | None]:
        return {
            "requested_method": self.requested_method,
            "effective_method": self.effective_method,
            "fallback_reason": self.fallback_reason,
            "embedding_used": self.embedding_used,
        }


@dataclass
class RetrievalOutcome:
    results: list[SearchResult]
    status: RetrievalStatus


ENGLISH_STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "does", "for", "from", "how",
    "in", "is", "it", "of", "on", "or", "that", "the", "this", "to", "what",
    "when", "where", "which", "who", "why", "with",
}
CHINESE_STOP_CHARS = set("的是了吗呢在有和与及为能会要从把被这于中")
CHINESE_QUERY_STOP_CHARS = CHINESE_STOP_CHARS | set("请如何什么哪些怎么一个这份")


def _normalize_english(token: str) -> str:
    if token.endswith("ies") and len(token) > 4:
        return token[:-3] + "y"
    if token.endswith("ses") and len(token) > 4:
        return token[:-2]
    if token.endswith("s") and len(token) > 3:
        return token[:-1]
    return token


def query_terms(text: str) -> list[str]:
    terms: list[str] = []
    for match in re.finditer(r"[\u4e00-\u9fff]+|[a-zA-Z0-9]+", text.lower()):
        value = match.group(0)
        if re.fullmatch(r"[a-z0-9]+", value):
            if value not in ENGLISH_STOPWORDS:
                terms.append(_normalize_english(value))
            continue
        characters = [character for character in value if character not in CHINESE_QUERY_STOP_CHARS]
        terms.extend("".join(characters[index:index + size])
                     for size in (2, 3) for index in range(len(characters) - size + 1))
    return list(dict.fromkeys(term for term in terms if len(term) > 1 or term.isascii()))


def _tokens(text: str) -> list[str]:
    tokens: list[str] = []
    for match in re.finditer(r"[\u4e00-\u9fff]+|[a-zA-Z0-9]+", text.lower()):
        value = match.group(0)
        if re.fullmatch(r"[a-z0-9]+", value):
            if value not in ENGLISH_STOPWORDS:
                tokens.append(_normalize_english(value))
        else:
            tokens.extend(character for character in value if character not in CHINESE_STOP_CHARS)
    return tokens


def _fts_tokens(text: str) -> list[str]:
    terms: list[str] = []
    for match in re.finditer(r"[\u4e00-\u9fff]+|[a-zA-Z0-9]+", text.lower()):
        value = match.group(0)
        if re.fullmatch(r"[a-z0-9]+", value):
            if value not in ENGLISH_STOPWORDS:
                terms.append(_normalize_english(value))
            continue
        characters = [character for character in value if character not in CHINESE_STOP_CHARS]
        terms.extend("".join(characters[index:index + size])
                     for size in (2, 3) for index in range(len(characters) - size + 1))
    return list(dict.fromkeys(terms))


def build_index_text(text: str) -> str:
    return " ".join(_fts_tokens(text))


def build_fts_query(query: str) -> str:
    return " OR ".join(f'"{term.replace(chr(34), "")}"' for term in query_terms(query))


def _bounded(value: int, maximum: int = 200) -> int:
    return max(1, min(int(value), maximum))


def _matches(chunk: dict, document_id: str | None, filename: str | None) -> bool:
    return (not document_id or chunk.get("document_id") == document_id) and (
        not filename or chunk.get("filename") == filename
    )


def _highlight(content: str, terms: list[str]) -> str:
    terms = sorted(set(terms), key=len, reverse=True)
    if not terms:
        return content[:240]
    pattern = re.compile("|".join(re.escape(term) for term in terms), re.IGNORECASE)
    match = pattern.search(content)
    excerpt = content[max(0, (match.start() if match else 0) - 100):]
    excerpt = excerpt[:240]
    return pattern.sub(lambda item: f"<mark>{item.group(0)}</mark>", excerpt)


def _decorate(results: list[SearchResult], query: str) -> list[SearchResult]:
    terms = query_terms(query)
    for result in results:
        result.matched_terms = [term for term in terms if term.lower() in result.content.lower()]
        result.highlight = _highlight(result.content, result.matched_terms)
    return results


def _finish(results: list[SearchResult], query: str, top_k: int, min_score: float) -> list[SearchResult]:
    return _decorate([result for result in results if result.score >= min_score][: _bounded(top_k)], query)


class TfidfRetriever:
    method = "tfidf"

    def __init__(self, chunks: list[dict], cache_key=None):
        self.chunks = chunks
        self.cache_key = cache_key
        self.vectorizer = None
        self.matrix = None
        if chunks:
            self.vectorizer = TfidfVectorizer(analyzer=_tokens, lowercase=False, min_df=1)
            self.matrix = self.vectorizer.fit_transform([chunk["content"] for chunk in chunks])

    def search(self, query: str, top_k: int = 5, **kwargs) -> list[SearchResult]:
        return self.search_with_status(query, top_k, **kwargs).results

    def search_with_status(self, query: str, top_k: int = 5, *, min_score: float = 0.0,
                           max_candidates: int = 50, document_id: str | None = None,
                           filename: str | None = None) -> RetrievalOutcome:
        if not self.chunks or not query.strip() or self.vectorizer is None:
            results = []
        else:
            selected = [(index, chunk) for index, chunk in enumerate(self.chunks)
                        if _matches(chunk, document_id, filename)]
            if not selected:
                results = []
            else:
                indices = [item[0] for item in selected]
                scores = cosine_similarity(self.vectorizer.transform([query]), self.matrix[indices]).ravel()
                ranked = sorted(zip(selected, scores), key=lambda item: item[1], reverse=True)
                results = [SearchResult(
                    chunk_id=chunk["id"], document_id=chunk["document_id"], filename=chunk["filename"],
                    chunk_index=chunk["chunk_index"], content=chunk["content"], score=float(score),
                    title_path=chunk.get("title_path", []),
                ) for (_, chunk), score in ranked[:_bounded(max_candidates)]]
        return RetrievalOutcome(_finish(results, query, top_k, min_score),
                                RetrievalStatus("tfidf", "tfidf", None, False))


class Bm25Retriever:
    method = "bm25"

    def __init__(self, database):
        self.database = database

    def search(self, query: str, top_k: int = 5, **kwargs) -> list[SearchResult]:
        return self.search_with_status(query, top_k, **kwargs).results

    def search_with_status(self, query: str, top_k: int = 5, *, min_score: float = 0.0,
                           max_candidates: int = 50, document_id: str | None = None,
                           filename: str | None = None) -> RetrievalOutcome:
        rows = self.database.search_bm25(query, _bounded(max_candidates), document_id, filename)
        raw_scores = [max(0.0, -float(row["bm25_rank"])) for row in rows]
        max_raw = max(raw_scores, default=0.0)
        query_term_set = set(query_terms(query))
        results = []
        for row, raw_score in zip(rows, raw_scores):
            content_terms = set(_fts_tokens(row["content"]))
            overlap = len(query_term_set & content_terms) / len(query_term_set) if query_term_set else 0.0
            score = 0.8 * overlap + 0.2 * (raw_score / max_raw if max_raw else 0.0)
            results.append(SearchResult(
                chunk_id=row["id"], document_id=row["document_id"], filename=row["filename"],
                chunk_index=row["chunk_index"], content=row["content"], score=float(score),
                title_path=row.get("title_path", []),
            ))
        return RetrievalOutcome(_finish(results, query, top_k, min_score),
                                RetrievalStatus("bm25", "bm25", None, False))


class EmbeddingRetriever:
    method = "embedding"

    def __init__(self, database, client: OpenAICompatibleEmbeddingClient | None = None,
                 config: Settings | None = None):
        self.database = database
        self.client = client or create_embedding_client(config or Settings())
        self.fallback = Bm25Retriever(database)

    @property
    def provider_configured(self) -> bool:
        return self.client is not None

    def _load_vectors(self, chunks: list[dict]) -> dict[str, list[float]]:
        if self.client is None:
            return {}
        vectors = {row["chunk_id"]: [float(value) for value in json.loads(row["vector_json"])]
                   for row in self.database.get_embeddings(self.client.model)}
        if len({len(vector) for vector in vectors.values()}) > 1:
            self.database.delete_embeddings(self.client.model)
            vectors = {}
        missing = [chunk for chunk in chunks if chunk["id"] not in vectors]
        if missing:
            generated = self.client.embed_texts([chunk["content"] for chunk in missing])
            if len(generated) != len(missing) or any(not vector for vector in generated):
                raise ProviderError("Embedding provider returned invalid vectors", "provider_invalid_response")
            generated = [[float(value) for value in vector] for vector in generated]
            records = [{"chunk_id": chunk["id"], "model": self.client.model,
                        "dimensions": len(vector), "vector": vector}
                       for chunk, vector in zip(missing, generated)]
            self.database.upsert_embeddings(records)
            vectors.update({chunk["id"]: vector for chunk, vector in zip(missing, generated)})
        return vectors

    def search(self, query: str, top_k: int = 5, **kwargs) -> list[SearchResult]:
        return self.search_with_status(query, top_k, **kwargs).results

    def search_with_status(self, query: str, top_k: int = 5, *, min_score: float = 0.0,
                           max_candidates: int = 50, document_id: str | None = None,
                           filename: str | None = None) -> RetrievalOutcome:
        fallback_kwargs = {"min_score": min_score, "max_candidates": max_candidates,
                           "document_id": document_id, "filename": filename}
        if not query.strip():
            return RetrievalOutcome([], RetrievalStatus("embedding", "bm25" if self.client is None else "embedding",
                                                       "provider_not_configured" if self.client is None else None,
                                                       self.client is not None))
        if self.client is None:
            return RetrievalOutcome(self.fallback.search(query, top_k, **fallback_kwargs),
                                    RetrievalStatus("embedding", "bm25", "provider_not_configured", False))
        chunks = [chunk for chunk in self.database.get_chunks() if _matches(chunk, document_id, filename)]
        if not chunks:
            return RetrievalOutcome([], RetrievalStatus("embedding", "embedding", None, True))
        try:
            vectors = self._load_vectors(chunks)
            query_vector = [float(value) for value in self.client.embed_texts([query])[0]]
            dimensions = len(query_vector)
            if any(len(vector) != dimensions for vector in vectors.values()):
                self.database.delete_embeddings(self.client.model)
                vectors = self._load_vectors(chunks)
        except ProviderError as exc:
            return RetrievalOutcome(self.fallback.search(query, top_k, **fallback_kwargs),
                                    RetrievalStatus("embedding", "bm25", exc.reason, False))
        except (OSError, ValueError, TypeError, IndexError, KeyError):
            return RetrievalOutcome(self.fallback.search(query, top_k, **fallback_kwargs),
                                    RetrievalStatus("embedding", "bm25", "provider_request_failed", False))
        query_norm = sum(value * value for value in query_vector) ** 0.5
        ranked = []
        for chunk in chunks:
            vector = vectors.get(chunk["id"])
            if not vector or len(vector) != len(query_vector):
                continue
            vector_norm = sum(value * value for value in vector) ** 0.5
            cosine = sum(left * right for left, right in zip(query_vector, vector))
            score = cosine / (query_norm * vector_norm) if query_norm and vector_norm else 0.0
            ranked.append((score, chunk))
        results = [SearchResult(
            chunk_id=chunk["id"], document_id=chunk["document_id"], filename=chunk["filename"],
            chunk_index=chunk["chunk_index"], content=chunk["content"], score=float(max(0.0, min(1.0, (score + 1) / 2))),
            title_path=chunk.get("title_path", []),
        ) for score, chunk in sorted(ranked, key=lambda item: item[0], reverse=True)[:_bounded(max_candidates)]]
        return RetrievalOutcome(_finish(results, query, top_k, min_score),
                                RetrievalStatus("embedding", "embedding", None, True))


class HybridRetriever:
    method = "hybrid"

    def __init__(self, database, config: Settings | None = None):
        config = config or Settings()
        self.bm25 = Bm25Retriever(database)
        self.embedding = EmbeddingRetriever(database, config=config)
        self.lexical_weight = config.hybrid_lexical_weight
        self.semantic_weight = config.hybrid_semantic_weight

    def search(self, query: str, top_k: int = 5, **kwargs) -> list[SearchResult]:
        return self.search_with_status(query, top_k, **kwargs).results

    def search_with_status(self, query: str, top_k: int = 5, **kwargs) -> RetrievalOutcome:
        max_candidates = kwargs.get("max_candidates", 50)
        lexical = self.bm25.search(query, max_candidates, min_score=0.0,
                                    max_candidates=max_candidates, document_id=kwargs.get("document_id"),
                                    filename=kwargs.get("filename"))
        embedding_kwargs = dict(kwargs)
        embedding_kwargs["max_candidates"] = max_candidates
        semantic_outcome = self.embedding.search_with_status(query, max_candidates, **embedding_kwargs)
        semantic = semantic_outcome.results
        if not semantic or not semantic_outcome.status.embedding_used:
            return RetrievalOutcome(_finish(lexical, query, top_k, kwargs.get("min_score", 0.0)),
                                    RetrievalStatus("hybrid", "bm25", semantic_outcome.status.fallback_reason, False))
        lexical_by_id = {result.chunk_id: result for result in lexical}
        semantic_by_id = {result.chunk_id: result for result in semantic}
        ranked = []
        for chunk_id in set(lexical_by_id) | set(semantic_by_id):
            lexical_result = lexical_by_id.get(chunk_id)
            semantic_result = semantic_by_id.get(chunk_id)
            result = semantic_result or lexical_result
            score = self.lexical_weight * (lexical_result.score if lexical_result else 0.0)
            score += self.semantic_weight * (semantic_result.score if semantic_result else 0.0)
            ranked.append((score, result))
        results = [SearchResult(
            chunk_id=result.chunk_id, document_id=result.document_id, filename=result.filename,
            chunk_index=result.chunk_index, content=result.content, score=float(score),
            title_path=result.title_path,
        ) for score, result in sorted(ranked, key=lambda item: item[0], reverse=True)[:_bounded(max_candidates)]]
        return RetrievalOutcome(_finish(results, query, top_k, kwargs.get("min_score", 0.0)),
                                RetrievalStatus("hybrid", "hybrid", None, True))


def create_retriever(method: str, database, chunks: list[dict] | None = None,
                     config: Settings | None = None):
    normalized = method.lower().strip()
    if normalized == "bm25":
        return Bm25Retriever(database)
    if normalized == "tfidf":
        signature = database.get_index_signature()
        if signature not in _TFIDF_CACHE:
            _TFIDF_CACHE[signature] = TfidfRetriever(chunks if chunks is not None else database.get_chunks(), signature)
        return _TFIDF_CACHE[signature]
    if normalized == "embedding":
        return EmbeddingRetriever(database, config=config)
    if normalized == "hybrid":
        return HybridRetriever(database, config=config)
    raise ValueError(f"unsupported retrieval method: {method}")
