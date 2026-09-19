from dataclasses import dataclass
from pathlib import Path
import os


ROOT_DIR = Path(__file__).resolve().parent.parent


@dataclass(frozen=True)
class Settings:
    database_path: Path = Path(os.getenv("DATABASE_PATH", str(ROOT_DIR / "data" / "knowledge.db")))
    upload_dir: Path = Path(os.getenv("UPLOAD_DIR", str(ROOT_DIR / "data" / "uploads")))
    chunk_size: int = int(os.getenv("CHUNK_SIZE", "700"))
    chunk_overlap: int = int(os.getenv("CHUNK_OVERLAP", "100"))
    retrieval_threshold: float = float(os.getenv("RETRIEVAL_THRESHOLD", "0.08"))
    threshold_candidates: str = os.getenv("THRESHOLD_CANDIDATES", "0.05,0.08,0.12,0.16,0.20,0.25,0.30")
    retrieval_method: str = os.getenv("RETRIEVAL_METHOD", "bm25").lower()
    evidence_min_score_gap: float = float(os.getenv("EVIDENCE_MIN_SCORE_GAP", "0.01"))
    embedding_api_base: str = os.getenv("EMBEDDING_API_BASE", "").rstrip("/")
    embedding_api_key: str = os.getenv("EMBEDDING_API_KEY", "")
    embedding_model: str = os.getenv("EMBEDDING_MODEL", "")
    embedding_timeout_seconds: float = float(os.getenv("EMBEDDING_TIMEOUT_SECONDS", "30"))
    llm_api_base: str = os.getenv("LLM_API_BASE", "").rstrip("/")
    llm_api_key: str = os.getenv("LLM_API_KEY", "")
    llm_model: str = os.getenv("LLM_MODEL", "")
    llm_timeout_seconds: float = float(os.getenv("LLM_TIMEOUT_SECONDS", "60"))
    llm_max_retries: int = int(os.getenv("LLM_MAX_RETRIES", "2"))
    llm_max_context_chars: int = int(os.getenv("LLM_MAX_CONTEXT_CHARS", "12000"))
    llm_max_answer_chars: int = int(os.getenv("LLM_MAX_ANSWER_CHARS", "4000"))
    llm_max_citations: int = int(os.getenv("LLM_MAX_CITATIONS", "5"))
    extractive_max_answer_chars: int = int(os.getenv("EXTRACTIVE_MAX_ANSWER_CHARS", "4000"))
    extractive_max_citations: int = int(os.getenv("EXTRACTIVE_MAX_CITATIONS", "3"))
    generation_method: str = os.getenv("GENERATION_METHOD", "extractive").lower()
    max_upload_bytes: int = int(os.getenv("MAX_UPLOAD_BYTES", str(20 * 1024 * 1024)))
    hybrid_lexical_weight: float = float(os.getenv("HYBRID_LEXICAL_WEIGHT", "0.35"))
    hybrid_semantic_weight: float = float(os.getenv("HYBRID_SEMANTIC_WEIGHT", "0.65"))
    retrieval_top_k: int = int(os.getenv("RETRIEVAL_TOP_K", "5"))
    retrieval_min_score: float = float(os.getenv("RETRIEVAL_MIN_SCORE", "0"))
    retrieval_max_candidates: int = int(os.getenv("RETRIEVAL_MAX_CANDIDATES", "50"))
    log_level: str = os.getenv("LOG_LEVEL", "INFO")
    log_text_max_chars: int = int(os.getenv("LOG_TEXT_MAX_CHARS", "1000"))


settings = Settings()
