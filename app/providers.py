"""Optional OpenAI-compatible providers used by retrieval and generation."""

import json
import time
from typing import Any

import httpx

from .config import Settings


class ProviderError(RuntimeError):
    """A provider could not return a valid response."""

    def __init__(self, message: str, reason: str = "provider_request_failed"):
        super().__init__(message)
        self.reason = reason


def _headers(api_key: str) -> dict[str, str]:
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    return headers


class OpenAICompatibleEmbeddingClient:
    def __init__(
        self,
        api_base: str,
        api_key: str,
        model: str,
        timeout: float,
        transport: httpx.BaseTransport | None = None,
    ):
        self.api_base = api_base.rstrip("/")
        self.api_key = api_key
        self.model = model
        self.timeout = timeout
        self.transport = transport

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        payload = {"model": self.model, "input": texts}
        try:
            with httpx.Client(timeout=self.timeout, transport=self.transport) as client:
                response = client.post(
                    f"{self.api_base}/embeddings", json=payload,
                    headers=_headers(self.api_key),
                )
                response.raise_for_status()
        except httpx.HTTPError as exc:
            raise ProviderError(f"Embedding provider request failed: {exc}") from exc
        try:
            body = response.json()
        except ValueError as exc:
            raise ProviderError(
                "Embedding provider returned invalid JSON", "provider_invalid_response"
            ) from exc

        data = body.get("data") if isinstance(body, dict) else None
        if not isinstance(data, list) or len(data) != len(texts):
            raise ProviderError(
                "Embedding provider returned an invalid data array",
                "provider_invalid_response",
            )
        try:
            if any(not isinstance(item, dict) for item in data):
                raise ValueError("items are not objects")
            indices = [int(item["index"]) for item in data]
            if sorted(indices) != list(range(len(texts))):
                raise ValueError("embedding indices are not a complete sequence")
            ordered = sorted(data, key=lambda item: int(item["index"]))
            vectors = [item["embedding"] for item in ordered]
            if any(not isinstance(vector, list) or not vector for vector in vectors):
                raise ValueError("empty embedding")
            converted = [[float(value) for value in vector] for vector in vectors]
            if any(not vector for vector in converted):
                raise ValueError("empty embedding")
            return converted
        except (KeyError, TypeError, ValueError) as exc:
            raise ProviderError(
                "Embedding provider returned invalid vectors", "provider_invalid_response"
            ) from exc


class OpenAICompatibleChatClient:
    def __init__(
        self,
        api_base: str,
        api_key: str,
        model: str,
        timeout: float,
        transport: httpx.BaseTransport | None = None,
        max_retries: int = 0,
    ):
        self.api_base = api_base.rstrip("/")
        self.api_key = api_key
        self.model = model
        self.timeout = timeout
        self.max_retries = max(0, int(max_retries))
        self.transport = transport

    def complete_json(self, messages: list[dict[str, str]]) -> dict[str, Any]:
        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": 0,
            "response_format": {"type": "json_object"},
        }
        last_error: ProviderError | None = None
        for attempt in range(self.max_retries + 1):
            try:
                with httpx.Client(timeout=self.timeout, transport=self.transport) as client:
                    response = client.post(
                        f"{self.api_base}/chat/completions", json=payload,
                        headers=_headers(self.api_key),
                    )
                    response.raise_for_status()
                try:
                    body = response.json()
                    content = body["choices"][0]["message"]["content"]
                    if not isinstance(content, str):
                        raise TypeError("content is not text")
                    return json.loads(content)
                except (KeyError, IndexError, TypeError, json.JSONDecodeError, ValueError) as exc:
                    last_error = ProviderError(
                        "LLM provider returned invalid JSON content", "provider_invalid_response"
                    )
            except httpx.TimeoutException as exc:
                last_error = ProviderError(f"LLM provider request timed out: {exc}", "provider_timeout")
            except httpx.HTTPStatusError as exc:
                last_error = ProviderError(f"LLM provider returned HTTP error: {exc}", "provider_http_error")
            except httpx.RequestError as exc:
                last_error = ProviderError(f"LLM provider request failed: {exc}", "provider_unavailable")
            except Exception as exc:
                last_error = ProviderError(f"LLM provider request failed: {exc}", "provider_request_failed")
            if attempt < self.max_retries:
                time.sleep(min(0.25 * (2 ** attempt), 1.0))
        raise last_error or ProviderError("LLM provider request failed")


def create_embedding_client(config: Settings) -> OpenAICompatibleEmbeddingClient | None:
    """Build an embedding client from explicitly supplied configuration."""
    if not config.embedding_api_base or not config.embedding_model:
        return None
    return OpenAICompatibleEmbeddingClient(
        config.embedding_api_base,
        config.embedding_api_key,
        config.embedding_model,
        config.embedding_timeout_seconds,
    )


def create_chat_client(config: Settings) -> OpenAICompatibleChatClient | None:
    """Build a chat client from explicitly supplied configuration."""
    if not config.llm_api_base or not config.llm_model:
        return None
    return OpenAICompatibleChatClient(
        config.llm_api_base,
        config.llm_api_key,
        config.llm_model,
        config.llm_timeout_seconds,
        max_retries=config.llm_max_retries,
    )
