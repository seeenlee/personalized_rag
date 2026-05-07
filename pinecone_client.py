"""Standalone Pinecone access helpers for the experiment runner."""

from __future__ import annotations

import importlib
import os
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

PINECONE_ENV_VAR_NAME = "PINECONE_API_KEY"
DEFAULT_INDEX_NAME = "541"
DEFAULT_USER_NAMESPACE = "users"
DEFAULT_EMBED_MODEL = "llama-text-embed-v2"
DEFAULT_TOP_K = 5
TEXT_FIELD_NAME = "chunk_text"
METADATA_TEXT_FIELD = "text"


@dataclass(frozen=True)
class RetrievedChunk:
    """Normalized retrieval hit."""

    chunk_id: str
    text: str
    score: float | None


def repo_root() -> Path:
    """Return the repository root for this package."""
    return Path(__file__).resolve().parents[1]


def load_api_key(env_path: Path | None = None) -> str:
    """Load the Pinecone API key from `.env` or the process environment."""
    dotenv = importlib.import_module("dotenv")
    dotenv.load_dotenv(env_path or repo_root() / ".env")
    api_key = os.getenv(PINECONE_ENV_VAR_NAME)
    if not api_key:
        raise RuntimeError(
            f"Missing {PINECONE_ENV_VAR_NAME}. Add it to the repo .env or environment."
        )
    return api_key


def connect_to_index(api_key: str, index_name: str = DEFAULT_INDEX_NAME) -> tuple[Any, Any]:
    """Connect to an existing Pinecone index."""
    Pinecone = importlib.import_module("pinecone").Pinecone
    pc = Pinecone(api_key=api_key)
    if hasattr(pc, "has_index") and not pc.has_index(index_name):
        raise RuntimeError(f"Pinecone index '{index_name}' does not exist.")

    host = None
    if hasattr(pc, "describe_index"):
        description = pc.describe_index(name=index_name)
        host = _safe_get(description, "host")

    index = pc.Index(host=host) if host else pc.Index(index_name)
    return pc, index


def _safe_get(obj: Any, key: str) -> Any:
    """Read a key from either a mapping or an object attribute."""
    if obj is None:
        return None
    if isinstance(obj, Mapping):
        return obj.get(key)
    return getattr(obj, key, None)


def _as_sequence(value: Any) -> list[Any]:
    """Normalize list-like values while ignoring strings."""
    if value is None or isinstance(value, (str, bytes)):
        return []
    if isinstance(value, Sequence):
        return list(value)
    return []


def is_missing_namespace_error(exc: Exception) -> bool:
    """Return whether a Pinecone exception represents a missing namespace."""
    parts = [
        str(exc),
        str(getattr(exc, "body", "")),
        str(getattr(exc, "reason", "")),
    ]
    return getattr(exc, "status", None) == 404 and any(
        "Namespace not found" in part for part in parts
    )


def _coerce_float_vector(value: Any) -> np.ndarray | None:
    """Convert a sequence of numeric values into a 1-D NumPy vector."""
    if value is None or isinstance(value, (str, bytes)):
        return None
    try:
        vector = np.asarray(value, dtype=float)
    except (TypeError, ValueError):
        return None
    if vector.ndim != 1 or vector.size == 0:
        return None
    return vector


def _extract_vector_from_record(record: Any) -> np.ndarray | None:
    """Extract vector values from common Pinecone response shapes."""
    candidates = [
        _safe_get(record, "values"),
        _safe_get(record, "vector"),
        _safe_get(record, "embedding"),
    ]
    for candidate in candidates:
        vector = _coerce_float_vector(candidate)
        if vector is not None:
            return vector
        nested = _safe_get(candidate, "values")
        vector = _coerce_float_vector(nested)
        if vector is not None:
            return vector
    return None


def fetch_user_vector(index: Any, user_namespace: str, username: str) -> np.ndarray | None:
    """Fetch a user vector by username from Pinecone."""
    try:
        response = index.fetch(ids=[username], namespace=user_namespace)
    except Exception as exc:
        if is_missing_namespace_error(exc):
            return None
        raise

    vectors = _safe_get(response, "vectors")
    if vectors is None:
        return None
    if isinstance(vectors, Mapping):
        record = vectors.get(username)
    else:
        matches = [
            vector
            for vector in _as_sequence(vectors)
            if (_safe_get(vector, "id") or _safe_get(vector, "_id")) == username
        ]
        record = matches[0] if matches else None
    return _extract_vector_from_record(record)


def delete_user_vector(index: Any, user_namespace: str, username: str) -> None:
    """Delete a user vector if it exists."""
    try:
        index.delete(ids=[username], namespace=user_namespace)
    except Exception as exc:
        if is_missing_namespace_error(exc):
            return
        raise


def upsert_user_vector(index: Any, user_namespace: str, username: str, vector: np.ndarray) -> None:
    """Upsert a user vector into the user namespace."""
    index.upsert(
        vectors=[{"id": username, "values": to_pinecone_vector(vector)}],
        namespace=user_namespace,
    )


def embed_text(pc: Any, model: str, text: str, input_type: str) -> np.ndarray:
    """Embed text with Pinecone Inference."""
    response = pc.inference.embed(
        model=model,
        inputs=[text],
        parameters={"input_type": input_type, "truncate": "END"},
    )

    for container_key in ("data", "embeddings", "results"):
        for record in _as_sequence(_safe_get(response, container_key)):
            vector = _extract_vector_from_record(record)
            if vector is not None:
                return vector.astype(float, copy=False)

    for record in _as_sequence(response):
        vector = _extract_vector_from_record(record)
        if vector is not None:
            return vector.astype(float, copy=False)

    vector = _extract_vector_from_record(response)
    if vector is not None:
        return vector.astype(float, copy=False)

    raise RuntimeError("Unable to extract embedding from Pinecone response.")


def embed_query(pc: Any, model: str, query: str) -> np.ndarray:
    """Embed a query string."""
    return embed_text(pc=pc, model=model, text=query, input_type="query")


def to_pinecone_vector(vector: np.ndarray) -> list[float]:
    """Convert a NumPy vector into Pinecone's dense vector format."""
    return np.asarray(vector, dtype=float).tolist()


def query_chunks(index: Any, namespace: str, vector: np.ndarray, top_k: int) -> list[RetrievedChunk]:
    """Query a chunk namespace and return normalized hits."""
    response = index.query(
        namespace=namespace,
        vector=to_pinecone_vector(vector),
        top_k=top_k,
        include_metadata=True,
        include_values=False,
    )
    return normalize_hits(extract_hits(response))


def extract_hits(search_result: Any) -> list[Any]:
    """Best-effort extraction of Pinecone hits across response shapes."""
    pending = [search_result]
    visited: set[int] = set()
    while pending:
        current = pending.pop(0)
        current_id = id(current)
        if current_id in visited:
            continue
        visited.add(current_id)

        matches = _as_sequence(_safe_get(current, "matches"))
        if matches:
            return matches
        hits = _as_sequence(_safe_get(current, "hits"))
        if hits:
            return hits
        for child_key in ("result", "results", "data"):
            child = _safe_get(current, child_key)
            if child is not None:
                pending.append(child)
    return []


def normalize_hits(raw_hits: list[Any]) -> list[RetrievedChunk]:
    """Convert Pinecone hits into stable retrieved chunks."""
    normalized: list[RetrievedChunk] = []
    for raw_hit in raw_hits:
        vector = _safe_get(raw_hit, "vector")
        field_sources = [
            _safe_get(raw_hit, "fields"),
            _safe_get(raw_hit, "metadata"),
            _safe_get(vector, "metadata"),
            raw_hit,
        ]

        text = ""
        for source in field_sources:
            if source is None:
                continue
            text = (
                _safe_get(source, TEXT_FIELD_NAME)
                or _safe_get(source, METADATA_TEXT_FIELD)
                or _safe_get(source, "content")
                or _safe_get(source, "page_content")
                or ""
            )
            if text:
                break

        chunk_id = (
            _safe_get(raw_hit, "_id")
            or _safe_get(raw_hit, "id")
            or _safe_get(vector, "id")
            or "unknown"
        )
        score = _safe_get(raw_hit, "_score")
        if score is None:
            score = _safe_get(raw_hit, "score")
        normalized.append(
            RetrievedChunk(
                chunk_id=str(chunk_id),
                text=str(text).strip(),
                score=float(score) if score is not None else None,
            )
        )
    return normalized

