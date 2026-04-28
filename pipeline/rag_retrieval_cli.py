"""Interactive CLI for testing RAG retrieval against Pinecone."""

import argparse
import importlib
import os
import sys
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any
import numpy as np

PINECONE_ENV_VAR_NAME = "PINECONE_API_KEY"
DEFAULT_INDEX_NAME = "541"
DEFAULT_RETRIEVAL_NAMESPACE = "civil"
DEFAULT_USER_NAMESPACE = "users"
DEFAULT_EMBED_MODEL = "llama-text-embed-v2"
DEFAULT_TOP_K = 5
TEXT_FIELD_NAME = "chunk_text"
METADATA_TEXT_FIELD = "text"

COMBINE_STRATEGIES = ("query-only", "linear-comb", "spherical-comb", "average")
RERANK_STRATEGIES = ("none", "cross-encoder", "llm")
UPDATE_STRATEGIES = ("none", "moving-average", "replace")


@dataclass(frozen=True)
class RetrievedChunk:
    """A normalized retrieval result from Pinecone."""

    chunk_id: str
    text: str
    score: float | None


def load_api_key() -> str:
    """Load the Pinecone API key from the repo .env or environment."""
    repo_root = Path(__file__).resolve().parents[1]
    load_dotenv = importlib.import_module("dotenv").load_dotenv
    load_dotenv(repo_root / ".env")

    api_key = os.getenv(PINECONE_ENV_VAR_NAME)
    if not api_key:
        raise RuntimeError(
            f"Missing {PINECONE_ENV_VAR_NAME}. Add it to the repo .env or your "
            "environment."
        )

    return api_key


def parse_args(argv: list[str]) -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Run an interactive RAG retrieval pipeline against Pinecone."
    )
    parser.add_argument(
        "--index-name",
        default=DEFAULT_INDEX_NAME,
        help="Name of the Pinecone index to query",
    )
    parser.add_argument(
        "--namespace",
        default=DEFAULT_RETRIEVAL_NAMESPACE,
        help="Namespace containing document chunks",
    )
    parser.add_argument(
        "--user-namespace",
        default=DEFAULT_USER_NAMESPACE,
        help="Namespace containing per-user vectors",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=DEFAULT_TOP_K,
        help="Number of chunks to retrieve",
    )
    parser.add_argument(
        "--embed-model",
        default=DEFAULT_EMBED_MODEL,
        help="Pinecone embedding model for query embeddings",
    )
    parser.add_argument(
        "--combine-strategy",
        choices=COMBINE_STRATEGIES,
        default="query-only",
        help="Strategy for combining user and query vectors",
    )
    parser.add_argument(
        "--rerank-strategy",
        choices=RERANK_STRATEGIES,
        default="none",
        help="Strategy for reranking retrieved chunks",
    )
    parser.add_argument(
        "--update-strategy",
        choices=UPDATE_STRATEGIES,
        default="none",
        help="Strategy for updating the user vector after retrieval",
    )
    return parser.parse_args(argv)


def connect_to_index(api_key: str, index_name: str) -> tuple[Any, Any]:
    """Connect to an existing Pinecone index."""
    Pinecone = importlib.import_module("pinecone").Pinecone
    pc = Pinecone(api_key=api_key)
    if not pc.has_index(index_name):
        raise RuntimeError(f"Pinecone index '{index_name}' does not exist.")

    index_host = pc.describe_index(name=index_name).host
    return pc, pc.Index(host=index_host)


def _safe_get(obj: Any, key: str) -> Any:
    """Read a key from either a mapping or an object attribute."""
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


def _coerce_float_vector(value: Any) -> list[float] | None:
    """Convert a sequence of numeric values into floats."""
    values = _as_sequence(value)
    if not values:
        return None

    try:
        return [float(item) for item in values]
    except (TypeError, ValueError):
        return None


def _extract_vector_from_record(record: Any) -> list[float] | None:
    """Extract vector values from common Pinecone record shapes."""
    candidates = [
        _safe_get(record, "values"),
        _safe_get(record, "vector"),
        _safe_get(record, "embedding"),
    ]

    for candidate in candidates:
        vector = _coerce_float_vector(candidate)
        if vector is not None:
            return vector

        nested_values = _safe_get(candidate, "values")
        vector = _coerce_float_vector(nested_values)
        if vector is not None:
            return vector

    return None


def fetch_user_vector(index: Any, namespace: str, username: str) -> list[float] | None:
    """Fetch a user vector by username from Pinecone."""
    result = index.fetch(ids=[username], namespace=namespace)
    vectors = _safe_get(result, "vectors")
    if vectors is None:
        return None

    if isinstance(vectors, Mapping):
        record = vectors.get(username)
    else:
        matching_records = [
            vector
            for vector in _as_sequence(vectors)
            if (_safe_get(vector, "id") or _safe_get(vector, "_id")) == username
        ]
        record = matching_records[0] if matching_records else None

    if record is None:
        return None

    return _extract_vector_from_record(record)


def embed_query(pc: Any, model: str, query: str) -> list[float]:
    """Embed a query with Pinecone Inference."""
    response = pc.inference.embed(
        model=model,
        inputs=[query],
        parameters={"input_type": "query", "truncate": "END"},
    )

    for container_key in ("data", "embeddings", "results"):
        records = _as_sequence(_safe_get(response, container_key))
        for record in records:
            vector = _extract_vector_from_record(record)
            if vector is not None:
                return vector

    for record in _as_sequence(response):
        vector = _extract_vector_from_record(record)
        if vector is not None:
            return vector

    vector = _extract_vector_from_record(response)
    if vector is not None:
        return vector

    raise RuntimeError("Unable to extract query embedding from Pinecone response.")

def linear_combination(user_vector, query_vector, alpha):
    """
    Performs a weighted linear combination (Lerp) of two vectors.
    alpha = 0.0 returns v_user (User Vector)
    alpha = 1.0 returns v_query (Query Vector)
    """
    # 1. Perform the weighted addition
    # Formula: (1 - alpha) * v_user + alpha * v_query
    v_fused = ((1 - alpha) * user_vector) + (alpha * query_vector)
    
    # 2. Re-normalize to a unit vector (Length = 1)
    # This is essential for Cosine Similarity search!
    norm = np.linalg.norm(v_fused)
    
    if norm > 0:
        return v_fused / norm
    else:
        # Fallback in case of zero-vector (unlikely with embeddings)
        return query_vector


def spherical_combination(v0, v1, alpha):
    """
    Spherical linear interpolation between two normalized vectors.
    alpha = 0.0 returns v0 (User Vector)
    alpha = 1.0 returns v1 (Query Vector)
    """
    # Ensure inputs are unit vectors
    v0 = v0 / np.linalg.norm(v0)
    v1 = v1 / np.linalg.norm(v1)
    
    # Compute the cosine of the angle between the vectors
    dot = np.sum(v0 * v1)
    
    # Clip to avoid errors from floating point precision
    dot = np.clip(dot, -1.0, 1.0)
    
    # Calculate the angle theta
    theta_0 = np.arccos(dot)
    theta = theta_0 * alpha
    
    # Compute the orthogonal vector v2
    v2 = v1 - v0 * dot
    v2 = v2 / (np.linalg.norm(v2) + 1e-10) # Avoid division by zero
    
    # Calculate the final interpolated vector
    return v0 * np.cos(theta) + v2 * np.sin(theta)


def combine_vectors(
    user_vector: list[float], query_vector: list[float], strategy: str
) -> list[float]:
    """Combine user and query vectors with the selected strategy."""
    if strategy == "query-only":
        return query_vector
    elif strategy == "linear-comb":
        return linear_combination(user_vector, query_vector, alpha=0.8)
    elif strategy == "spherical-comb":
        return spherical_combination(user_vector, query_vector, alpha=0.5)


    raise NotImplementedError(
        f"Combination strategy '{strategy}' is not implemented yet."
    )


def search_chunks(
    index: Any, namespace: str, combined_vector: list[float], top_k: int
) -> Any:
    """Search Pinecone chunks by dense vector."""
    return index.query(
        namespace=namespace,
        vector=combined_vector,
        top_k=top_k,
        include_metadata=True,
        include_values=False,
    )


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
        score = _safe_get(raw_hit, "_score") or _safe_get(raw_hit, "score")
        normalized.append(
            RetrievedChunk(
                chunk_id=str(chunk_id),
                text=str(text).strip(),
                score=float(score) if score is not None else None,
            )
        )

    return normalized


def rerank_chunks(chunks: list[RetrievedChunk], strategy: str) -> list[RetrievedChunk]:
    """Rerank chunks with the selected strategy."""
    if strategy == "none":
        return chunks

    raise NotImplementedError(f"Rerank strategy '{strategy}' is not implemented yet.")


def update_user_vector(
    index: Any,
    username: str,
    user_namespace: str,
    user_vector: list[float],
    query_vector: list[float],
    strategy: str,
) -> None:
    """Update the user vector with the selected strategy."""
    _ = (index, username, user_namespace, user_vector, query_vector)

    if strategy == "none":
        return

    raise NotImplementedError(
        f"User update strategy '{strategy}' is not implemented yet."
    )


def prompt_required(prompt: str) -> str:
    """Prompt until the user enters a non-empty value."""
    while True:
        value = input(prompt).strip()
        if value:
            return value
        print("Please enter a non-empty value.")


def print_result(chunks: list[RetrievedChunk]) -> None:
    """Print all retrieved chunks in their current ranking order."""
    if not chunks:
        print("\nNo chunks retrieved.")
        return

    print(f"\nRetrieved chunks ({len(chunks)} results)")
    for rank, chunk in enumerate(chunks, start=1):
        print(f"\nRank {rank}")
        print(f"ID: {chunk.chunk_id}")
        if chunk.score is not None:
            print(f"Score: {chunk.score:.6f}")
        print("Chunk:")
        print(chunk.text or "[No text returned]")


def run_pipeline(args: argparse.Namespace) -> None:
    """Run the interactive retrieval pipeline."""
    api_key = load_api_key()
    pc, index = connect_to_index(api_key, args.index_name)

    print("RAG Retrieval CLI Pipeline")
    username = prompt_required("Username: ")
    query = prompt_required("Query: ")

    query_vector = embed_query(pc, args.embed_model, query)
    user_vector = fetch_user_vector(index, args.user_namespace, username)
    if user_vector is None:
        user_vector = [0.0] * len(query_vector)
        print(
            f"No user vector found for '{username}' in namespace "
            f"'{args.user_namespace}'. Using a zero vector."
        )

    combined_vector = combine_vectors(
        user_vector=user_vector,
        query_vector=query_vector,
        strategy=args.combine_strategy,
    )
    search_result = search_chunks(
        index=index,
        namespace=args.namespace,
        combined_vector=combined_vector,
        top_k=args.top_k,
    )
    chunks = normalize_hits(extract_hits(search_result))
    reranked_chunks = rerank_chunks(chunks, args.rerank_strategy)
    update_user_vector(
        index=index,
        username=username,
        user_namespace=args.user_namespace,
        user_vector=user_vector,
        query_vector=query_vector,
        strategy=args.update_strategy,
    )
    print_result(reranked_chunks)


def main(argv: list[str] | None = None) -> int:
    """Run the command-line interface."""
    args = parse_args(sys.argv[1:] if argv is None else argv)

    try:
        run_pipeline(args)
    except (KeyboardInterrupt, EOFError):
        print("\nExiting.", file=sys.stderr)
        return 130
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
