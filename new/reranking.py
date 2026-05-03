"""Standalone reranking helpers."""

from __future__ import annotations

from dataclasses import replace
from functools import lru_cache
from typing import Any

from .pinecone_client import RetrievedChunk

RERANK_STRATEGIES = ("none", "cross-encoder")
DEFAULT_CROSS_ENCODER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"


@lru_cache(maxsize=4)
def load_cross_encoder(model_name: str) -> Any:
    """Load a sentence-transformers cross encoder."""
    from sentence_transformers import CrossEncoder

    return CrossEncoder(model_name)


def rerank_chunks(
    chunks: list[RetrievedChunk],
    query: str,
    strategy: str,
    *,
    model_name: str = DEFAULT_CROSS_ENCODER_MODEL,
) -> list[RetrievedChunk]:
    """Rerank retrieved chunks with the selected strategy."""
    if strategy == "none":
        return list(chunks)
    if strategy != "cross-encoder":
        raise ValueError(f"unknown rerank strategy: {strategy}")
    if not query.strip():
        return list(chunks)

    pairs = [(query, chunk.text) for chunk in chunks if chunk.text]
    if not pairs:
        return list(chunks)

    model = load_cross_encoder(model_name)
    scores = model.predict(pairs)
    reranked: list[RetrievedChunk] = []
    score_index = 0
    for chunk in chunks:
        if chunk.text:
            reranked.append(replace(chunk, score=float(scores[score_index])))
            score_index += 1
        else:
            reranked.append(chunk)

    return sorted(
        reranked,
        key=lambda chunk: float("-inf") if chunk.score is None else chunk.score,
        reverse=True,
    )

