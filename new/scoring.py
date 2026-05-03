"""Scoring helpers for persona retrieval experiments."""

from __future__ import annotations

import math

GROUND_TRUTH_SCORE = 10.0
PERSONA_ALIGNMENT_SCORE = 1.0


def persona_rank_score(
    user_type: str,
    expected_chunk: str,
    retrieved_chunks: list[str],
) -> float:
    """Score ranked chunks by expected hit and persona-aligned hits."""
    total_score = 0.0
    for rank, retrieved_chunk in enumerate(retrieved_chunks, start=1):
        rank_discount = math.log(rank + 1, 2)
        if retrieved_chunk == expected_chunk:
            total_score += GROUND_TRUTH_SCORE / rank_discount
            continue
        chunk_category = retrieved_chunk.split("-", 1)[0]
        if chunk_category == user_type:
            total_score += PERSONA_ALIGNMENT_SCORE / rank_discount
    return total_score


def find_expected_rank(chunk_ids: list[str], expected_chunk_id: str) -> int | None:
    """Return the 1-based rank of the expected chunk if it was retrieved."""
    for rank, chunk_id in enumerate(chunk_ids, start=1):
        if chunk_id == expected_chunk_id:
            return rank
    return None

