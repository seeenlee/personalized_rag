from __future__ import annotations
import math


def persona_rank_score(
    user_type: str,
    expected_chunk: str,
    retrieved_chunks: list[str],
) -> float:
    ground_truth_score = 10
    user_chunk_alignment_score = 1

    total_score = 0.0
    for rank, retrieved_chunk in enumerate(retrieved_chunks, start=1):
        rank_discount = math.log(rank + 1, 2)
        if retrieved_chunk == expected_chunk:
            total_score += ground_truth_score / rank_discount
            continue

        chunk_category = retrieved_chunk.split("-")[0]
        if chunk_category == user_type:
            total_score += user_chunk_alignment_score / rank_discount
        # else:
        #     total_score -= user_chunk_alignment_score / math.log(i + 1)

    return total_score