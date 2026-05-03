"""CSV reporting for standalone experiment results."""

from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path

from .evaluator import EvaluationResult, resolve_output_path

DETAIL_FIELDS = [
    "run_id",
    "dataset",
    "namespace",
    "index_name",
    "user_namespace",
    "embed_model",
    "top_k",
    "combine_strategy",
    "rerank_strategy",
    "update_strategy",
    "alpha_mode",
    "alpha_value",
    "alpha_start",
    "alpha_step",
    "alpha_floor",
    "final_alpha",
    "persona",
    "question_number",
    "username",
    "neutral_question",
    "expected_chunk_id",
    "priming_question_count",
    "baseline_score",
    "post_priming_score",
    "delta",
    "baseline_expected_rank",
    "post_priming_expected_rank",
    "baseline_chunk_ids",
    "post_priming_chunk_ids",
]

SUMMARY_FIELDS = [
    "dataset",
    "namespace",
    "index_name",
    "user_namespace",
    "embed_model",
    "top_k",
    "persona",
    "combine_strategy",
    "rerank_strategy",
    "update_strategy",
    "alpha_mode",
    "alpha_value",
    "alpha_start",
    "alpha_step",
    "alpha_floor",
    "final_alpha",
    "case_count",
    "mean_baseline_score",
    "mean_post_score",
    "mean_delta",
    "wins",
    "mean_baseline_rank",
    "mean_post_rank",
    "expected_retrieval_rate",
]


def _fmt_float(value: float | None) -> str:
    return "" if value is None else f"{value:.6f}"


def _fmt_rank(value: int | None) -> str:
    return "" if value is None else str(value)


def result_to_detail_row(result: EvaluationResult) -> dict[str, str | int]:
    """Convert one result into a detailed CSV row."""
    alpha = result.alpha_config
    return {
        "run_id": result.run_id,
        "dataset": result.dataset,
        "namespace": result.namespace,
        "index_name": result.index_name,
        "user_namespace": result.user_namespace,
        "embed_model": result.embed_model,
        "top_k": result.top_k,
        "combine_strategy": result.combine_strategy,
        "rerank_strategy": result.rerank_strategy,
        "update_strategy": result.update_strategy,
        "alpha_mode": alpha.mode,
        "alpha_value": _fmt_float(alpha.value),
        "alpha_start": _fmt_float(alpha.start),
        "alpha_step": _fmt_float(alpha.step),
        "alpha_floor": _fmt_float(alpha.floor),
        "final_alpha": _fmt_float(result.final_alpha),
        "persona": result.persona,
        "question_number": result.question_number,
        "username": result.username,
        "neutral_question": result.neutral_question,
        "expected_chunk_id": result.expected_chunk_id,
        "priming_question_count": result.priming_question_count,
        "baseline_score": _fmt_float(result.baseline.score),
        "post_priming_score": _fmt_float(result.post_priming.score),
        "delta": _fmt_float(result.delta),
        "baseline_expected_rank": _fmt_rank(result.baseline.expected_rank),
        "post_priming_expected_rank": _fmt_rank(result.post_priming.expected_rank),
        "baseline_chunk_ids": "|".join(result.baseline.chunk_ids),
        "post_priming_chunk_ids": "|".join(result.post_priming.chunk_ids),
    }


def write_detailed_csv(results: list[EvaluationResult], output_path: Path) -> Path:
    """Write detailed per-case CSV output."""
    resolved = resolve_output_path(output_path)
    with resolved.open("w", encoding="utf-8", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=DETAIL_FIELDS)
        writer.writeheader()
        for result in results:
            writer.writerow(result_to_detail_row(result))
    return resolved


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _rank_value(rank: int | None, top_k: int) -> int:
    return rank if rank is not None else top_k + 1


def _summary_key(result: EvaluationResult) -> tuple[object, ...]:
    alpha = result.alpha_config
    return (
        result.dataset,
        result.namespace,
        result.index_name,
        result.user_namespace,
        result.embed_model,
        result.top_k,
        result.persona,
        result.combine_strategy,
        result.rerank_strategy,
        result.update_strategy,
        alpha.mode,
        alpha.value,
        alpha.start,
        alpha.step,
        alpha.floor,
        result.final_alpha,
    )


def summarize_results(results: list[EvaluationResult]) -> list[dict[str, str | int]]:
    """Aggregate detailed results by dataset, persona, and experiment factors."""
    grouped: dict[tuple[object, ...], list[EvaluationResult]] = defaultdict(list)
    for result in results:
        grouped[_summary_key(result)].append(result)

    rows: list[dict[str, str | int]] = []
    for key, group in grouped.items():
        (
            dataset,
            namespace,
            index_name,
            user_namespace,
            embed_model,
            top_k,
            persona,
            combine_strategy,
            rerank_strategy,
            update_strategy,
            alpha_mode,
            alpha_value,
            alpha_start,
            alpha_step,
            alpha_floor,
            final_alpha,
        ) = key
        baseline_scores = [result.baseline.score for result in group]
        post_scores = [result.post_priming.score for result in group]
        deltas = [result.delta for result in group]
        baseline_ranks = [
            _rank_value(result.baseline.expected_rank, result.top_k)
            for result in group
        ]
        post_ranks = [
            _rank_value(result.post_priming.expected_rank, result.top_k)
            for result in group
        ]
        rows.append(
            {
                "dataset": str(dataset),
                "namespace": str(namespace),
                "index_name": str(index_name),
                "user_namespace": str(user_namespace),
                "embed_model": str(embed_model),
                "top_k": int(top_k),
                "persona": str(persona),
                "combine_strategy": str(combine_strategy),
                "rerank_strategy": str(rerank_strategy),
                "update_strategy": str(update_strategy),
                "alpha_mode": str(alpha_mode),
                "alpha_value": _fmt_float(alpha_value),
                "alpha_start": _fmt_float(alpha_start),
                "alpha_step": _fmt_float(alpha_step),
                "alpha_floor": _fmt_float(alpha_floor),
                "final_alpha": _fmt_float(final_alpha),
                "case_count": len(group),
                "mean_baseline_score": _fmt_float(_mean(baseline_scores)),
                "mean_post_score": _fmt_float(_mean(post_scores)),
                "mean_delta": _fmt_float(_mean(deltas)),
                "wins": sum(1 for delta in deltas if delta > 0.0),
                "mean_baseline_rank": _fmt_float(_mean([float(rank) for rank in baseline_ranks])),
                "mean_post_rank": _fmt_float(_mean([float(rank) for rank in post_ranks])),
                "expected_retrieval_rate": _fmt_float(
                    sum(1 for result in group if result.post_priming.expected_rank is not None)
                    / len(group)
                ),
            }
        )

    return sorted(
        rows,
        key=lambda row: (
            str(row["dataset"]),
            str(row["persona"]),
            str(row["combine_strategy"]),
            str(row["rerank_strategy"]),
            str(row["update_strategy"]),
            str(row["alpha_mode"]),
            str(row["alpha_value"]),
        ),
    )


def write_summary_csv(results: list[EvaluationResult], output_path: Path) -> Path:
    """Write aggregate summary CSV output."""
    resolved = resolve_output_path(output_path)
    with resolved.open("w", encoding="utf-8", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=SUMMARY_FIELDS)
        writer.writeheader()
        for row in summarize_results(results):
            writer.writerow(row)
    return resolved

