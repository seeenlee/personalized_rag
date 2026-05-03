"""Experiment execution loop."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .datasets import LoadedDataset, load_dataset
from .experiment_config import (
    AlphaConfig,
    ExperimentConfig,
    ExperimentFactor,
    alpha_run,
    iter_experiment_factors,
)
from .pinecone_client import (
    connect_to_index,
    delete_user_vector,
    embed_query,
    fetch_user_vector,
    load_api_key,
    query_chunks,
    upsert_user_vector,
)
from .reranking import rerank_chunks
from .scoring import find_expected_rank, persona_rank_score
from .vector_math import combine_vectors, moving_average_user_vector


@dataclass(frozen=True)
class RetrievalRun:
    """Retrieved chunk IDs and score for one query."""

    chunk_ids: list[str]
    score: float
    expected_rank: int | None


@dataclass(frozen=True)
class EvaluationResult:
    """Detailed baseline and post-priming result for one case."""

    run_id: str
    dataset: str
    namespace: str
    index_name: str
    user_namespace: str
    embed_model: str
    top_k: int
    combine_strategy: str
    rerank_strategy: str
    update_strategy: str
    alpha_config: AlphaConfig
    final_alpha: float | None
    persona: str
    question_number: int
    username: str
    neutral_question: str
    expected_chunk_id: str
    priming_question_count: int
    baseline: RetrievalRun
    post_priming: RetrievalRun

    @property
    def delta(self) -> float:
        return self.post_priming.score - self.baseline.score


@dataclass(frozen=True)
class DryRunDatasetCount:
    """Dry-run case count for one dataset."""

    dataset: str
    namespace: str
    persona_count: int
    question_count: int
    case_count: int


@dataclass(frozen=True)
class DryRunSummary:
    """Dry-run matrix summary."""

    factor_count: int
    dataset_counts: list[DryRunDatasetCount]

    @property
    def total_cases(self) -> int:
        return sum(item.case_count for item in self.dataset_counts)


def score_retrieval(persona: str, expected_chunk_id: str, chunk_ids: list[str]) -> RetrievalRun:
    """Score a retrieval result."""
    return RetrievalRun(
        chunk_ids=chunk_ids,
        score=persona_rank_score(
            user_type=persona,
            expected_chunk=expected_chunk_id,
            retrieved_chunks=chunk_ids,
        ),
        expected_rank=find_expected_rank(chunk_ids, expected_chunk_id),
    )


def retrieve_chunk_ids(
    *,
    pc: Any,
    index: Any,
    username: str,
    query: str,
    namespace: str,
    user_namespace: str,
    embed_model: str,
    top_k: int,
    combine_strategy: str,
    rerank_strategy: str,
    update_strategy: str,
    alpha: float | None,
    cross_encoder_model: str,
) -> list[str]:
    """Run one retrieval through the standalone pipeline."""
    query_vector = embed_query(pc, embed_model, query)
    user_vector = fetch_user_vector(index, user_namespace, username)
    if user_vector is None:
        user_vector = np.zeros_like(query_vector)

    combined_vector = combine_vectors(
        user_vector=user_vector,
        query_vector=query_vector,
        strategy=combine_strategy,
        alpha=alpha,
    )
    chunks = query_chunks(index=index, namespace=namespace, vector=combined_vector, top_k=top_k)
    reranked = rerank_chunks(
        chunks=chunks,
        query=query,
        strategy=rerank_strategy,
        model_name=cross_encoder_model,
    )

    if update_strategy == "moving-average":
        updated = moving_average_user_vector(user_vector, query_vector)
        upsert_user_vector(index, user_namespace, username, updated)
    elif update_strategy != "none":
        raise ValueError(f"unknown update strategy: {update_strategy}")

    return [chunk.chunk_id for chunk in reranked]


def _initial_alpha(alpha_config: AlphaConfig) -> float | None:
    if alpha_config.mode == "none":
        return None
    if alpha_config.mode == "static":
        return alpha_config.value
    return alpha_config.start


def _username(config: ExperimentConfig, dataset: str, persona: str, question_number: int, factor: ExperimentFactor) -> str:
    return (
        f"{config.run_id}-{dataset}-{persona}-q{question_number:02d}-"
        f"{factor.slug}"
    )


def evaluate_case(
    *,
    config: ExperimentConfig,
    pc: Any,
    index: Any,
    loaded_dataset: LoadedDataset,
    factor: ExperimentFactor,
    persona: str,
    question_number: int,
    neutral_question: str,
    expected_chunk_id: str,
) -> EvaluationResult:
    """Evaluate one neutral question before and after optional persona priming."""
    username = _username(
        config=config,
        dataset=loaded_dataset.spec.name,
        persona=persona,
        question_number=question_number,
        factor=factor,
    )
    delete_user_vector(index, config.user_namespace, username)

    baseline_chunk_ids = retrieve_chunk_ids(
        pc=pc,
        index=index,
        username=username,
        query=neutral_question,
        namespace=loaded_dataset.spec.namespace,
        user_namespace=config.user_namespace,
        embed_model=config.embed_model,
        top_k=config.top_k,
        combine_strategy=factor.combine_strategy,
        rerank_strategy=factor.rerank_strategy,
        update_strategy="none",
        alpha=_initial_alpha(factor.alpha_config),
        cross_encoder_model=config.cross_encoder_model,
    )
    baseline = score_retrieval(persona, expected_chunk_id, baseline_chunk_ids)

    priming_questions = loaded_dataset.persona_questions[persona]
    active_priming_questions = (
        priming_questions if factor.update_strategy != "none" else []
    )
    alpha_plan = alpha_run(factor.alpha_config, len(active_priming_questions))
    for priming_question, alpha in zip(
        active_priming_questions,
        alpha_plan.priming_alphas,
        strict=True,
    ):
        retrieve_chunk_ids(
            pc=pc,
            index=index,
            username=username,
            query=priming_question,
            namespace=loaded_dataset.spec.namespace,
            user_namespace=config.user_namespace,
            embed_model=config.embed_model,
            top_k=config.top_k,
            combine_strategy=factor.combine_strategy,
            rerank_strategy=factor.rerank_strategy,
            update_strategy=factor.update_strategy,
            alpha=alpha,
            cross_encoder_model=config.cross_encoder_model,
        )

    post_chunk_ids = retrieve_chunk_ids(
        pc=pc,
        index=index,
        username=username,
        query=neutral_question,
        namespace=loaded_dataset.spec.namespace,
        user_namespace=config.user_namespace,
        embed_model=config.embed_model,
        top_k=config.top_k,
        combine_strategy=factor.combine_strategy,
        rerank_strategy=factor.rerank_strategy,
        update_strategy="none",
        alpha=alpha_plan.final_alpha,
        cross_encoder_model=config.cross_encoder_model,
    )
    post_priming = score_retrieval(persona, expected_chunk_id, post_chunk_ids)

    return EvaluationResult(
        run_id=config.run_id,
        dataset=loaded_dataset.spec.name,
        namespace=loaded_dataset.spec.namespace,
        index_name=config.index_name,
        user_namespace=config.user_namespace,
        embed_model=config.embed_model,
        top_k=config.top_k,
        combine_strategy=factor.combine_strategy,
        rerank_strategy=factor.rerank_strategy,
        update_strategy=factor.update_strategy,
        alpha_config=factor.alpha_config,
        final_alpha=alpha_plan.final_alpha,
        persona=persona,
        question_number=question_number,
        username=username,
        neutral_question=neutral_question,
        expected_chunk_id=expected_chunk_id,
        priming_question_count=len(active_priming_questions),
        baseline=baseline,
        post_priming=post_priming,
    )


def load_configured_datasets(config: ExperimentConfig) -> list[LoadedDataset]:
    """Load every dataset selected by the runtime config."""
    return [
        load_dataset(
            dataset_name,
            limit_personas=config.limit_personas,
            limit_questions=config.limit_questions,
        )
        for dataset_name in config.datasets
    ]


def dry_run_summary(config: ExperimentConfig) -> DryRunSummary:
    """Calculate matrix sizes without touching Pinecone."""
    factors = iter_experiment_factors(config)
    counts: list[DryRunDatasetCount] = []
    for loaded_dataset in load_configured_datasets(config):
        persona_count = len(loaded_dataset.persona_questions)
        question_count = len(loaded_dataset.neutral_questions)
        counts.append(
            DryRunDatasetCount(
                dataset=loaded_dataset.spec.name,
                namespace=loaded_dataset.spec.namespace,
                persona_count=persona_count,
                question_count=question_count,
                case_count=persona_count * question_count * len(factors),
            )
        )
    return DryRunSummary(factor_count=len(factors), dataset_counts=counts)


def run_experiments(config: ExperimentConfig) -> list[EvaluationResult]:
    """Run the configured experiment grid against Pinecone."""
    loaded_datasets = load_configured_datasets(config)
    factors = iter_experiment_factors(config)
    api_key = load_api_key()
    pc, index = connect_to_index(api_key, config.index_name)

    results: list[EvaluationResult] = []
    for loaded_dataset in loaded_datasets:
        if not loaded_dataset.persona_questions:
            print(
                f"Skipping {loaded_dataset.spec.name}: no personas selected",
                flush=True,
            )
            continue
        for factor in factors:
            for persona in loaded_dataset.spec.personas:
                if persona not in loaded_dataset.persona_questions:
                    continue
                for question_index, neutral_question in enumerate(
                    loaded_dataset.neutral_questions,
                    start=1,
                ):
                    result = evaluate_case(
                        config=config,
                        pc=pc,
                        index=index,
                        loaded_dataset=loaded_dataset,
                        factor=factor,
                        persona=persona,
                        question_number=question_index,
                        neutral_question=neutral_question,
                        expected_chunk_id=loaded_dataset.expected_chunk_ids[persona][
                            question_index - 1
                        ],
                    )
                    results.append(result)
                    print_progress(result)
    return results


def print_progress(result: EvaluationResult) -> None:
    """Print a compact progress line."""
    print(
        f"{result.dataset:8} {result.combine_strategy:14} "
        f"{result.rerank_strategy:13} {result.update_strategy:14} "
        f"{result.alpha_config.slug:18} {result.persona:10} "
        f"q{result.question_number:02d} "
        f"baseline={result.baseline.score:.4f} "
        f"post={result.post_priming.score:.4f} "
        f"delta={result.delta:+.4f}",
        flush=True,
    )


def resolve_output_path(path: Path) -> Path:
    """Resolve an output path and create its parent directory."""
    resolved = path.expanduser().resolve()
    resolved.parent.mkdir(parents=True, exist_ok=True)
    return resolved

