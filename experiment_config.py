"""CLI parsing, experiment grid construction, and alpha schedules."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from .datasets import dataset_names
from .pinecone_client import (
    DEFAULT_EMBED_MODEL,
    DEFAULT_INDEX_NAME,
    DEFAULT_TOP_K,
    DEFAULT_USER_NAMESPACE,
)
from .reranking import DEFAULT_CROSS_ENCODER_MODEL, RERANK_STRATEGIES
from .vector_math import COMBINE_STRATEGIES, UPDATE_STRATEGIES, validate_alpha

DEFAULT_STATIC_ALPHAS = (0.5, 0.8, 0.95)
DEFAULT_LINEAR_ALPHA_START = 0.99
DEFAULT_LINEAR_ALPHA_STEP = 0.02
DEFAULT_LINEAR_ALPHA_FLOOR = 0.70
DEFAULT_SPHERICAL_ALPHA_START = 0.99
DEFAULT_SPHERICAL_ALPHA_STEP = 0.05
DEFAULT_SPHERICAL_ALPHA_FLOOR = 0.50


@dataclass(frozen=True)
class AlphaConfig:
    """Alpha configuration for one experiment factor."""

    mode: str
    value: float | None = None
    start: float | None = None
    step: float | None = None
    floor: float | None = None

    @property
    def slug(self) -> str:
        """Return a compact stable identifier."""
        if self.mode == "none":
            return "no-alpha"
        if self.mode == "static":
            return f"static-{_float_slug(self.value)}"
        return (
            f"sliding-{_float_slug(self.start)}-"
            f"{_float_slug(self.step)}-{_float_slug(self.floor)}"
        )


@dataclass(frozen=True)
class AlphaRun:
    """Concrete alpha values for the priming pass and final query."""

    priming_alphas: list[float | None]
    final_alpha: float | None


@dataclass(frozen=True)
class ExperimentFactor:
    """One point in the experiment grid."""

    combine_strategy: str
    rerank_strategy: str
    update_strategy: str
    alpha_config: AlphaConfig

    @property
    def slug(self) -> str:
        return "-".join(
            (
                self.combine_strategy,
                self.rerank_strategy,
                self.update_strategy,
                self.alpha_config.slug,
            )
        )


@dataclass(frozen=True)
class ExperimentConfig:
    """Runtime configuration for an experiment run."""

    run_id: str
    index_name: str
    user_namespace: str
    embed_model: str
    top_k: int
    datasets: tuple[str, ...]
    combine_strategies: tuple[str, ...]
    rerank_strategies: tuple[str, ...]
    update_strategies: tuple[str, ...]
    static_alphas: tuple[float, ...]
    output_dir: Path
    details_csv: Path
    summary_csv: Path
    dry_run: bool
    limit_personas: set[str] | None
    limit_questions: int | None
    cross_encoder_model: str


def generate_run_id() -> str:
    """Generate a sortable run identifier."""
    return datetime.now(timezone.utc).strftime("run_%Y%m%dT%H%M%SZ")


def _float_slug(value: float | None) -> str:
    if value is None:
        return "none"
    return f"{value:g}".replace(".", "p")


def alpha_value(raw: str) -> float:
    """Parse and validate a CLI alpha value."""
    try:
        return validate_alpha(float(raw))
    except ValueError as exc:
        raise argparse.ArgumentTypeError("alpha must be a float between 0 and 1") from exc


def alpha_run(alpha_config: AlphaConfig, priming_count: int) -> AlphaRun:
    """Generate alpha values for priming queries and the final neutral query."""
    if priming_count < 0:
        raise ValueError("priming_count cannot be negative")
    if alpha_config.mode == "none":
        return AlphaRun(priming_alphas=[None] * priming_count, final_alpha=None)
    if alpha_config.mode == "static":
        return AlphaRun(
            priming_alphas=[alpha_config.value] * priming_count,
            final_alpha=alpha_config.value,
        )
    if alpha_config.mode != "sliding":
        raise ValueError(f"unknown alpha mode: {alpha_config.mode}")

    if (
        alpha_config.start is None
        or alpha_config.step is None
        or alpha_config.floor is None
    ):
        raise ValueError("sliding alpha requires start, step, and floor")
    current = alpha_config.start
    values: list[float] = []
    for _ in range(priming_count):
        values.append(round(current, 10))
        current = max(alpha_config.floor, current - alpha_config.step)
    return AlphaRun(priming_alphas=values, final_alpha=round(current, 10))


def alpha_configs_for_strategy(
    combine_strategy: str,
    static_alphas: tuple[float, ...] = DEFAULT_STATIC_ALPHAS,
) -> tuple[AlphaConfig, ...]:
    """Return applicable alpha configs for one combination strategy."""
    if combine_strategy == "query-only":
        return (AlphaConfig(mode="none"),)
    static = tuple(AlphaConfig(mode="static", value=alpha) for alpha in static_alphas)
    if combine_strategy == "linear-comb":
        return static + (
            AlphaConfig(
                mode="sliding",
                start=DEFAULT_LINEAR_ALPHA_START,
                step=DEFAULT_LINEAR_ALPHA_STEP,
                floor=DEFAULT_LINEAR_ALPHA_FLOOR,
            ),
        )
    if combine_strategy == "spherical-comb":
        return static + (
            AlphaConfig(
                mode="sliding",
                start=DEFAULT_SPHERICAL_ALPHA_START,
                step=DEFAULT_SPHERICAL_ALPHA_STEP,
                floor=DEFAULT_SPHERICAL_ALPHA_FLOOR,
            ),
        )
    raise ValueError(f"unknown combination strategy: {combine_strategy}")


def iter_experiment_factors(config: ExperimentConfig) -> list[ExperimentFactor]:
    """Build the full experiment factor grid."""
    factors: list[ExperimentFactor] = []
    for combine_strategy in config.combine_strategies:
        for alpha_config in alpha_configs_for_strategy(
            combine_strategy,
            config.static_alphas,
        ):
            for rerank_strategy in config.rerank_strategies:
                for update_strategy in config.update_strategies:
                    factors.append(
                        ExperimentFactor(
                            combine_strategy=combine_strategy,
                            rerank_strategy=rerank_strategy,
                            update_strategy=update_strategy,
                            alpha_config=alpha_config,
                        )
                    )
    return factors


def parse_args(argv: list[str] | None = None) -> ExperimentConfig:
    """Parse CLI arguments into an experiment config."""
    parser = argparse.ArgumentParser(description="Run standalone persona experiments.")
    parser.add_argument("--index-name", default=DEFAULT_INDEX_NAME)
    parser.add_argument("--user-namespace", default=DEFAULT_USER_NAMESPACE)
    parser.add_argument("--embed-model", default=DEFAULT_EMBED_MODEL)
    parser.add_argument("--top-k", type=int, default=DEFAULT_TOP_K)
    parser.add_argument(
        "--datasets",
        nargs="+",
        choices=dataset_names(),
        default=list(dataset_names()),
    )
    parser.add_argument(
        "--combine-strategies",
        nargs="+",
        choices=COMBINE_STRATEGIES,
        default=list(COMBINE_STRATEGIES),
    )
    parser.add_argument(
        "--rerank-strategies",
        nargs="+",
        choices=RERANK_STRATEGIES,
        default=list(RERANK_STRATEGIES),
    )
    parser.add_argument(
        "--update-strategies",
        nargs="+",
        choices=UPDATE_STRATEGIES,
        default=list(UPDATE_STRATEGIES),
    )
    parser.add_argument(
        "--static-alphas",
        nargs="+",
        type=alpha_value,
        default=list(DEFAULT_STATIC_ALPHAS),
    )
    parser.add_argument("--output-dir", type=Path, default=Path("new") / "results")
    parser.add_argument("--details-csv", type=Path, default=None)
    parser.add_argument("--summary-csv", type=Path, default=None)
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--limit-personas", nargs="+", default=None)
    parser.add_argument("--limit-questions", type=int, default=None)
    parser.add_argument("--cross-encoder-model", default=DEFAULT_CROSS_ENCODER_MODEL)
    args = parser.parse_args(argv)

    if args.top_k < 1:
        parser.error("--top-k must be at least 1")
    if args.limit_questions is not None and args.limit_questions < 1:
        parser.error("--limit-questions must be at least 1")

    run_id = args.run_id or generate_run_id()
    output_dir = args.output_dir
    details_csv = args.details_csv or output_dir / f"{run_id}_details.csv"
    summary_csv = args.summary_csv or output_dir / f"{run_id}_summary.csv"

    return ExperimentConfig(
        run_id=run_id,
        index_name=args.index_name,
        user_namespace=args.user_namespace,
        embed_model=args.embed_model,
        top_k=args.top_k,
        datasets=tuple(args.datasets),
        combine_strategies=tuple(args.combine_strategies),
        rerank_strategies=tuple(args.rerank_strategies),
        update_strategies=tuple(args.update_strategies),
        static_alphas=tuple(args.static_alphas),
        output_dir=output_dir,
        details_csv=details_csv,
        summary_csv=summary_csv,
        dry_run=args.dry_run,
        limit_personas=set(args.limit_personas) if args.limit_personas else None,
        limit_questions=args.limit_questions,
        cross_encoder_model=args.cross_encoder_model,
    )

