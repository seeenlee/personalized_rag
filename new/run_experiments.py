"""Master entrypoint for standalone experiments."""

from __future__ import annotations

import sys

from .evaluator import dry_run_summary, run_experiments
from .experiment_config import iter_experiment_factors, parse_args
from .reporting import write_detailed_csv, write_summary_csv


def print_dry_run(config) -> None:
    """Print matrix size without connecting to Pinecone."""
    summary = dry_run_summary(config)
    factors = iter_experiment_factors(config)
    print(f"Run ID: {config.run_id}")
    print(f"Index: {config.index_name}")
    print(f"User namespace: {config.user_namespace}")
    print(f"Experiment factors: {summary.factor_count}")
    print(f"Total cases: {summary.total_cases}")
    for item in summary.dataset_counts:
        print(
            f"- {item.dataset}: namespace={item.namespace}, "
            f"personas={item.persona_count}, questions={item.question_count}, "
            f"cases={item.case_count}"
        )
    print(f"Details CSV: {config.details_csv}")
    print(f"Summary CSV: {config.summary_csv}")
    print("Factors:")
    for factor in factors:
        print(
            f"- {factor.combine_strategy}, {factor.rerank_strategy}, "
            f"{factor.update_strategy}, {factor.alpha_config.slug}"
        )


def main(argv: list[str] | None = None) -> int:
    """Run the CLI."""
    config = parse_args(sys.argv[1:] if argv is None else argv)
    try:
        if config.dry_run:
            print_dry_run(config)
            return 0

        results = run_experiments(config)
        details_path = write_detailed_csv(results, config.details_csv)
        summary_path = write_summary_csv(results, config.summary_csv)
        print(f"Wrote detailed results to {details_path}")
        print(f"Wrote summary results to {summary_path}")
    except (KeyboardInterrupt, EOFError):
        print("\nExiting.", file=sys.stderr)
        return 130
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

