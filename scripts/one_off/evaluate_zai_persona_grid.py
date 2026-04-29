"""Evaluate ZAI persona retrieval across RAG strategy combinations."""

from __future__ import annotations

import argparse
import csv
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from pipeline.rag_retrieval_cli import (  # noqa: E402
    DEFAULT_EMBED_MODEL,
    DEFAULT_INDEX_NAME,
    DEFAULT_TOP_K,
    DEFAULT_USER_NAMESPACE,
    combine_vectors,
    connect_to_index,
    embed_query,
    extract_hits,
    fetch_user_vector,
    load_api_key,
    normalize_hits,
    rerank_chunks,
    search_chunks,
    update_user_vector,
)
from pipeline.scoring_functions import persona_rank_score  # noqa: E402

DEFAULT_NAMESPACE = "zai"
DEFAULT_BOTH_QUESTIONS_PATH = REPO_ROOT / "data" / "zai" / "questions" / "both.txt"
DEFAULT_CIVIL_QUESTIONS_PATH = REPO_ROOT / "data" / "zai" / "questions" / "civil.txt"
DEFAULT_MINECRAFT_QUESTIONS_PATH = (
    REPO_ROOT / "data" / "zai" / "questions" / "minecraft.txt"
)
DEFAULT_CIVIL_ANSWERS_PATH = REPO_ROOT / "data" / "zai" / "answers" / "civil.txt"
DEFAULT_MINECRAFT_ANSWERS_PATH = (
    REPO_ROOT / "data" / "zai" / "answers" / "minecraft.txt"
)
DEFAULT_OUTPUT_CSV_PATH = REPO_ROOT / "data" / "zai" / "evaluation_results.csv"

COMBINE_STRATEGIES = ("query-only", "linear-comb", "spherical-comb")
RERANK_STRATEGY = "cross-encoder"
UPDATE_STRATEGY = "moving-average"
PERSONAS = ("civil", "minecraft")


@dataclass(frozen=True)
class RetrievalRun:
    """Retrieved chunk IDs and persona score for one ask."""

    chunk_ids: list[str]
    score: float
    expected_rank: int | None


@dataclass(frozen=True)
class EvaluationResult:
    """Baseline and post-persona scores for one strategy/persona/question."""

    combine_strategy: str
    rerank_strategy: str
    update_strategy: str
    persona: str
    question_number: int
    username: str
    neutral_question: str
    expected_chunk_id: str
    priming_question_count: int
    baseline: RetrievalRun
    post_priming: RetrievalRun | None

    @property
    def delta(self) -> float | None:
        if self.post_priming is None:
            return None
        return self.post_priming.score - self.baseline.score


def parse_args(argv: list[str]) -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description=(
            "Grid-search the ZAI RAG persona retrieval pipeline and score baseline "
            "versus post-priming retrieval."
        )
    )
    parser.add_argument(
        "--index-name",
        default=DEFAULT_INDEX_NAME,
        help="Name of the Pinecone index to query",
    )
    parser.add_argument(
        "--namespace",
        default=DEFAULT_NAMESPACE,
        help="Pinecone namespace containing the ZAI civil and Minecraft chunks",
    )
    parser.add_argument(
        "--user-namespace",
        default=DEFAULT_USER_NAMESPACE,
        help="Pinecone namespace containing per-user vectors",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=DEFAULT_TOP_K,
        help="Number of chunks to retrieve per ask",
    )
    parser.add_argument(
        "--embed-model",
        default=DEFAULT_EMBED_MODEL,
        help="Pinecone embedding model for query embeddings",
    )
    parser.add_argument(
        "--both-questions-path",
        default=str(DEFAULT_BOTH_QUESTIONS_PATH),
        help="Path to neutral questions asked before and after persona priming",
    )
    parser.add_argument(
        "--civil-questions-path",
        default=str(DEFAULT_CIVIL_QUESTIONS_PATH),
        help="Path to civil persona priming questions",
    )
    parser.add_argument(
        "--minecraft-questions-path",
        default=str(DEFAULT_MINECRAFT_QUESTIONS_PATH),
        help="Path to Minecraft persona priming questions",
    )
    parser.add_argument(
        "--civil-answers-path",
        default=str(DEFAULT_CIVIL_ANSWERS_PATH),
        help="Path to expected civil chunk numbers for neutral questions",
    )
    parser.add_argument(
        "--minecraft-answers-path",
        default=str(DEFAULT_MINECRAFT_ANSWERS_PATH),
        help="Path to expected Minecraft chunk numbers for neutral questions",
    )
    parser.add_argument(
        "--output-csv",
        default=str(DEFAULT_OUTPUT_CSV_PATH),
        help="Path where per-case CSV results should be written",
    )
    return parser.parse_args(argv)


def load_questions(path: Path) -> list[str]:
    """Load non-empty question lines from a text file."""
    resolved_path = path.expanduser().resolve()
    if not resolved_path.is_file():
        raise RuntimeError(f"Question file not found: {resolved_path}")

    questions = [
        line.strip()
        for line in resolved_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if not questions:
        raise RuntimeError(f"No questions found in {resolved_path}")

    return questions


def load_expected_chunk_ids(path: Path, persona: str, expected_count: int) -> list[str]:
    """Load answer numbers and convert them to persona chunk IDs."""
    answers = load_questions(path)
    if len(answers) != expected_count:
        raise RuntimeError(
            f"Expected {expected_count} answers for {persona}, found {len(answers)} "
            f"in {path.expanduser().resolve()}"
        )

    chunk_ids: list[str] = []
    for line_number, answer in enumerate(answers, start=1):
        if not answer.isdigit():
            raise RuntimeError(
                f"Answer line {line_number} for {persona} must be a chunk number: "
                f"{answer!r}"
            )
        chunk_ids.append(f"{persona}-{int(answer)}")

    return chunk_ids


def reset_user_vector(index: Any, user_namespace: str, username: str) -> None:
    """Delete a deterministic evaluation user before reuse."""
    index.delete(ids=[username], namespace=user_namespace)


def find_expected_rank(chunk_ids: list[str], expected_chunk_id: str) -> int | None:
    """Return the 1-based rank of the expected chunk if retrieved."""
    for rank, chunk_id in enumerate(chunk_ids, start=1):
        if chunk_id == expected_chunk_id:
            return rank
    return None


def retrieve_chunks(
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
) -> list[str]:
    """Run one non-interactive retrieval through the RAG pipeline helpers."""
    query_vector = embed_query(pc, embed_model, query)
    user_vector = fetch_user_vector(index, user_namespace, username)
    if user_vector is None:
        user_vector = np.zeros_like(query_vector)

    combined_vector = combine_vectors(
        user_vector=user_vector,
        query_vector=query_vector,
        strategy=combine_strategy,
    )
    search_result = search_chunks(
        index=index,
        namespace=namespace,
        combined_vector=combined_vector,
        top_k=top_k,
    )
    chunks = normalize_hits(extract_hits(search_result))
    reranked_chunks = rerank_chunks(
        chunks=chunks,
        query=query,
        strategy=rerank_strategy,
    )
    update_user_vector(
        index=index,
        username=username,
        user_namespace=user_namespace,
        user_vector=user_vector,
        query_vector=query_vector,
        strategy=update_strategy,
    )

    return [chunk.chunk_id for chunk in reranked_chunks]


def score_retrieval(
    *,
    persona: str,
    expected_chunk_id: str,
    chunk_ids: list[str],
) -> RetrievalRun:
    """Score one retrieval result with the persona rank scoring function."""
    return RetrievalRun(
        chunk_ids=chunk_ids,
        score=persona_rank_score(
            user_type=persona,
            expected_chunk=expected_chunk_id,
            retrieved_chunks=chunk_ids,
        ),
        expected_rank=find_expected_rank(chunk_ids, expected_chunk_id),
    )


def evaluate_case(
    *,
    pc: Any,
    index: Any,
    namespace: str,
    user_namespace: str,
    embed_model: str,
    top_k: int,
    combine_strategy: str,
    persona: str,
    question_number: int,
    neutral_question: str,
    expected_chunk_id: str,
    priming_questions: list[str],
    baseline_only: bool,
) -> EvaluationResult:
    """Evaluate one neutral question before and after persona priming."""
    username = f"eval-{persona}-q{question_number:02d}-{combine_strategy}"
    reset_user_vector(index, user_namespace, username)

    baseline_chunk_ids = retrieve_chunks(
        pc=pc,
        index=index,
        username=username,
        query=neutral_question,
        namespace=namespace,
        user_namespace=user_namespace,
        embed_model=embed_model,
        top_k=top_k,
        combine_strategy=combine_strategy,
        rerank_strategy=RERANK_STRATEGY,
        update_strategy="none",
    )
    baseline = score_retrieval(
        persona=persona,
        expected_chunk_id=expected_chunk_id,
        chunk_ids=baseline_chunk_ids,
    )

    if baseline_only:
        return EvaluationResult(
            combine_strategy=combine_strategy,
            rerank_strategy=RERANK_STRATEGY,
            update_strategy=UPDATE_STRATEGY,
            persona=persona,
            question_number=question_number,
            username=username,
            neutral_question=neutral_question,
            expected_chunk_id=expected_chunk_id,
            priming_question_count=0,
            baseline=baseline,
            post_priming=None,
        )

    for priming_question in priming_questions:
        retrieve_chunks(
            pc=pc,
            index=index,
            username=username,
            query=priming_question,
            namespace=namespace,
            user_namespace=user_namespace,
            embed_model=embed_model,
            top_k=top_k,
            combine_strategy=combine_strategy,
            rerank_strategy=RERANK_STRATEGY,
            update_strategy=UPDATE_STRATEGY,
        )

    post_chunk_ids = retrieve_chunks(
        pc=pc,
        index=index,
        username=username,
        query=neutral_question,
        namespace=namespace,
        user_namespace=user_namespace,
        embed_model=embed_model,
        top_k=top_k,
        combine_strategy=combine_strategy,
        rerank_strategy=RERANK_STRATEGY,
        update_strategy="none",
    )
    post_priming = score_retrieval(
        persona=persona,
        expected_chunk_id=expected_chunk_id,
        chunk_ids=post_chunk_ids,
    )

    return EvaluationResult(
        combine_strategy=combine_strategy,
        rerank_strategy=RERANK_STRATEGY,
        update_strategy=UPDATE_STRATEGY,
        persona=persona,
        question_number=question_number,
        username=username,
        neutral_question=neutral_question,
        expected_chunk_id=expected_chunk_id,
        priming_question_count=len(priming_questions),
        baseline=baseline,
        post_priming=post_priming,
    )


def evaluate_grid(args: argparse.Namespace) -> list[EvaluationResult]:
    """Run the full strategy/persona/question evaluation grid."""
    neutral_questions = load_questions(Path(args.both_questions_path))
    persona_questions = {
        "civil": load_questions(Path(args.civil_questions_path)),
        "minecraft": load_questions(Path(args.minecraft_questions_path)),
    }
    expected_chunk_ids = {
        "civil": load_expected_chunk_ids(
            Path(args.civil_answers_path), "civil", len(neutral_questions)
        ),
        "minecraft": load_expected_chunk_ids(
            Path(args.minecraft_answers_path), "minecraft", len(neutral_questions)
        ),
    }

    api_key = load_api_key()
    pc, index = connect_to_index(api_key, args.index_name)
    results: list[EvaluationResult] = []

    for combine_strategy in COMBINE_STRATEGIES:
        for question_number, neutral_question in enumerate(neutral_questions, start=1):
            for persona in PERSONAS:
                result = evaluate_case(
                    pc=pc,
                    index=index,
                    namespace=args.namespace,
                    user_namespace=args.user_namespace,
                    embed_model=args.embed_model,
                    top_k=args.top_k,
                    combine_strategy=combine_strategy,
                    persona=persona,
                    question_number=question_number,
                    neutral_question=neutral_question,
                    expected_chunk_id=expected_chunk_ids[persona][
                        question_number - 1
                    ],
                    priming_questions=persona_questions[persona],
                    baseline_only=combine_strategy == "query-only",
                )
                results.append(result)
                print_case_progress(result)

    return results


def rank_display(rank: int | None) -> str:
    """Format an optional rank for reports."""
    return str(rank) if rank is not None else "not retrieved"


def score_display(run: RetrievalRun | None) -> str:
    """Format an optional retrieval score for reports."""
    return f"{run.score:.4f}" if run is not None else "n/a"


def delta_display(delta: float | None) -> str:
    """Format an optional delta for reports."""
    return f"{delta:+.4f}" if delta is not None else "n/a"


def print_case_progress(result: EvaluationResult) -> None:
    """Print a compact row as each case completes."""
    print(
        f"{result.combine_strategy:14} {result.persona:9} "
        f"q{result.question_number:02d} "
        f"baseline={result.baseline.score:.4f} "
        f"post={score_display(result.post_priming)} "
        f"delta={delta_display(result.delta)}"
    )


def write_csv(results: list[EvaluationResult], output_path: Path) -> None:
    """Write per-case evaluation rows to CSV."""
    resolved_path = output_path.expanduser().resolve()
    resolved_path.parent.mkdir(parents=True, exist_ok=True)

    with resolved_path.open("w", encoding="utf-8", newline="") as csv_file:
        writer = csv.DictWriter(
            csv_file,
            fieldnames=[
                "combine_strategy",
                "rerank_strategy",
                "update_strategy",
                "persona",
                "question_number",
                "username",
                "expected_chunk_id",
                "neutral_question",
                "priming_question_count",
                "baseline_score",
                "post_priming_score",
                "delta",
                "baseline_expected_rank",
                "post_priming_expected_rank",
                "baseline_chunk_ids",
                "post_priming_chunk_ids",
            ],
        )
        writer.writeheader()
        for result in results:
            writer.writerow(
                {
                    "combine_strategy": result.combine_strategy,
                    "rerank_strategy": result.rerank_strategy,
                    "update_strategy": result.update_strategy,
                    "persona": result.persona,
                    "question_number": result.question_number,
                    "username": result.username,
                    "expected_chunk_id": result.expected_chunk_id,
                    "neutral_question": result.neutral_question,
                    "priming_question_count": result.priming_question_count,
                    "baseline_score": f"{result.baseline.score:.6f}",
                    "post_priming_score": (
                        f"{result.post_priming.score:.6f}"
                        if result.post_priming is not None
                        else ""
                    ),
                    "delta": (
                        f"{result.delta:.6f}" if result.delta is not None else ""
                    ),
                    "baseline_expected_rank": rank_display(
                        result.baseline.expected_rank
                    ),
                    "post_priming_expected_rank": (
                        rank_display(result.post_priming.expected_rank)
                        if result.post_priming is not None
                        else ""
                    ),
                    "baseline_chunk_ids": "|".join(result.baseline.chunk_ids),
                    "post_priming_chunk_ids": (
                        "|".join(result.post_priming.chunk_ids)
                        if result.post_priming is not None
                        else ""
                    ),
                }
            )


def mean(values: list[float]) -> float:
    """Compute a simple mean for summary reporting."""
    return sum(values) / len(values) if values else 0.0


def print_report(results: list[EvaluationResult], output_csv: Path) -> None:
    """Print leaderboard and per-question summaries."""
    if not results:
        print("No evaluation results produced.")
        return

    grouped_results: dict[tuple[str, str], list[EvaluationResult]] = defaultdict(list)
    for result in results:
        grouped_results[(result.combine_strategy, result.persona)].append(result)

    summaries = []
    for (combine_strategy, persona), group in grouped_results.items():
        post_group = [result for result in group if result.post_priming is not None]
        baseline_mean = mean([result.baseline.score for result in group])
        post_mean = mean([result.post_priming.score for result in post_group])
        deltas = [result.delta for result in post_group if result.delta is not None]
        delta_mean = mean(deltas) if deltas else None
        wins = sum(
            1 for result in post_group if result.delta is not None and result.delta > 0
        )
        summaries.append(
            (
                delta_mean if delta_mean is not None else float("-inf"),
                combine_strategy,
                persona,
                baseline_mean,
                post_mean if post_group else None,
                delta_mean,
                wins,
                len(group),
                len(post_group),
            )
        )

    print()
    print("ZAI Persona Strategy Leaderboard")
    print("================================")
    for (
        _sort_delta,
        combine_strategy,
        persona,
        baseline_mean,
        post_mean,
        delta_mean,
        wins,
        case_count,
        post_case_count,
    ) in sorted(summaries, reverse=True):
        post_mean_display = f"{post_mean:.4f}" if post_mean is not None else "n/a"
        print(
            f"{combine_strategy:14} {persona:9} "
            f"baseline_mean={baseline_mean:.4f} "
            f"post_mean={post_mean_display} "
            f"mean_delta={delta_display(delta_mean)} "
            f"wins={wins}/{post_case_count} "
            f"cases={case_count}"
        )

    print()
    print("Per-question Results")
    print("====================")
    for result in results:
        print(
            f"{result.combine_strategy:14} {result.persona:9} "
            f"q{result.question_number:02d} "
            f"expected={result.expected_chunk_id:12} "
            f"rank={rank_display(result.baseline.expected_rank)}"
            f"->{rank_display(result.post_priming.expected_rank) if result.post_priming is not None else 'n/a'} "
            f"score={result.baseline.score:.4f}"
            f"->{score_display(result.post_priming)} "
            f"delta={delta_display(result.delta)}"
        )

    print()
    print(f"Wrote CSV results to {output_csv.expanduser().resolve()}")


def main(argv: list[str] | None = None) -> int:
    """Run the ZAI persona grid evaluation."""
    args = parse_args(sys.argv[1:] if argv is None else argv)

    try:
        results = evaluate_grid(args)
        output_csv = Path(args.output_csv)
        write_csv(results, output_csv)
        print_report(results, output_csv)
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
