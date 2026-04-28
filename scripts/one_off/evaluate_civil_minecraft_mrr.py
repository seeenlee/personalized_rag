"""Evaluate shared-namespace retrieval for the civil and Minecraft benchmark."""

import argparse
import os
import sys
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from pinecone import Pinecone

REPO_ROOT = Path(__file__).resolve().parents[2]
from ..insert_text_file import (
    METADATA_TEXT_FIELD,
    TEXT_FIELD_NAME,
    split_paragraphs,
)

PINECONE_ENV_VAR_NAME = "PINECONE_API_KEY"
DEFAULT_CIVIL_PATH = REPO_ROOT / "data" / "civil" / "civil.txt"
DEFAULT_MINECRAFT_PATH = REPO_ROOT / "data" / "civil" / "minecraft.txt"
DEFAULT_NOISE_PATH = REPO_ROOT / "data" / "civil" / "noise.txt"
DEFAULT_TOP_K = 5


@dataclass(frozen=True)
class BenchmarkCase:
    """Expected retrieval target for a single benchmark query."""

    user_label: str
    question: str
    expected_chunk_id: str
    expected_section: str


@dataclass
class RetrievedChunk:
    """Normalized retrieval result for reporting and evaluation."""

    chunk_id: str
    text: str
    score: float | None


@dataclass
class CaseResult:
    """Evaluation result for a single benchmark query."""

    benchmark_case: BenchmarkCase
    expected_rank: int | None
    reciprocal_rank: float
    retrieved_chunks: list[RetrievedChunk]


BENCHMARK_CASES = [
    BenchmarkCase(
        user_label="Civil Engineer",
        question="How to design a structurally sound house?",
        expected_chunk_id="civil-1",
        expected_section="Section 1: Residential Structural Integrity",
    ),
    BenchmarkCase(
        user_label="Minecraft Player",
        question="How to design a structurally sound house?",
        expected_chunk_id="minecraft-1",
        expected_section="Section 1: Designing to protect against mobs",
    ),
    BenchmarkCase(
        user_label="Civil Engineer",
        question="How to clear bedrock when building?",
        expected_chunk_id="civil-2",
        expected_section="Section 2: Blasting & hydraulic hammers",
    ),
    BenchmarkCase(
        user_label="Minecraft Player",
        question="How to clear bedrock when building?",
        expected_chunk_id="minecraft-2",
        expected_section="Section 2: Indestructible/Creative Mode",
    ),
    BenchmarkCase(
        user_label="Civil Engineer",
        question="How to stop gravel from falling when digging?",
        expected_chunk_id="civil-3",
        expected_section="Section 3: Shoring & steel casings",
    ),
    BenchmarkCase(
        user_label="Minecraft Player",
        question="How to stop gravel from falling when digging?",
        expected_chunk_id="minecraft-3",
        expected_section="Section 3: Using torches/solid blocks",
    ),
    BenchmarkCase(
        user_label="Civil Engineer",
        question="How to set up automated lighting using redstone?",
        expected_chunk_id="civil-4",
        expected_section="Section 4: Low-voltage masonry conduits",
    ),
    BenchmarkCase(
        user_label="Minecraft Player",
        question="How to set up automated lighting using redstone?",
        expected_chunk_id="minecraft-4",
        expected_section="Section 4: Redstone dust & sensors",
    ),
    BenchmarkCase(
        user_label="Civil Engineer",
        question="Challenges of building in a desert biome?",
        expected_chunk_id="civil-5",
        expected_section="Section 5: Thermal expansion & abrasion",
    ),
    BenchmarkCase(
        user_label="Minecraft Player",
        question="Challenges of building in a desert biome?",
        expected_chunk_id="minecraft-5",
        expected_section="Section 5: Sandstone & resource scarcity",
    ),
    BenchmarkCase(
        user_label="Civil Engineer",
        question="Building at high altitudes?",
        expected_chunk_id="civil-5",
        expected_section="Section 5: Freeze-thaw & frost lines",
    ),
    BenchmarkCase(
        user_label="Minecraft Player",
        question="Building at high altitudes?",
        expected_chunk_id="minecraft-5",
        expected_section="Section 5: Extreme Hills & build limits",
    ),
    BenchmarkCase(
        user_label="Civil Engineer",
        question="Building a foundation on sand?",
        expected_chunk_id="civil-3",
        expected_section="Section 3: Slurry walls & moisture",
    ),
    BenchmarkCase(
        user_label="Minecraft Player",
        question="Building a foundation on sand?",
        expected_chunk_id="minecraft-3",
        expected_section="Section 3: Swapping blocks/updates",
    ),
    # Topic: Material Creep vs. Mob Creep
    BenchmarkCase(
        user_label="Civil Engineer",
        question="How to account for long-term creep in a structure?",
        expected_chunk_id="civil-9",
        expected_section="Section 9: Material Creep & Deformation",
    ),
    BenchmarkCase(
        user_label="Minecraft Player",
        question="How to account for long-term creep in a structure?",
        expected_chunk_id="minecraft-9",
        expected_section="Section 9: Creeper Blast Resistance & Obsidian",
    ),

    # Topic: Hydraulic Pistons
    BenchmarkCase(
        user_label="Civil Engineer",
        question="Best way to move heavy loads vertically using pistons?",
        expected_chunk_id="civil-6",
        expected_section="Section 6: Hydraulic Jacking & Superstructures",
    ),
    BenchmarkCase(
        user_label="Minecraft Player",
        question="Best way to move heavy loads vertically using pistons?",
        expected_chunk_id="minecraft-6",
        expected_section="Section 6: Pistons & Sticky Pistons",
    ),

    # Topic: Trapdoors & Pathfinding
    BenchmarkCase(
        user_label="Civil Engineer",
        question="How to use trapdoors to control movement?",
        expected_chunk_id="civil-8",
        expected_section="Section 8: Egress & Access Hatches",
    ),
    BenchmarkCase(
        user_label="Minecraft Player",
        question="How to use trapdoors to control movement?",
        expected_chunk_id="minecraft-8",
        expected_section="Section 8: Mob Pathfinding & AI trickery",
    ),

    # Topic: Bookshelves & Load/Power
    BenchmarkCase(
        user_label="Civil Engineer",
        question="Safety standards for library bookshelves?",
        expected_chunk_id="civil-14",
        expected_section="Section 14: Live Loads & Seismic Bracing",
    ),
    BenchmarkCase(
        user_label="Minecraft Player",
        question="Safety standards for library bookshelves?",
        expected_chunk_id="minecraft-14",
        expected_section="Section 14: Enchantment Spacing & Air Gaps",
    ),

    # Topic: Tunnel/Nether Portals
    BenchmarkCase(
        user_label="Civil Engineer",
        question="How to stabilize a portal entrance?",
        expected_chunk_id="civil-12",
        expected_section="Section 12: Headwalls & Wingwalls",
    ),
    BenchmarkCase(
        user_label="Minecraft Player",
        question="How to stabilize a portal entrance?",
        expected_chunk_id="minecraft-12",
        expected_section="Section 12: Obsidian frames & Coordinate Math",
    ),

    # Topic: Aesthetic Lighting
    BenchmarkCase(
        user_label="Civil Engineer",
        question="How to hide light sources for better aesthetics?",
        expected_chunk_id="civil-13",
        expected_section="Section 13: Cove Lighting & Foot-candles",
    ),
    BenchmarkCase(
        user_label="Minecraft Player",
        question="How to hide light sources for better aesthetics?",
        expected_chunk_id="minecraft-13",
        expected_section="Section 13: Carpets & Block Light",
    ),

    # Topic: Water Management
    BenchmarkCase(
        user_label="Civil Engineer",
        question="How to manage water displacement in a foundation?",
        expected_chunk_id="civil-7",
        expected_section="Section 7: French drains & Sump pumps",
    ),
    BenchmarkCase(
        user_label="Minecraft Player",
        question="How to manage water displacement in a foundation?",
        expected_chunk_id="minecraft-7",
        expected_section="Section 7: Source blocks & Sponges",
    ),

    # Topic: Automated Delivery/Rail
    BenchmarkCase(
        user_label="Civil Engineer",
        question="How to automate item delivery?",
        expected_chunk_id="civil-10",
        expected_section="Section 10: Heavy Rail & Grade Management",
    ),
    BenchmarkCase(
        user_label="Minecraft Player",
        question="How to automate item delivery?",
        expected_chunk_id="minecraft-10",
        expected_section="Section 10: Minecarts, Chests, & Hoppers",
    ),

    # Topic: Urban/Efficient Farming
    BenchmarkCase(
        user_label="Civil Engineer",
        question="Improving growth in a rooftop or urban garden?",
        expected_chunk_id="civil-11",
        expected_section="Section 11: Soil Remediation & Load Analysis",
    ),
    BenchmarkCase(
        user_label="Minecraft Player",
        question="Improving growth in a rooftop or urban garden?",
        expected_chunk_id="minecraft-11",
        expected_section="Section 11: Bone Meal & Hydration Squares",
    ),

    # Topic: Lava/Thermal Hazards
    BenchmarkCase(
        user_label="Civil Engineer",
        question="Safely managing thermal hazards and molten materials?",
        expected_chunk_id="civil-6",
        expected_section="Section 6: Thermal Hazards & PPE",
    ),
    BenchmarkCase(
        user_label="Minecraft Player",
        question="Safely managing thermal hazards and molten materials?",
        expected_chunk_id="minecraft-6",
        expected_section="Section 6: Water buckets & Fire Resistance",
    ),
]


def load_api_key() -> str:
    """Load the Pinecone API key from the repo .env or environment."""
    load_dotenv(REPO_ROOT / ".env")

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
        description=(
            "Evaluate shared-namespace Pinecone retrieval for the civil and "
            "Minecraft benchmark using mean reciprocal rank."
        )
    )
    parser.add_argument("index_name", help="Name of the Pinecone index to query")
    parser.add_argument("namespace", help="Shared namespace in the Pinecone index")
    parser.add_argument(
        "--civil-path",
        default=str(DEFAULT_CIVIL_PATH),
        help="Path to the civil source text file",
    )
    parser.add_argument(
        "--minecraft-path",
        default=str(DEFAULT_MINECRAFT_PATH),
        help="Path to the Minecraft source text file",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=DEFAULT_TOP_K,
        help="Number of Pinecone chunks to retrieve per benchmark query",
    )
    return parser.parse_args(argv)


def ensure_file_exists(path: Path) -> Path:
    """Resolve and validate that a file exists."""
    resolved = path.resolve()
    if not resolved.is_file():
        raise RuntimeError(f"Source file not found: {resolved}")
    return resolved


def build_chunk_id_set(article_path: Path) -> set[str]:
    """Build the expected chunk IDs for a paragraph-chunked source file."""
    article_text = article_path.read_text(encoding="utf-8")
    paragraphs = split_paragraphs(article_text)
    if not paragraphs:
        raise RuntimeError(f"No paragraphs found in {article_path}.")

    return {
        f"{article_path.stem}-{paragraph_number}"
        for paragraph_number in range(1, len(paragraphs) + 1)
    }


def validate_benchmark_cases(civil_path: Path, minecraft_path: Path) -> None:
    """Confirm the expected benchmark chunk IDs exist in the local source files."""
    available_chunk_ids = build_chunk_id_set(civil_path) | build_chunk_id_set(minecraft_path)
    missing_chunk_ids = sorted(
        {
            benchmark_case.expected_chunk_id
            for benchmark_case in BENCHMARK_CASES
            if benchmark_case.expected_chunk_id not in available_chunk_ids
        }
    )
    if missing_chunk_ids:
        raise RuntimeError(
            "Benchmark references chunk IDs that are not produced by the local source "
            f"files: {', '.join(missing_chunk_ids)}"
        )


def connect_to_index(api_key: str, index_name: str) -> Any:
    """Connect to an existing Pinecone index."""
    pc = Pinecone(api_key=api_key)
    if not pc.has_index(index_name):
        raise RuntimeError(f"Pinecone index '{index_name}' does not exist.")

    index_host = pc.describe_index(name=index_name).host
    return pc.Index(host=index_host)


def search_pinecone(index: Any, namespace: str, question: str, top_k: int) -> Any:
    """Run a text search against an integrated-embedding Pinecone index."""
    query = {
        "inputs": {"text": question},
        "top_k": top_k,
    }
    fields = [TEXT_FIELD_NAME, METADATA_TEXT_FIELD]

    if hasattr(index, "search"):
        return index.search(namespace=namespace, query=query, fields=fields)

    if hasattr(index, "search_records"):
        return index.search_records(namespace=namespace, query=query, fields=fields)

    raise RuntimeError(
        "The installed Pinecone client does not expose a supported search method."
    )


def _safe_get(obj: Any, key: str) -> Any:
    """Read a key from either a mapping or an object attribute."""
    if isinstance(obj, Mapping):
        return obj.get(key)
    return getattr(obj, key, None)


def _as_sequence(value: Any) -> list[Any]:
    """Normalize a list-like value while ignoring strings."""
    if value is None or isinstance(value, (str, bytes)):
        return []
    if isinstance(value, Sequence):
        return list(value)
    return []


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

        hits = _as_sequence(_safe_get(current, "hits"))
        if hits:
            return hits

        matches = _as_sequence(_safe_get(current, "matches"))
        if matches:
            return matches

        for child_key in ("result", "results", "data"):
            child = _safe_get(current, child_key)
            if child is not None:
                pending.append(child)

    return []


def normalize_hits(raw_hits: list[Any]) -> list[RetrievedChunk]:
    """Convert Pinecone hits into a stable structure."""
    normalized: list[RetrievedChunk] = []

    for raw_hit in raw_hits:
        field_sources = [
            _safe_get(raw_hit, "fields"),
            _safe_get(raw_hit, "metadata"),
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

        chunk_id = _safe_get(raw_hit, "_id") or _safe_get(raw_hit, "id") or "unknown"
        score = _safe_get(raw_hit, "_score") or _safe_get(raw_hit, "score")
        normalized.append(
            RetrievedChunk(
                chunk_id=str(chunk_id),
                text=str(text).strip(),
                score=float(score) if score is not None else None,
            )
        )

    return normalized


def find_expected_rank(
    retrieved_chunks: list[RetrievedChunk], expected_chunk_id: str
) -> int | None:
    """Return the 1-based rank of the expected chunk if retrieved."""
    for rank, chunk in enumerate(retrieved_chunks, start=1):
        if chunk.chunk_id == expected_chunk_id:
            return rank
    return None


def reciprocal_rank(rank: int | None) -> float:
    """Convert a rank into reciprocal rank."""
    if rank is None:
        return 0.0
    return 1.0 / rank


def evaluate_case(index: Any, namespace: str, benchmark_case: BenchmarkCase, top_k: int) -> CaseResult:
    """Run retrieval for one benchmark query and score it."""
    raw_search_result = search_pinecone(
        index=index,
        namespace=namespace,
        question=benchmark_case.question,
        top_k=top_k,
    )
    retrieved_chunks = normalize_hits(extract_hits(raw_search_result))
    expected_rank = find_expected_rank(
        retrieved_chunks, benchmark_case.expected_chunk_id
    )
    return CaseResult(
        benchmark_case=benchmark_case,
        expected_rank=expected_rank,
        reciprocal_rank=reciprocal_rank(expected_rank),
        retrieved_chunks=retrieved_chunks,
    )


def compute_mean_reciprocal_rank(case_results: list[CaseResult]) -> float:
    """Compute MRR across the full benchmark set."""
    if not case_results:
        return 0.0
    return sum(result.reciprocal_rank for result in case_results) / len(case_results)


def print_report(case_results: list[CaseResult]) -> None:
    """Print an inspection-friendly benchmark report."""
    overall_mrr = compute_mean_reciprocal_rank(case_results)

    print("Civil/Minecraft Shared-Namespace Evaluation")
    print("===========================================")
    print(f"Benchmark cases: {len(case_results)}")
    print(f"Overall MRR: {overall_mrr:.4f}")
    print()
    print("Per-case results")
    print("----------------")
    for index, result in enumerate(case_results, start=1):
        top_chunk_ids = ", ".join(
            chunk.chunk_id for chunk in result.retrieved_chunks[:5]
        ) or "no hits"
        rank_display = result.expected_rank if result.expected_rank is not None else "not retrieved"
        print(
            f"{index}. [{result.benchmark_case.user_label}] "
            f"{result.benchmark_case.question}"
        )
        print(f"   expected: {result.benchmark_case.expected_chunk_id}")
        print(f"   section: {result.benchmark_case.expected_section}")
        print(f"   rank: {rank_display}")
        print(f"   reciprocal rank: {result.reciprocal_rank:.4f}")
        print(f"   top hits: {top_chunk_ids}")

    print()
    print("Grouped summary")
    print("---------------")
    user_labels = sorted({result.benchmark_case.user_label for result in case_results})
    for user_label in user_labels:
        matching_results = [
            result
            for result in case_results
            if result.benchmark_case.user_label == user_label
        ]
        user_mrr = compute_mean_reciprocal_rank(matching_results)
        retrieved_count = sum(
            1 for result in matching_results if result.expected_rank is not None
        )
        print(
            f"{user_label}: MRR={user_mrr:.4f} | "
            f"retrieved={retrieved_count}/{len(matching_results)}"
        )


def main(argv: list[str] | None = None) -> int:
    """Run the benchmark evaluation workflow."""
    args = parse_args(sys.argv[1:] if argv is None else argv)

    try:
        civil_path = ensure_file_exists(Path(args.civil_path))
        minecraft_path = ensure_file_exists(Path(args.minecraft_path))
        validate_benchmark_cases(civil_path, minecraft_path)
        pinecone_api_key = load_api_key()
        index = connect_to_index(pinecone_api_key, args.index_name)
        case_results = [
            evaluate_case(index, args.namespace, benchmark_case, args.top_k)
            for benchmark_case in BENCHMARK_CASES
        ]
        print_report(case_results)
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
