"""Insert paragraph records from a saved Wikipedia text file into Pinecone."""

import argparse
import os
import re
import sys
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from pinecone import Pinecone

ENV_VAR_NAME = "PINECONE_API_KEY"
TEXT_FIELD_NAME = "chunk_text"
METADATA_TEXT_FIELD = "text"
UPSERT_BATCH_SIZE = 96


def load_api_key() -> str:
    """Load the Pinecone API key from the repo's .env file or environment."""
    repo_root = Path(__file__).resolve().parents[1]
    load_dotenv(repo_root / ".env")

    api_key = os.getenv(ENV_VAR_NAME)
    if not api_key:
        raise RuntimeError(
            f"Missing {ENV_VAR_NAME}. Add it to your environment or the repo .env file."
        )

    return api_key


def parse_args(argv: list[str]) -> argparse.Namespace:
    """Parse command line arguments for Wikipedia paragraph insertion."""
    parser = argparse.ArgumentParser(
        description=(
            "Insert paragraph records from a Wikipedia text file into a Pinecone index."
        )
    )
    parser.add_argument("index_name", help="Name of the Pinecone index to insert into")
    parser.add_argument("namespace", help="Namespace in the Pinecone index to insert into")
    parser.add_argument(
        "txt_file_path",
        help="Path to a .txt file under data/wikipedia",
    )
    return parser.parse_args(argv)


def resolve_wikipedia_path(raw_path: str) -> Path:
    """Resolve and validate that the input file lives under data/wikipedia."""
    repo_root = Path(__file__).resolve().parents[1]
    wikipedia_dir = (repo_root / "data" / "wikipedia").resolve()
    candidate = Path(raw_path)
    candidate = candidate if candidate.is_absolute() else repo_root / candidate

    try:
        resolved_path = candidate.resolve(strict=True)
    except FileNotFoundError as exc:
        raise RuntimeError(f"File not found: {candidate}") from exc

    if resolved_path.suffix.lower() != ".txt":
        raise RuntimeError("Input file must be a .txt file.")

    try:
        resolved_path.relative_to(wikipedia_dir)
    except ValueError as exc:
        raise RuntimeError(
            f"Input file must live under {wikipedia_dir}."
        ) from exc

    return resolved_path


def split_paragraphs(article_text: str) -> list[str]:
    """Split article text into paragraphs using blank lines as separators."""
    normalized_text = article_text.replace("\r\n", "\n").replace("\r", "\n").strip()
    if not normalized_text:
        return []

    paragraphs = [
        paragraph.strip()
        for paragraph in re.split(r"\n\s*\n+", normalized_text)
        if paragraph.strip()
    ]
    return paragraphs


def build_records(file_path: Path, paragraphs: list[str]) -> list[dict[str, Any]]:
    """Build Pinecone records for each paragraph."""
    records: list[dict[str, Any]] = []
    for paragraph_number, paragraph in enumerate(paragraphs, start=1):
        records.append(
            {
                "_id": f"{file_path.stem}-{paragraph_number}",
                TEXT_FIELD_NAME: paragraph,
                METADATA_TEXT_FIELD: paragraph,
            }
        )
    return records


def batched(records: list[dict[str, Any]], batch_size: int) -> list[list[dict[str, Any]]]:
    """Group records into batches for upsert requests."""
    return [
        records[start_index : start_index + batch_size]
        for start_index in range(0, len(records), batch_size)
    ]


def insert_paragraphs(
    api_key: str, index_name: str, namespace: str, file_path: Path
) -> int:
    """Insert paragraph records into the target Pinecone index."""
    article_text = file_path.read_text(encoding="utf-8")
    paragraphs = split_paragraphs(article_text)
    if not paragraphs:
        raise RuntimeError(f"No paragraphs found in {file_path}.")

    pc = Pinecone(api_key=api_key)
    if not pc.has_index(index_name):
        raise RuntimeError(f"Pinecone index '{index_name}' does not exist.")

    index_host = pc.describe_index(name=index_name).host
    index = pc.Index(host=index_host)
    records = build_records(file_path, paragraphs)

    for batch in batched(records, UPSERT_BATCH_SIZE):
        index.upsert_records(namespace, batch)

    return len(records)


def main(argv: list[str] | None = None) -> int:
    """Run the command-line interface."""
    args = parse_args(sys.argv[1:] if argv is None else argv)

    try:
        api_key = load_api_key()
        file_path = resolve_wikipedia_path(args.txt_file_path)
        inserted_count = insert_paragraphs(
            api_key, args.index_name, args.namespace, file_path
        )
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    print(
        f"Inserted {inserted_count} paragraph records from {file_path.name} into "
        f"Pinecone index '{args.index_name}' namespace '{args.namespace}'."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
