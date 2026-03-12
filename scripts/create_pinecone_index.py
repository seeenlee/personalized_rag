"""Create the Pinecone index used by this project."""

import argparse
import os
import sys
from pathlib import Path

from dotenv import load_dotenv
from pinecone import IndexEmbed, Pinecone

CLOUD = "aws"
REGION = "us-east-1"
EMBED_MODEL = "llama-text-embed-v2"
FIELD_MAP = {"text": "chunk_text"}
ENV_VAR_NAME = "PINECONE_API_KEY"


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
    """Parse command line arguments for index creation."""
    parser = argparse.ArgumentParser(
        description="Create a Pinecone index for this project."
    )
    parser.add_argument("index_name", help="Name of the Pinecone index to create")
    return parser.parse_args(argv)


def create_index(api_key: str, index_name: str) -> None:
    """Create the configured Pinecone index if it does not already exist."""
    pc = Pinecone(api_key=api_key)

    if pc.has_index(index_name):
        print(f"Pinecone index '{index_name}' already exists.")
        return

    pc.create_index_for_model(
        name=index_name,
        cloud=CLOUD,
        region=REGION,
        embed=IndexEmbed(
            model=EMBED_MODEL,
            field_map=FIELD_MAP,
            metric="cosine",
        ),
    )
    print(
        f"Created Pinecone index '{index_name}' with model '{EMBED_MODEL}' in "
        f"{CLOUD}/{REGION}."
    )


def main(argv: list[str] | None = None) -> int:
    """Run the index creation workflow."""
    args = parse_args(sys.argv[1:] if argv is None else argv)

    try:
        api_key = load_api_key()
        create_index(api_key, args.index_name)
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
