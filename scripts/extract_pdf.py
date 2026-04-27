"""Extract text from a PDF into the repo's data/pdf directory."""

import argparse
import sys
from pathlib import Path


def parse_args(argv: list[str]) -> argparse.Namespace:
    """Parse command line arguments for PDF extraction."""
    parser = argparse.ArgumentParser(
        description="Extract text from a PDF and save it under data/pdf."
    )
    parser.add_argument("input_file", help="Path to the input PDF file")
    parser.add_argument(
        "output_filename",
        help="Name of the output text file to create in data/pdf",
    )
    return parser.parse_args(argv)


def resolve_output_path(output_filename: str) -> Path:
    """Build the repo-local output path and reject nested paths."""
    output_name = Path(output_filename)
    if output_name.name != output_filename:
        raise ValueError("Output file must be a file name, not a path.")

    repo_root = Path(__file__).resolve().parents[1]
    output_dir = repo_root / "data" / "pdf"
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir / output_name


def main(argv: list[str] | None = None) -> int:
    """Run the PDF extraction workflow."""
    args = parse_args(sys.argv[1:] if argv is None else argv)

    input_path = Path(args.input_file).expanduser()

    try:
        import pymupdf4llm

        text = pymupdf4llm.to_text(str(input_path))
        output_path = resolve_output_path(args.output_filename)
        output_path.write_text(text, encoding="utf-8")
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    print(f"Wrote extracted text to {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())