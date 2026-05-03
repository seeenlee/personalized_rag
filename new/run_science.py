"""Run the standalone experiment grid for the science dataset."""

from __future__ import annotations

import sys

from .run_experiments import main as run_main


def main(argv: list[str] | None = None) -> int:
    args = sys.argv[1:] if argv is None else argv
    return run_main(["--datasets", "science", *args])


if __name__ == "__main__":
    raise SystemExit(main())

