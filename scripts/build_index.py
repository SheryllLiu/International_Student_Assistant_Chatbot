"""Thin wrapper around ``f1a build-index``."""

from __future__ import annotations

import argparse
from pathlib import Path

from f1_immigration_assistant.config import DEFAULT_CONFIG
from f1_immigration_assistant.logging_config import setup_logging
from f1_immigration_assistant.pipeline import Pipeline


def main() -> None:
    """Build a BM25 index from previously crawled HTML."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw", type=Path, default=DEFAULT_CONFIG.raw_dir)
    parser.add_argument("--out", type=Path, default=DEFAULT_CONFIG.index_dir)
    args = parser.parse_args()

    setup_logging()
    chunks = Pipeline().build_index(raw_dir=args.raw, index_dir=args.out)
    print(f"indexed {len(chunks)} chunks -> {args.out}")


if __name__ == "__main__":
    main()
