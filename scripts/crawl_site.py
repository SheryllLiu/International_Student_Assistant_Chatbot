"""Thin wrapper around ``f1a crawl`` so the crawl can be run without the CLI."""

from __future__ import annotations

import argparse
from pathlib import Path

from f1_immigration_assistant.config import DEFAULT_CONFIG
from f1_immigration_assistant.logging_config import setup_logging
from f1_immigration_assistant.pipeline import Pipeline


def main() -> None:
    """Crawl the allowlisted Georgetown OGS pages."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=DEFAULT_CONFIG.raw_dir)
    args = parser.parse_args()

    setup_logging()
    saved = Pipeline().crawl(out_dir=args.out)
    print(f"saved {len(saved)} pages to {args.out}")


if __name__ == "__main__":
    main()
