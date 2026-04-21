"""Thin wrapper around ``f1a evaluate``."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from f1_immigration_assistant.config import DEFAULT_CONFIG
from f1_immigration_assistant.logging_config import setup_logging
from f1_immigration_assistant.pipeline import Pipeline


def main() -> None:
    """Evaluate BM25 retrieval on the judged query set."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--index", type=Path, default=DEFAULT_CONFIG.index_dir)
    parser.add_argument("--queries", type=Path, default=Path("data/eval/queries.json"))
    parser.add_argument("-k", type=int, default=DEFAULT_CONFIG.retrieval.top_k)
    args = parser.parse_args()

    setup_logging()
    report = Pipeline().load_index(args.index).evaluate(args.queries, k=args.k)
    print(
        json.dumps(
            {
                "k": report.k,
                "precision_at_k": round(report.precision_at_k, 4),
                "recall_at_k": round(report.recall_at_k, 4),
                "mean_ap": round(report.mean_ap, 4),
                "n_queries": len(report.per_query),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
