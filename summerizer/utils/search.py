"""Run simple searches over the saved BM25 index.

Think of this file as the easiest way to *use* the retriever from the command
line. It lets you type a question, inspect the top results, and optionally
save those results as JSON.
"""

from __future__ import annotations

import argparse
from pathlib import Path


def load_logger():
    """Import ``write_log`` in a way that works for both package and script runs."""

    try:
        from summerizer.utils.logger import write_log
    except ModuleNotFoundError:
        from logger import write_log  # type: ignore
    return write_log


def load_retriever_parts():
    """Import retriever helpers in a way that works for both package and script runs."""

    try:
        from summerizer.utils.retriever import (
            INDEX_PATH,
            RetrievalResult,
            search_saved_index,
            write_search_results_json,
        )
    except ModuleNotFoundError:
        from retriever import (  # type: ignore
            INDEX_PATH,
            RetrievalResult,
            search_saved_index,
            write_search_results_json,
        )

    return INDEX_PATH, RetrievalResult, search_saved_index, write_search_results_json


try:
    write_log = load_logger()
    INDEX_PATH, RetrievalResult, search_saved_index, write_search_results_json = (
        load_retriever_parts()
    )
except Exception as import_error:  # pragma: no cover - defensive startup guard
    raise import_error


def format_result(result: RetrievalResult, rank: int) -> str:
    """Turn one retrieval result into readable text."""

    lines = [
        f"Result {rank}",
        f"Title: {result.title}",
        f"URL: {result.url}",
        f"Score: {result.score:.4f}",
        f"Chunk Index: {result.chunk_index}",
        f"Text: {result.text}",
    ]
    return "\n".join(lines)


def print_results(results: list[RetrievalResult]) -> None:
    """Print search results to the terminal."""

    if not results:
        print("No matching results were found.")
        return

    for rank, result in enumerate(results, start=1):
        print(format_result(result, rank))
        print()


def run_search(query: str, top_k: int = 5, index_path: Path | str = INDEX_PATH) -> list[RetrievalResult]:
    """Run a search over the saved BM25 index."""

    return search_saved_index(query=query, index_path=index_path, top_k=top_k)


def main() -> None:
    """Read command-line arguments and run a search."""

    parser = argparse.ArgumentParser(description="Search the saved BM25 index.")
    parser.add_argument("query", type=str, help="The question or search phrase to look up.")
    parser.add_argument(
        "--top-k",
        type=int,
        default=5,
        help="How many results to return.",
    )
    parser.add_argument(
        "--save-json",
        type=Path,
        default=None,
        help="Optional path to save the search results as JSON.",
    )

    args = parser.parse_args()

    # Log what the user searched for so later we can trace what was run.
    write_log(f"search.py query='{args.query}' top_k={args.top_k}")
    results = run_search(query=args.query, top_k=args.top_k)
    print_results(results)

    if args.save_json is not None:
        output_path = write_search_results_json(
            query=args.query,
            out_path=args.save_json,
            top_k=args.top_k,
        )
        print(f"Saved JSON results to {output_path}")


if __name__ == "__main__":
    main()
