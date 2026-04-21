"""End-to-end demo on the bundled sample corpus.

No network required. Parses the tiny HTML files under ``data/sample/``,
builds a BM25 index in memory, runs a few sample F-1 queries, and prints
the grounded answers. Uses the stub generator unless ``OPENAI_API_KEY``
is set.
"""

from __future__ import annotations

from pathlib import Path

from f1_immigration_assistant.chunker import chunk_documents
from f1_immigration_assistant.logging_config import setup_logging
from f1_immigration_assistant.parser import parse_raw_dir
from f1_immigration_assistant.pipeline import Pipeline

SAMPLE_QUERIES = [
    "How many hours can I work on campus during the semester?",
    "When can I apply for post-completion OPT?",
    "What do I need to re-enter the US after traveling abroad?",
    "Do I need to file taxes if I had no US income?",
    "I dropped below full-time — am I out of status?",
]


def main() -> None:
    """Run the offline demo."""

    setup_logging("INFO")
    sample_dir = Path(__file__).resolve().parents[1] / "data" / "sample"

    docs = parse_raw_dir(sample_dir)
    if not docs:
        raise SystemExit(f"no sample HTML in {sample_dir}")
    chunks = chunk_documents(docs)

    pipeline = Pipeline().fit_chunks(chunks)

    for q in SAMPLE_QUERIES:
        print("\n" + "=" * 80)
        print("Q:", q)
        result = pipeline.answer(q)
        print("A:", result.answer)
        if result.warning:
            print("⚠", result.warning)
        print("Sources:")
        for url in result.citations:
            print(" -", url)
        print(f"(used_llm={result.used_llm})")


if __name__ == "__main__":
    main()
