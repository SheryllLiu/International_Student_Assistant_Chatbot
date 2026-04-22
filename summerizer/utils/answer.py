"""Generate a simple grounded answer from retrieved BM25 chunks.

Think of this file as the first baseline "answering" layer.

It does not call GPT or Ollama. Instead, it:
1. retrieves the top BM25 chunks
2. looks for sentences that match the question well
3. stitches those sentences into a short answer
4. shows the source URLs used as evidence
"""

from __future__ import annotations

import argparse
import json
import re
from dataclasses import asdict, dataclass
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
        )
    except ModuleNotFoundError:
        from retriever import (  # type: ignore
            INDEX_PATH,
            RetrievalResult,
            search_saved_index,
        )

    return INDEX_PATH, RetrievalResult, search_saved_index


try:
    write_log = load_logger()
    INDEX_PATH, RetrievalResult, search_saved_index = load_retriever_parts()
except Exception as import_error:  # pragma: no cover - defensive startup guard
    raise import_error


@dataclass
class AnswerResult:
    """A simple answer built from retrieved evidence."""

    query: str
    answer: str
    citations: list[str]
    supporting_chunks: list[dict]


STOPWORDS: frozenset[str] = frozenset(
    """
    a an the and or but if while of at by for with about against between into through during
    before after above below to from up down in out on off over under again further then once
    here there when where why how all any both each few more most other some such no nor not
    only own same so than too very is are was were be been being have has had do does did doing
    this that these those i you he she it we they them their our my your his her its as also
    """.split()
)


def split_into_sentences(text: str) -> list[str]:
    """Split text into rough sentence-sized pieces."""

    cleaned_text = re.sub(r"\s+", " ", text).strip()
    if not cleaned_text:
        return []
    return [part.strip() for part in re.split(r"(?<=[.!?])\s+", cleaned_text) if part.strip()]


def tokenize_for_overlap(text: str) -> list[str]:
    """Convert text into lowercase tokens for overlap scoring."""

    cleaned_text = re.sub(r"\s+", " ", text).strip().lower()
    raw_tokens = re.findall(r"[a-zA-Z]+|\d+", cleaned_text)

    useful_tokens: list[str] = []
    for token in raw_tokens:
        if len(token) < 2:
            continue
        if token in STOPWORDS:
            continue
        useful_tokens.append(token)
    return useful_tokens


def score_sentence_against_query(query: str, sentence: str) -> int:
    """Score a sentence by how much it overlaps with the query words."""

    query_tokens = set(tokenize_for_overlap(query))
    sentence_tokens = tokenize_for_overlap(sentence)
    overlap_count = sum(1 for token in sentence_tokens if token in query_tokens)
    return overlap_count


def choose_supporting_sentences(
    query: str,
    results: list[RetrievalResult],
    max_sentences: int = 3,
) -> list[str]:
    """Choose the most query-relevant sentences from the retrieved chunks.

    We score each sentence by token overlap with the question. Higher overlap
    means the sentence is more likely to directly answer what the user asked.
    """

    scored_sentences: list[tuple[int, int, int, str]] = []
    seen_sentences: set[str] = set()

    for result_rank, result in enumerate(results):
        for sentence_index, sentence in enumerate(split_into_sentences(result.text)):
            if sentence in seen_sentences:
                continue
            if len(sentence.split()) < 6:
                continue

            sentence_score = score_sentence_against_query(query, sentence)
            if sentence_score == 0:
                continue

            # We sort first by overlap score, then use earlier retrieval rank
            # and earlier sentence position as tie-breakers.
            scored_sentences.append((sentence_score, -result_rank, -sentence_index, sentence))
            seen_sentences.add(sentence)

    scored_sentences.sort(reverse=True)
    return [sentence for _, _, _, sentence in scored_sentences[:max_sentences]]


def build_citations(results: list[RetrievalResult]) -> list[str]:
    """Return source URLs in retrieval order without duplicates."""

    citations: list[str] = []
    seen_urls: set[str] = set()

    for result in results:
        if result.url and result.url not in seen_urls:
            citations.append(result.url)
            seen_urls.add(result.url)

    return citations


def build_answer_text(query: str, results: list[RetrievalResult]) -> str:
    """Turn retrieved chunks into a short readable answer."""

    if not results:
        return (
            f"I could not find a strong match in the indexed documents for: {query}. "
            "Try rephrasing the question or retrieving more results."
        )

    supporting_sentences = choose_supporting_sentences(query, results, max_sentences=3)
    if not supporting_sentences:
        return (
            "I found related documents, but could not extract a clear short answer "
            "from the top retrieved chunks."
        )

    answer_parts = ["Based on the retrieved documents:"]
    answer_parts.extend(supporting_sentences)
    return " ".join(answer_parts)


def answer_question(query: str, top_k: int = 5, index_path: Path | str = INDEX_PATH) -> AnswerResult:
    """Retrieve evidence and build a grounded answer."""

    results = search_saved_index(query=query, index_path=index_path, top_k=top_k)
    answer_text = build_answer_text(query, results)
    citations = build_citations(results)
    supporting_chunks = [asdict(result) for result in results]

    return AnswerResult(
        query=query,
        answer=answer_text,
        citations=citations,
        supporting_chunks=supporting_chunks,
    )


def print_answer(result: AnswerResult) -> None:
    """Print the final answer in a readable format."""

    print("Question:")
    print(result.query)
    print()

    print("Answer:")
    print(result.answer)
    print()

    print("Citations:")
    if not result.citations:
        print("No citations found.")
    else:
        for citation in result.citations:
            print(citation)


def write_answer_json(result: AnswerResult, out_path: Path | str) -> Path:
    """Save the answer result as JSON."""

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(asdict(result), indent=2, ensure_ascii=False), encoding="utf-8")
    return out_path


def main() -> None:
    """Read command-line arguments and answer a question."""

    parser = argparse.ArgumentParser(description="Answer a question using the saved BM25 index.")
    parser.add_argument("query", type=str, help="The question to answer.")
    parser.add_argument("--top-k", type=int, default=5, help="How many chunks to retrieve.")
    parser.add_argument(
        "--save-json",
        type=Path,
        default=None,
        help="Optional path to save the answer as JSON.",
    )

    args = parser.parse_args()

    # Log the question so we can trace what was run later.
    write_log(f"answer.py query='{args.query}' top_k={args.top_k}")
    result = answer_question(query=args.query, top_k=args.top_k)
    print_answer(result)

    if args.save_json is not None:
        output_path = write_answer_json(result, args.save_json)
        print()
        print(f"Saved answer JSON to {output_path}")
        write_log(f"answer.py saved answer JSON to {output_path}")


if __name__ == "__main__":
    main()
