"""Build and query a BM25 index over chunked documents.

Think of this file as the "search engine" step of the pipeline.

It does three jobs:
1. read the chunk data from ``data/chunks/chunks.json``
2. build and save a BM25 index
3. run searches over that saved index
"""

from __future__ import annotations

import json
import pickle
import re
from dataclasses import asdict, dataclass
from pathlib import Path

from rank_bm25 import BM25Okapi

try:
    from summerizer.utils.logger import write_log
except ModuleNotFoundError:
    from logger import write_log  # type: ignore

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CHUNKS_PATH = PROJECT_ROOT / "data" / "chunks" / "chunks.json"
INDEX_DIR = PROJECT_ROOT / "data" / "index"
INDEX_PATH = INDEX_DIR / "bm25_index.pkl"

# These common words usually do not help search much, so we drop them from
# both the chunk text and the user query.
STOPWORDS: frozenset[str] = frozenset(
    """
    a an the and or but if while of at by for with about against between into through during
    before after above below to from up down in out on off over under again further then once
    here there when where why how all any both each few more most other some such no nor not
    only own same so than too very is are was were be been being have has had do does did doing
    this that these those i you he she it we they them their our my your his her its as also
    """.split()
)

TOKEN_PATTERN = re.compile(r"[A-Za-z][A-Za-z']*[A-Za-z]|[A-Za-z]|\d+")


@dataclass
class RetrievalResult:
    """One retrieved chunk and its BM25 score."""

    chunk_id: str
    url: str
    title: str
    chunk_index: int
    text: str
    score: float


@dataclass
class SavedIndex:
    """Everything we need to save and later reload the BM25 index."""

    chunks: list[dict]
    tokenized_chunks: list[list[str]]
    bm25: BM25Okapi


def normalize_whitespace(text: str) -> str:
    """Turn repeated whitespace into single spaces."""

    return re.sub(r"\s+", " ", text).strip()


def tokenize(text: str) -> list[str]:
    """Convert text into simple lowercase search tokens.

    We use the same tokenizer for both:
    - chunk text when we build the index
    - user queries when we search

    That keeps the search vocabulary consistent.
    """

    cleaned_text = normalize_whitespace(text).lower()
    raw_tokens = TOKEN_PATTERN.findall(cleaned_text)

    useful_tokens: list[str] = []
    for token in raw_tokens:
        if len(token) < 2:
            continue
        if token in STOPWORDS:
            continue
        useful_tokens.append(token)

    return useful_tokens


def load_chunks(chunks_path: Path | str = CHUNKS_PATH) -> list[dict]:
    """Load chunk data from JSON."""

    chunks_path = Path(chunks_path)
    return json.loads(chunks_path.read_text(encoding="utf-8"))


def build_index(chunks: list[dict]) -> SavedIndex:
    """Build a BM25 index from a list of chunk dictionaries."""

    if not chunks:
        raise ValueError("build_index requires at least one chunk")

    # BM25 does not work on raw strings directly. It expects tokenized text.
    tokenized_chunks = [tokenize(chunk.get("text", "")) for chunk in chunks]
    bm25 = BM25Okapi(tokenized_chunks)

    return SavedIndex(
        chunks=chunks,
        tokenized_chunks=tokenized_chunks,
        bm25=bm25,
    )


def save_index(saved_index: SavedIndex, index_path: Path | str = INDEX_PATH) -> Path:
    """Save the BM25 index to disk.

    We store a plain dictionary instead of pickling the dataclass directly.
    That makes the saved file safer to load later, even if the index was
    created by running this file as a script.
    """

    index_path = Path(index_path)
    index_path.parent.mkdir(parents=True, exist_ok=True)

    pickle_ready_index = {
        "chunks": saved_index.chunks,
        "tokenized_chunks": saved_index.tokenized_chunks,
        "bm25": saved_index.bm25,
    }

    with index_path.open("wb") as file:
        pickle.dump(pickle_ready_index, file)

    return index_path


def load_index(index_path: Path | str = INDEX_PATH) -> SavedIndex:
    """Load a previously saved BM25 index from disk."""

    index_path = Path(index_path)
    with index_path.open("rb") as file:
        loaded_data = pickle.load(file)

    return SavedIndex(
        chunks=loaded_data["chunks"],
        tokenized_chunks=loaded_data["tokenized_chunks"],
        bm25=loaded_data["bm25"],
    )


def build_and_save_index(
    chunks_path: Path | str = CHUNKS_PATH,
    index_path: Path | str = INDEX_PATH,
) -> Path:
    """Read chunks, build the index, and save it."""

    write_log("retriever.py started")
    chunks = load_chunks(chunks_path)
    saved_index = build_index(chunks)
    written_path = save_index(saved_index, index_path)
    write_log(f"retriever.py built BM25 index with {len(chunks)} chunks at {written_path}")
    return written_path


def search_index(
    query: str,
    saved_index: SavedIndex,
    top_k: int = 5,
) -> list[RetrievalResult]:
    """Search the BM25 index and return the top matching chunks."""

    query_tokens = tokenize(query)
    if not query_tokens:
        return []

    # BM25 gives one score per chunk. We sort from highest score to lowest and
    # keep only the top results the caller asked for.
    scores = saved_index.bm25.get_scores(query_tokens)
    ranked_scores = sorted(
        enumerate(scores),
        key=lambda item: item[1],
        reverse=True,
    )[:top_k]

    results: list[RetrievalResult] = []
    for chunk_position, score in ranked_scores:
        chunk = saved_index.chunks[chunk_position]
        results.append(
            RetrievalResult(
                chunk_id=chunk.get("chunk_id", ""),
                url=chunk.get("url", ""),
                title=chunk.get("title", ""),
                chunk_index=chunk.get("chunk_index", 0),
                text=chunk.get("text", ""),
                score=float(score),
            )
        )

    return results


def search_saved_index(
    query: str,
    index_path: Path | str = INDEX_PATH,
    top_k: int = 5,
) -> list[RetrievalResult]:
    """Load the saved index and search it."""

    saved_index = load_index(index_path)
    return search_index(query, saved_index, top_k=top_k)


def write_search_results_json(
    query: str,
    out_path: Path | str,
    index_path: Path | str = INDEX_PATH,
    top_k: int = 5,
) -> Path:
    """Run a search and save the results as JSON."""

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    results = search_saved_index(query, index_path=index_path, top_k=top_k)
    json_ready_results = [asdict(result) for result in results]

    out_path.write_text(
        json.dumps(json_ready_results, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    write_log(
        f"retriever.py saved {len(results)} search results for query '{query}' to {out_path}"
    )
    return out_path


if __name__ == "__main__":
    output = build_and_save_index()
    print(f"wrote BM25 index to {output}")
