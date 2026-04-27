"""Build per-field BM25 indexes from the cleaned corpus.

Reads ``data/processed/cleaned_data.csv`` and builds three independent BM25
indexes — one for each of ``topic``, ``title``, ``text``. Each field gets its
own inverted index, doc lengths, doc frequencies, idf table, and avgdl, so
short fields (``topic``, ``title``) are length-normalized against other short
fields rather than against the much longer body text.

The pickled output at ``data/indices/bm25_index.pkl`` has the shape::

    {
        "fields": {
            "topic": {inverted_index, doc_len, doc_freq, idf, avgdl, N},
            "title": {...},
            "text":  {...},
        },
        "k1": 1.5,
        "b":  0.3,
        "doc_store": {doc_id: {topic, title, section, text, raw_document, ...}},
        "N": <total docs>,
    }

Run from the repo root::

    python -m summerizer.utils.inverted_index
"""

from __future__ import annotations

import logging
import math
import pickle
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import pandas as pd

from rag_chatbot.utils.text_cleaning import clean_text

logger = logging.getLogger("isa.index.bm25")

IN_FILE = Path("data/processed/cleaned_data.csv")
OUT_DIR = Path("data/indices")
OUT_FILE = OUT_DIR / "bm25_index.pkl"

DEFAULT_K1 = 1.5
DEFAULT_B = 0.3

# Fields that get their own BM25 index. Each is cleaned with the same
# pipeline used on queries so token surfaces match.
INDEXED_FIELDS = ("topic", "title", "text")


def tokenize_field(raw: Any) -> list[str]:
    """Clean a raw field value with the shared pipeline, then whitespace-split."""
    if not isinstance(raw, str) or not raw:
        return []
    cleaned = clean_text(raw)
    return cleaned.split() if cleaned else []


def compute_idf(doc_freq: dict[str, int], N: int) -> dict[str, float]:
    """Standard BM25 idf, non-negative for all df ≥ 1."""
    return {term: math.log(1 + (N - df + 0.5) / (df + 0.5)) for term, df in doc_freq.items()}


def build_field_index(tokens_by_doc: dict[int, list[str]]) -> dict[str, Any]:
    """Build one BM25 sub-index from ``doc_id -> tokens``."""
    inverted_index: dict[str, list[tuple[int, int]]] = defaultdict(list)
    doc_len: dict[int, int] = {}
    doc_freq: dict[str, int] = defaultdict(int)

    for doc_id, tokens in tokens_by_doc.items():
        doc_len[doc_id] = len(tokens)
        for term, tf in Counter(tokens).items():
            inverted_index[term].append((doc_id, tf))
            doc_freq[term] += 1

    N = len(tokens_by_doc)
    idf = compute_idf(dict(doc_freq), N)
    total_len = sum(doc_len.values())
    avgdl = (total_len / N) if N else 0.0

    return {
        "inverted_index": dict(inverted_index),
        "doc_len": doc_len,
        "doc_freq": dict(doc_freq),
        "idf": idf,
        "avgdl": avgdl,
        "N": N,
    }


def _row_to_store_entry(row: pd.Series, columns: list[str]) -> dict[str, Any]:
    """Snapshot a DataFrame row as a plain dict, replacing NaN with empty strings."""
    entry: dict[str, Any] = {}
    for col in columns:
        val = row[col]
        entry[col] = "" if pd.isna(val) else val
    return entry


def build_indexes(
    df: pd.DataFrame,
    k1: float = DEFAULT_K1,
    b: float = DEFAULT_B,
) -> dict[str, Any]:
    """Build per-field BM25 indexes + a shared doc_store."""
    df = df.reset_index(drop=True)
    N = len(df)
    store_columns = list(df.columns)

    tokens_per_field: dict[str, dict[int, list[str]]] = {f: {} for f in INDEXED_FIELDS}
    doc_store: dict[int, dict[str, Any]] = {}

    for doc_id, row in df.iterrows():
        for field in INDEXED_FIELDS:
            raw = row.get(field, "")
            tokens_per_field[field][doc_id] = tokenize_field(raw)
        doc_store[doc_id] = _row_to_store_entry(row, store_columns)

    fields = {field: build_field_index(tokens_per_field[field]) for field in INDEXED_FIELDS}

    return {
        "fields": fields,
        "k1": k1,
        "b": b,
        "doc_store": doc_store,
        "N": N,
    }


def save_index(index: dict[str, Any], output_path: Path) -> None:
    """Pickle ``index`` to ``output_path``, creating parent dirs as needed."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "wb") as f:
        pickle.dump(index, f, protocol=pickle.HIGHEST_PROTOCOL)


def load_index(index_path: Path) -> dict[str, Any]:
    """Read a previously pickled BM25 index from disk."""
    with open(index_path, "rb") as f:
        return pickle.load(f)


def main() -> None:
    """Build the per-field BM25 indexes from the cleaned corpus and persist them."""
    if not IN_FILE.exists():
        raise FileNotFoundError(f"input not found: {IN_FILE}")
    logger.info("loading corpus from %s", IN_FILE)
    df = pd.read_csv(IN_FILE, keep_default_na=False)
    logger.info("building BM25 indexes for %d docs", len(df))
    index = build_indexes(df)
    save_index(index, OUT_FILE)
    for field, sub in index["fields"].items():
        logger.info(
            "field=%s  vocab=%d  avgdl=%.2f", field, len(sub["inverted_index"]), sub["avgdl"]
        )
    logger.info("saved BM25 index — N=%d docs -> %s", index["N"], OUT_FILE)


if __name__ == "__main__":
    main()
