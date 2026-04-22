"""Build an inverted index + BM25 statistics from the cleaned corpus.

Reads ``data/processed/cleaned_data.csv``, tokenizes each row's
``cleaned_document`` column on whitespace, and pickles a single dict
containing:

* ``inverted_index``  — ``term -> [(doc_id, tf), ...]``
* ``doc_len``          — ``doc_id -> int``
* ``doc_freq``         — ``term -> df``
* ``idf``              — precomputed ``term -> float`` using the standard
                         BM25 idf formula
* ``avgdl``, ``N``     — corpus-level stats
* ``k1``, ``b``        — BM25 constants (defaults 1.5 / 0.75)
* ``doc_store``        — ``doc_id -> {topic, cleaned_document, ...}`` so the
                         retriever can return human-readable results; any
                         extra columns present in the CSV (e.g. title,
                         section, text) are copied over verbatim.

Run from the repo root::

    python utils/inverted_index.py
"""
from __future__ import annotations

import math
import pickle
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import pandas as pd

IN_FILE = Path("data/processed/cleaned_data.csv")
OUT_DIR = Path("data/processed/indexes")
OUT_FILE = OUT_DIR / "bm25_index.pkl"

DEFAULT_K1 = 1.5
DEFAULT_B = 0.75

CLEANED_COLUMN = "cleaned_document"
TOPIC_COLUMN = "topic"


def tokenize_document(text: str) -> list[str]:
    """Whitespace-split a cleaned document into tokens.

    The corpus has already been normalized by ``utils.text_cleaning``, so
    splitting on whitespace is sufficient — no re-tokenization here.
    """
    if not isinstance(text, str) or not text:
        return []
    return text.split()


def compute_idf(doc_freq: dict[str, int], N: int) -> dict[str, float]:
    """Precompute BM25 idf for every term.

    Uses the standard BM25 idf, which is non-negative for all df ≥ 1::

        idf(t) = log(1 + (N - df + 0.5) / (df + 0.5))
    """
    return {
        term: math.log(1 + (N - df + 0.5) / (df + 0.5))
        for term, df in doc_freq.items()
    }


def _row_to_store_entry(row: pd.Series, columns: list[str]) -> dict[str, Any]:
    """Snapshot one DataFrame row into a plain dict for the doc_store.

    NaNs are coerced to empty strings so downstream consumers don't have to
    special-case missing metadata.
    """
    entry: dict[str, Any] = {}
    for col in columns:
        val = row[col]
        entry[col] = "" if pd.isna(val) else val
    return entry


def build_inverted_index(
    df: pd.DataFrame,
    k1: float = DEFAULT_K1,
    b: float = DEFAULT_B,
) -> dict[str, Any]:
    """Build the full inverted index + BM25 stats from a DataFrame.

    Each row becomes one document; ``doc_id`` is the row's 0-based position
    after ``reset_index``. The returned dict is directly picklable.
    """
    if CLEANED_COLUMN not in df.columns:
        raise ValueError(
            f"input DataFrame is missing required column '{CLEANED_COLUMN}'"
        )

    df = df.reset_index(drop=True)
    N = len(df)

    inverted_index: dict[str, list[tuple[int, int]]] = defaultdict(list)
    doc_len: dict[int, int] = {}
    doc_freq: dict[str, int] = defaultdict(int)
    doc_store: dict[int, dict[str, Any]] = {}

    store_columns = list(df.columns)

    for doc_id, row in df.iterrows():
        tokens = tokenize_document(row.get(CLEANED_COLUMN, ""))
        doc_len[doc_id] = len(tokens)

        term_counts = Counter(tokens)
        for term, tf in term_counts.items():
            inverted_index[term].append((doc_id, tf))
            doc_freq[term] += 1

        doc_store[doc_id] = _row_to_store_entry(row, store_columns)

    inverted_index = dict(inverted_index)
    doc_freq = dict(doc_freq)
    idf = compute_idf(doc_freq, N)
    avgdl = (sum(doc_len.values()) / N) if N else 0.0

    return {
        "inverted_index": inverted_index,
        "doc_len": doc_len,
        "doc_freq": doc_freq,
        "idf": idf,
        "avgdl": avgdl,
        "N": N,
        "k1": k1,
        "b": b,
        "doc_store": doc_store,
    }


def save_index(index: dict[str, Any], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "wb") as f:
        pickle.dump(index, f, protocol=pickle.HIGHEST_PROTOCOL)


def load_index(index_path: Path) -> dict[str, Any]:
    with open(index_path, "rb") as f:
        return pickle.load(f)


def main() -> None:
    if not IN_FILE.exists():
        raise FileNotFoundError(f"input not found: {IN_FILE}")
    df = pd.read_csv(IN_FILE, keep_default_na=False)
    index = build_inverted_index(df)
    save_index(index, OUT_FILE)
    print(
        f"[ok] indexed N={index['N']} docs, "
        f"|V|={len(index['inverted_index'])} unique terms, "
        f"avgdl={index['avgdl']:.2f} -> {OUT_FILE}"
    )


if __name__ == "__main__":
    main()
