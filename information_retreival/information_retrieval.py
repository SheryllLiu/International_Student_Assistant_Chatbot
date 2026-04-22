"""BM25 retriever over the pickled inverted index.

The retriever loads ``data/processed/indexes/bm25_index.pkl`` (produced by
``utils.inverted_index``) and answers queries with the standard BM25 formula::

    score(q, d) = sum_{t in q_unique}
        idf[t] * tf * (k1 + 1) / (tf + k1 * (1 - b + b * doc_len[d] / avgdl))

Only documents that contain at least one query term are scored — we walk the
posting lists rather than iterating the full corpus.
"""
from __future__ import annotations

import pickle
from collections import defaultdict
from pathlib import Path
from typing import Any

from utils.text_cleaning import clean_query_text

DEFAULT_INDEX_PATH = "data/processed/indexes/bm25_index.pkl"

# doc_id / score / topic / cleaned_document are always returned. These
# additional columns from ``doc_store`` are surfaced when present so the
# retriever gracefully supports both schemas (cleaned-only vs cleaned + raw).
OPTIONAL_RESULT_FIELDS = ("title", "section", "text")


class BM25Retriever:
    """Score and rank documents for a free-text query using BM25."""

    def __init__(self, index_path: str | Path = DEFAULT_INDEX_PATH):
        self.index_path = Path(index_path)
        self.load_index(self.index_path)

    def load_index(self, index_path: str | Path) -> None:
        """Unpickle the BM25 index and bind its components to ``self``."""
        path = Path(index_path)
        with open(path, "rb") as f:
            index = pickle.load(f)
        self.inverted_index: dict[str, list[tuple[int, int]]] = index["inverted_index"]
        self.doc_len: dict[int, int] = index["doc_len"]
        self.idf: dict[str, float] = index["idf"]
        self.avgdl: float = index["avgdl"]
        self.N: int = index["N"]
        self.k1: float = index["k1"]
        self.b: float = index["b"]
        self.doc_store: dict[int, dict[str, Any]] = index["doc_store"]

    def search(self, query: str, top_k: int = 2) -> list[dict[str, Any]]:
        """Return the top-``top_k`` documents for ``query``.

        An empty query (or one that cleans away to nothing) returns ``[]``.
        Query terms that don't appear in the inverted index are silently
        skipped. Dedup on query terms: repeating a token in the query does
        not inflate its BM25 contribution.
        """
        if not isinstance(query, str) or not query.strip():
            return []

        cleaned = clean_query_text(query)
        tokens = cleaned.split() if cleaned else []
        if not tokens:
            return []

        k1 = self.k1
        b = self.b
        avgdl = self.avgdl
        scores: dict[int, float] = defaultdict(float)

        for term in set(tokens):
            postings = self.inverted_index.get(term)
            if not postings:
                continue
            idf_t = self.idf.get(term, 0.0)
            for doc_id, tf in postings:
                dl = self.doc_len[doc_id]
                # Guard ``avgdl == 0`` so an empty corpus doesn't ZeroDivision.
                norm = (1 - b + b * dl / avgdl) if avgdl else 1.0
                denom = tf + k1 * norm
                scores[doc_id] += idf_t * tf * (k1 + 1) / denom

        if not scores:
            return []

        ranked = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)[:top_k]

        results: list[dict[str, Any]] = []
        for doc_id, score in ranked:
            meta = self.doc_store.get(doc_id, {})
            row: dict[str, Any] = {
                "doc_id": doc_id,
                "score": score,
                "topic": meta.get("topic", ""),
                "cleaned_document": meta.get("cleaned_document", ""),
            }
            for field in OPTIONAL_RESULT_FIELDS:
                if field in meta:
                    row[field] = meta[field]
            results.append(row)
        return results
