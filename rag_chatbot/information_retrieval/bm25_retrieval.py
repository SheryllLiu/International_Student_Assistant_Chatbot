"""Field-weighted BM25 retriever over per-field inverted indexes.

Loads the pickle produced by ``utils.inverted_index`` (which stores one BM25
sub-index per field) and scores each query on every field independently, then
combines the per-field scores with a fixed weight vector::

    final_score(q, d) = 3.0 * BM25_topic(q, d)
                      + 3.0 * BM25_title(q, d)
                      + 2.5 * BM25_text (q, d)

Only documents that hit at least one query term in at least one field are
scored — we walk the posting lists per field rather than iterating the full
corpus. OOV terms are silently skipped.
"""
from __future__ import annotations

import pickle
from collections import defaultdict
from pathlib import Path
from typing import Any

from rag_chatbot.utils.text_cleaning import clean_query_text

DEFAULT_INDEX_PATH = "data/indices/bm25_index.pkl"

# Field weights. Short, high-signal fields (topic, title) dominate; body text
# contributes but is down-weighted because it's much longer and noisier.
FIELD_WEIGHTS: dict[str, float] = {
    "topic": 3.0,
    "title": 2.0,
    "text":  3.0,
}

# Raw display fields surfaced alongside the score breakdown.
RAW_RESULT_FIELDS = ("title", "section", "text", "raw_document")


class BM25Retriever:
    """Field-weighted BM25 over topic / title / text sub-indexes."""

    def __init__(self, index_path: str | Path = DEFAULT_INDEX_PATH):
        self.index_path = Path(index_path)
        self.load_index(self.index_path)

    def load_index(self, index_path: str | Path) -> None:
        path = Path(index_path)
        with open(path, "rb") as f:
            index = pickle.load(f)
        self.fields: dict[str, dict[str, Any]] = index["fields"]
        self.k1: float = index["k1"]
        self.b: float = index["b"]
        self.doc_store: dict[int, dict[str, Any]] = index["doc_store"]
        self.N: int = index["N"]

    def _score_field(
        self, field_name: str, tokens: list[str]
    ) -> dict[int, float]:
        """Standard BM25 on one field's posting lists. Returns doc_id -> score."""
        sub = self.fields[field_name]
        inverted_index = sub["inverted_index"]
        doc_len = sub["doc_len"]
        idf_table = sub["idf"]
        avgdl = sub["avgdl"]

        k1 = self.k1
        b = self.b
        scores: dict[int, float] = defaultdict(float)

        for term in set(tokens):
            postings = inverted_index.get(term)
            if not postings:
                continue
            idf_t = idf_table.get(term, 0.0)
            for doc_id, tf in postings:
                dl = doc_len[doc_id]
                norm = (1 - b + b * dl / avgdl) if avgdl else 1.0
                denom = tf + k1 * norm
                scores[doc_id] += idf_t * tf * (k1 + 1) / denom
        return scores

    def search(self, query: str, top_k: int = 10) -> list[dict[str, Any]]:
        """Return the top-``top_k`` documents for ``query`` by final_score."""
        if not isinstance(query, str) or not query.strip():
            return []

        cleaned = clean_query_text(query)
        tokens = cleaned.split() if cleaned else []
        if not tokens:
            return []

        per_field_scores: dict[str, dict[int, float]] = {
            field: self._score_field(field, tokens) for field in FIELD_WEIGHTS
        }

        # Union of hit docs across all fields. A doc only needs to match on
        # one field to be scored; missing fields contribute 0.
        hit_doc_ids: set[int] = set()
        for field_scores in per_field_scores.values():
            hit_doc_ids.update(field_scores.keys())

        if not hit_doc_ids:
            return []

        final: list[tuple[int, float, dict[str, float]]] = []
        for doc_id in hit_doc_ids:
            breakdown: dict[str, float] = {}
            total = 0.0
            for field, weight in FIELD_WEIGHTS.items():
                s = per_field_scores[field].get(doc_id, 0.0)
                breakdown[field] = s
                total += weight * s
            final.append((doc_id, total, breakdown))

        final.sort(key=lambda t: t[1], reverse=True)
        final = final[:top_k]

        results: list[dict[str, Any]] = []
        for doc_id, total, breakdown in final:
            meta = self.doc_store.get(doc_id, {})
            row: dict[str, Any] = {
                "doc_id": doc_id,
                "final_score": total,
                "bm25_topic": breakdown["topic"],
                "bm25_title": breakdown["title"],
                "bm25_text":  breakdown["text"],
                "topic": meta.get("topic", ""),
            }
            for field in RAW_RESULT_FIELDS:
                if field in meta:
                    row[field] = meta[field]
            results.append(row)
        return results
