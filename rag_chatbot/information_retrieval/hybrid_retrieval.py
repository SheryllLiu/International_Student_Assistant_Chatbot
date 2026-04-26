"""Hybrid retriever: BM25 + Dense fused by Reciprocal Rank Fusion (RRF).

The two retrievers produce scores on very different scales (BM25 is an
unbounded additive sum of idf·tf terms; dense is a cosine in [-1, 1]), so
score-level fusion would need per-corpus min-max calibration. RRF sidesteps
that entirely: it only uses *ranks*.

    RRF(d) = Σ  1 / (k + rank_i(d))
             i

``i`` ranges over the two retrievers, ``rank_i(d)`` is the 1-based rank of
``d`` in retriever ``i``, and documents not returned by ``i`` contribute 0.
``k = 60`` is the value from Cormack et al. (2009) and is the de-facto
default — no tuning required.

Fusion key is ``doc_id``: BM25's ``doc_id`` and Dense's ``row_id`` are the
same integer because both indices are built from the same de-duplicated
source in the same row order.
"""
from __future__ import annotations

from collections import defaultdict
from typing import Any

from rag_chatbot.information_retrieval.dense_retrieval import DenseRetriever
from rag_chatbot.information_retrieval.bm25_retrieval import BM25Retriever

DEFAULT_RRF_K = 60
DEFAULT_CANDIDATE_K = 50

# Metadata fields we surface on each result. BM25 already carries
# ``raw_document`` in its doc_store, so prefer that side when both retrievers
# return the same doc.
META_FIELDS_PREFERRED = ("topic", "title", "section", "text", "raw_document")
META_FIELDS_FALLBACK = ("topic", "title", "section", "text")


class HybridRetriever:
    """BM25 + Dense, fused by RRF."""

    def __init__(
        self,
        bm25: BM25Retriever | None = None,
        dense: DenseRetriever | None = None,
        rrf_k: int = DEFAULT_RRF_K,
        candidate_k: int = DEFAULT_CANDIDATE_K,
    ):
        self.bm25 = bm25 if bm25 is not None else BM25Retriever()
        self.dense = dense if dense is not None else DenseRetriever()
        self.rrf_k = rrf_k
        self.candidate_k = candidate_k

    def search(self, query: str, top_k: int = 10) -> list[dict[str, Any]]:
        """Fuse BM25 and Dense top-``candidate_k`` by RRF, return top-``top_k``."""
        if not isinstance(query, str) or not query.strip():
            return []

        bm25_hits = self.bm25.search(query, top_k=self.candidate_k)
        dense_hits = self.dense.search(query, top_k=self.candidate_k)

        state: dict[int, dict[str, Any]] = defaultdict(
            lambda: {
                "rrf_score": 0.0,
                "bm25_rank": None,
                "bm25_score": 0.0,
                "dense_rank": None,
                "dense_score": 0.0,
                "meta": {},
            }
        )

        for rank, hit in enumerate(bm25_hits, start=1):
            s = state[hit["doc_id"]]
            s["rrf_score"] += 1.0 / (self.rrf_k + rank)
            s["bm25_rank"] = rank
            s["bm25_score"] = hit["final_score"]
            for field in META_FIELDS_PREFERRED:
                if field in hit and field not in s["meta"]:
                    s["meta"][field] = hit[field]

        for rank, hit in enumerate(dense_hits, start=1):
            s = state[hit["doc_id"]]
            s["rrf_score"] += 1.0 / (self.rrf_k + rank)
            s["dense_rank"] = rank
            s["dense_score"] = hit["final_score"]
            for field in META_FIELDS_FALLBACK:
                if field in hit and field not in s["meta"]:
                    s["meta"][field] = hit[field]

        ranked = sorted(
            state.items(), key=lambda kv: kv[1]["rrf_score"], reverse=True
        )[:top_k]

        results: list[dict[str, Any]] = []
        for doc_id, s in ranked:
            row: dict[str, Any] = {
                "doc_id": doc_id,
                "final_score": s["rrf_score"],
                "bm25_rank": s["bm25_rank"],
                "bm25_score": s["bm25_score"],
                "dense_rank": s["dense_rank"],
                "dense_score": s["dense_score"],
            }
            row.update(s["meta"])
            results.append(row)
        return results
