"""Compare BM25 vs Hybrid (BM25 + Dense, RRF) on a gold-labeled query set.

Metrics (averaged across queries), using ``top_k = 5``:

* **P@5**   — of the top-5 retrieved, what fraction are relevant.
* **R@5**   — of all relevant docs, what fraction made the top-5.
* **MRR**   — mean of 1 / (rank of first relevant hit); 0 if none found.
* **Hit@5** — fraction of queries with at least one relevant doc in top-5.
* **F1**    — harmonic mean of Precision and Recall.

All five are standard IR metrics.

Gold set lives at ``data/eval/queries.json`` as a list of
``{"query": str, "relevant": [doc_id, ...]}`` objects.

Run from repo root::

    python -m information_retreival.eval
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from rag_chatbot.information_retrieval.bm25_retrieval import BM25Retriever
from rag_chatbot.information_retrieval.hybrid_retrieval import HybridRetriever

QUERIES_PATH = Path("data/eval/queries.json")
TOP_K = 5


def precision_at_k(retrieved: list[int], relevant: set[int], k: int) -> float:
    """Fraction of the top-``k`` retrieved doc IDs that are in ``relevant``."""
    top = retrieved[:k]
    if not top:
        return 0.0
    return sum(1 for d in top if d in relevant) / k


def recall_at_k(retrieved: list[int], relevant: set[int], k: int) -> float:
    """Fraction of ``relevant`` doc IDs that appear in the top ``k`` retrieved."""
    if not relevant:
        return 0.0
    top = retrieved[:k]
    return sum(1 for d in top if d in relevant) / len(relevant)


def reciprocal_rank(retrieved: list[int], relevant: set[int]) -> float:
    """Return ``1 / rank`` of the first relevant doc, or 0 if none are found."""
    for i, d in enumerate(retrieved, start=1):
        if d in relevant:
            return 1.0 / i
    return 0.0


def hit_at_k(retrieved: list[int], relevant: set[int], k: int) -> float:
    """1.0 if any relevant doc is in the top ``k`` retrieved, else 0.0."""
    return 1.0 if any(d in relevant for d in retrieved[:k]) else 0.0


def f1_score(p: float, r: float) -> float:
    """Harmonic mean of precision ``p`` and recall ``r`` (0 when both are 0)."""
    if p + r == 0:
        return 0.0
    return 2 * p * r / (p + r)


def evaluate(retriever: Any, queries: list[dict], k: int = TOP_K) -> dict[str, Any]:
    """Run ``retriever`` over every query and return aggregate + per-query stats."""
    p_scores, r_scores, rr_scores, hit_scores, f1_scores = [], [], [], [], []
    per_query: list[dict[str, Any]] = []

    for q in queries:
        hits = retriever.search(q["query"], top_k=max(k, 10))
        retrieved = [h["doc_id"] for h in hits]
        relevant = set(q["relevant"])

        p = precision_at_k(retrieved, relevant, k)
        r = recall_at_k(retrieved, relevant, k)
        rr = reciprocal_rank(retrieved, relevant)
        h = hit_at_k(retrieved, relevant, k)

        # ✅ NEW: F1 calculation
        f1 = f1_score(p, r)

        p_scores.append(p)
        r_scores.append(r)
        rr_scores.append(rr)
        hit_scores.append(h)
        f1_scores.append(f1)

        per_query.append(
            {
                "query": q["query"],
                "relevant": sorted(relevant),
                "top_k": retrieved[:k],
                f"P@{k}": p,
                f"R@{k}": r,
                "RR": rr,
                "F1": f1,  # optional per-query visibility
            }
        )

    n = len(queries)
    return {
        f"P@{k}": sum(p_scores) / n,
        f"R@{k}": sum(r_scores) / n,
        "MRR": sum(rr_scores) / n,
        f"Hit@{k}": sum(hit_scores) / n,
        "F1": sum(f1_scores) / n,  # ✅ NEW
        "per_query": per_query,
    }


def print_summary(bm25_res: dict, hybrid_res: dict, k: int) -> None:
    """Print a side-by-side BM25 vs Hybrid table for the aggregate metrics."""
    print("=" * 68)
    print(f"{'Metric':<10} {'BM25':>12} {'Hybrid':>12} {'Δ (abs)':>12} {'Δ (%)':>12}")
    print("-" * 68)

    # ✅ UPDATED: include F1
    for metric in (f"P@{k}", f"R@{k}", "MRR", f"Hit@{k}", "F1"):
        b = bm25_res[metric]
        h = hybrid_res[metric]
        delta = h - b
        pct = f"{delta / b * 100:+.1f}%" if b else "—"
        print(f"{metric:<10} {b:>12.4f} {h:>12.4f} {delta:>+12.4f} {pct:>12}")

    print("=" * 68)


def print_per_query(bm25_res: dict, hybrid_res: dict, k: int) -> None:
    """Print per-query P@k for BM25 vs Hybrid with ↑/↓/= deltas."""
    print(f"\n[per-query P@{k} comparison]")
    print(f"{'Query':<55} {'BM25':>6} {'Hybrid':>7}  Δ")
    print("-" * 78)
    for bq, hq in zip(bm25_res["per_query"], hybrid_res["per_query"], strict=True):
        q_short = bq["query"][:54]
        d = hq[f"P@{k}"] - bq[f"P@{k}"]
        marker = "↑" if d > 0 else "↓" if d < 0 else "="
        print(f"{q_short:<55} {bq[f'P@{k}']:>6.2f} {hq[f'P@{k}']:>7.2f}  {marker}")


def main() -> None:
    """Load gold queries, evaluate BM25 and Hybrid, and print both views."""
    queries = json.loads(QUERIES_PATH.read_text())
    print(f"[info] {len(queries)} queries loaded from {QUERIES_PATH}")

    print("[info] running BM25...")
    bm25_res = evaluate(BM25Retriever(), queries)

    print("[info] running Hybrid (BM25 + Dense, RRF k=60)...\n")
    hybrid_res = evaluate(HybridRetriever(), queries)

    print_summary(bm25_res, hybrid_res, TOP_K)
    print_per_query(bm25_res, hybrid_res, TOP_K)


if __name__ == "__main__":
    main()
