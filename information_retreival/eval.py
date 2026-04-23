"""Compare BM25 vs Hybrid (BM25 + Dense, RRF) on a gold-labeled query set.

Metrics (averaged across queries), using ``top_k = 5``:

* **P@5**   — of the top-5 retrieved, what fraction are relevant.
* **R@5**   — of all relevant docs, what fraction made the top-5.
* **MRR**   — mean of 1 / (rank of first relevant hit); 0 if none found.
* **Hit@5** — fraction of queries with at least one relevant doc in top-5.

All four are standard IR metrics. P@5 and MRR are usually the two reported in
a slide; Hit@5 is a softer floor ("did we find *anything* useful?").

Gold set lives at ``data/eval/queries.json`` as a list of
``{"query": str, "relevant": [doc_id, ...]}`` objects. Doc ids are the
``row_id`` / BM25 ``doc_id`` (the two are aligned by construction).

Run from repo root::

    python -m information_retreival.eval
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from information_retreival.hybrid_retrieval import HybridRetriever
from information_retreival.information_retrieval import BM25Retriever

QUERIES_PATH = Path("data/eval/queries.json")
TOP_K = 5


def precision_at_k(retrieved: list[int], relevant: set[int], k: int) -> float:
    top = retrieved[:k]
    if not top:
        return 0.0
    return sum(1 for d in top if d in relevant) / k


def recall_at_k(retrieved: list[int], relevant: set[int], k: int) -> float:
    if not relevant:
        return 0.0
    top = retrieved[:k]
    return sum(1 for d in top if d in relevant) / len(relevant)


def reciprocal_rank(retrieved: list[int], relevant: set[int]) -> float:
    for i, d in enumerate(retrieved, start=1):
        if d in relevant:
            return 1.0 / i
    return 0.0


def hit_at_k(retrieved: list[int], relevant: set[int], k: int) -> float:
    return 1.0 if any(d in relevant for d in retrieved[:k]) else 0.0


def evaluate(
    retriever: Any, queries: list[dict], k: int = TOP_K
) -> dict[str, Any]:
    """Run ``retriever`` over every query and return aggregate + per-query stats."""
    p_scores, r_scores, rr_scores, hit_scores = [], [], [], []
    per_query: list[dict[str, Any]] = []

    for q in queries:
        hits = retriever.search(q["query"], top_k=max(k, 10))
        retrieved = [h["doc_id"] for h in hits]
        relevant = set(q["relevant"])

        p = precision_at_k(retrieved, relevant, k)
        r = recall_at_k(retrieved, relevant, k)
        rr = reciprocal_rank(retrieved, relevant)
        h = hit_at_k(retrieved, relevant, k)

        p_scores.append(p)
        r_scores.append(r)
        rr_scores.append(rr)
        hit_scores.append(h)

        per_query.append(
            {
                "query": q["query"],
                "relevant": sorted(relevant),
                "top_k": retrieved[:k],
                f"P@{k}": p,
                f"R@{k}": r,
                "RR": rr,
            }
        )

    n = len(queries)
    return {
        f"P@{k}": sum(p_scores) / n,
        f"R@{k}": sum(r_scores) / n,
        "MRR": sum(rr_scores) / n,
        f"Hit@{k}": sum(hit_scores) / n,
        "per_query": per_query,
    }


def print_summary(bm25_res: dict, hybrid_res: dict, k: int) -> None:
    print("=" * 68)
    print(f"{'Metric':<10} {'BM25':>12} {'Hybrid':>12} {'Δ (abs)':>12} {'Δ (%)':>12}")
    print("-" * 68)
    for metric in (f"P@{k}", f"R@{k}", "MRR", f"Hit@{k}"):
        b = bm25_res[metric]
        h = hybrid_res[metric]
        delta = h - b
        pct = f"{delta / b * 100:+.1f}%" if b else "—"
        print(f"{metric:<10} {b:>12.4f} {h:>12.4f} {delta:>+12.4f} {pct:>12}")
    print("=" * 68)


def print_per_query(bm25_res: dict, hybrid_res: dict, k: int) -> None:
    print(f"\n[per-query P@{k} comparison]")
    print(f"{'Query':<55} {'BM25':>6} {'Hybrid':>7}  Δ")
    print("-" * 78)
    for bq, hq in zip(
        bm25_res["per_query"], hybrid_res["per_query"], strict=True
    ):
        q_short = bq["query"][:54]
        d = hq[f"P@{k}"] - bq[f"P@{k}"]
        marker = "↑" if d > 0 else "↓" if d < 0 else "="
        print(
            f"{q_short:<55} {bq[f'P@{k}']:>6.2f} {hq[f'P@{k}']:>7.2f}  {marker}"
        )


def main() -> None:
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
