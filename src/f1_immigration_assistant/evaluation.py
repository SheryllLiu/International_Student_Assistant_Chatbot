"""Minimal retrieval evaluation: Precision@k, Recall@k, optional mAP.

Judgments file format (JSON):

    [
      {
        "query": "How many hours can I work on campus during the semester?",
        "relevant_urls": [
          "https://internationalservices.georgetown.edu/students/employment/f-1-on-campus-employment/"
        ]
      },
      ...
    ]

We judge relevance at the **source URL** level: a retrieved chunk counts as
relevant if its ``url`` is listed in ``relevant_urls`` for that query. This
keeps the gold set tiny and stable even as chunking parameters change.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

from f1_immigration_assistant.bm25_retriever import BM25Retriever
from f1_immigration_assistant.models import RetrievalResult
from f1_immigration_assistant.utils import read_json

logger = logging.getLogger(__name__)


@dataclass
class EvalReport:
    """Aggregate metrics plus per-query details."""

    precision_at_k: float
    recall_at_k: float
    mean_ap: float
    k: int
    per_query: list[dict]


def _is_relevant(result: RetrievalResult, relevant_urls: set[str]) -> bool:
    return result.chunk.url in relevant_urls


def precision_at_k(results: list[RetrievalResult], relevant_urls: set[str], k: int) -> float:
    """Fraction of the top-k URLs that are relevant (URL-level, no double-count)."""

    if k <= 0 or not results:
        return 0.0
    top = results[:k]
    seen_urls: set[str] = set()
    hits = 0
    for r in top:
        if r.chunk.url in relevant_urls and r.chunk.url not in seen_urls:
            hits += 1
        seen_urls.add(r.chunk.url)
    return hits / k


def recall_at_k(results: list[RetrievalResult], relevant_urls: set[str], k: int) -> float:
    """Fraction of relevant URLs that appear among the top-k results."""

    if not relevant_urls:
        return 0.0
    top_urls = {r.chunk.url for r in results[:k]}
    return len(top_urls & relevant_urls) / len(relevant_urls)


def average_precision(results: list[RetrievalResult], relevant_urls: set[str]) -> float:
    """Average precision over the ranked URL list (deduplicated)."""

    if not relevant_urls or not results:
        return 0.0
    seen_urls: set[str] = set()
    hits = 0
    precisions: list[float] = []
    rank = 0
    for r in results:
        if r.chunk.url in seen_urls:
            continue
        seen_urls.add(r.chunk.url)
        rank += 1
        if r.chunk.url in relevant_urls:
            hits += 1
            precisions.append(hits / rank)
    return sum(precisions) / len(relevant_urls) if precisions else 0.0


def evaluate(
    retriever: BM25Retriever,
    queries_path: Path | str,
    k: int = 5,
) -> EvalReport:
    """Run retrieval for every query and return aggregated metrics."""

    queries = read_json(Path(queries_path))
    p_scores: list[float] = []
    r_scores: list[float] = []
    ap_scores: list[float] = []
    per_query: list[dict] = []

    for item in queries:
        query = item["query"]
        relevant = set(item.get("relevant_urls", []))
        results = retriever.search(query, top_k=max(k, 10))

        p = precision_at_k(results, relevant, k)
        r = recall_at_k(results, relevant, k)
        ap = average_precision(results, relevant)
        p_scores.append(p)
        r_scores.append(r)
        ap_scores.append(ap)
        per_query.append(
            {
                "query": query,
                "precision_at_k": p,
                "recall_at_k": r,
                "average_precision": ap,
                "top_urls": [res.chunk.url for res in results[:k]],
            }
        )

    def mean(xs: list[float]) -> float:
        return sum(xs) / len(xs) if xs else 0.0

    report = EvalReport(
        precision_at_k=mean(p_scores),
        recall_at_k=mean(r_scores),
        mean_ap=mean(ap_scores),
        k=k,
        per_query=per_query,
    )
    logger.info(
        "eval@%d  P=%.3f  R=%.3f  mAP=%.3f  (n=%d)",
        k, report.precision_at_k, report.recall_at_k, report.mean_ap, len(queries),
    )
    return report
