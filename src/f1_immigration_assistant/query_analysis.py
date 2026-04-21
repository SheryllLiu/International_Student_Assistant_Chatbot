"""Lightweight query analysis.

We are deliberately not building a query-understanding framework. All we want
is a few cheap signals the pipeline can use: a small keyword list, one of four
intents (status / employment / travel / tax), and a short list of risk phrases
we want the answer to flag to the user (e.g. "overstayed", "dropped below
full-time", "worked off-campus without authorization").
"""

from __future__ import annotations

import re

from f1_immigration_assistant.models import QueryAnalysis
from f1_immigration_assistant.preprocessing import tokenize

INTENT_TERMS: dict[str, tuple[str, ...]] = {
    "employment": (
        "work", "job", "employment", "opt", "cpt", "stem",
        "internship", "authorization", "on-campus", "off-campus", "ead",
    ),
    "travel": (
        "travel", "reentry", "re-entry", "visa", "signature", "i-20",
        "abroad", "passport", "port", "entry",
    ),
    "tax": (
        "tax", "taxes", "1040", "1040nr", "irs", "sprintax", "ssn", "itin",
        "w-2", "1098", "1099", "refund", "filing",
    ),
    "status": (
        "status", "full-time", "enrollment", "credits", "reduced", "leave",
        "absence", "termination", "sevis", "grace", "maintain",
    ),
}

RISK_PHRASES: tuple[str, ...] = (
    "overstayed",
    "overstay",
    "out of status",
    "dropped below full-time",
    "dropped below full time",
    "worked off-campus",
    "worked off campus",
    "without authorization",
    "no authorization",
    "missed reporting",
    "sevis terminated",
    "visa expired",
    "i-20 expired",
)


def _detect_intent(lower: str, tokens: set[str]) -> str:
    """Return the intent whose term list matches ``tokens`` most, or 'unknown'."""

    scores = {intent: 0 for intent in INTENT_TERMS}
    for intent, terms in INTENT_TERMS.items():
        for term in terms:
            if " " in term or "-" in term:
                if term in lower:
                    scores[intent] += 1
            elif term in tokens:
                scores[intent] += 1
    best = max(scores, key=scores.get)
    return best if scores[best] > 0 else "unknown"


def analyze_query(query: str) -> QueryAnalysis:
    """Return a :class:`QueryAnalysis` for the user's query."""

    lower = (query or "").lower()
    tokens = tokenize(query)

    kw = [t for t in tokens if len(t) > 2]
    # De-duplicate while preserving order.
    seen: set[str] = set()
    keywords: list[str] = []
    for t in kw:
        if t not in seen:
            seen.add(t)
            keywords.append(t)

    intent = _detect_intent(lower, set(tokens))

    risks = [p for p in RISK_PHRASES if re.search(rf"\b{re.escape(p)}\b", lower)]

    return QueryAnalysis(query=query, keywords=keywords, intent=intent, risk_flags=risks)
