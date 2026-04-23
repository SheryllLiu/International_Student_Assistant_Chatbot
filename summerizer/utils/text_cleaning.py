"""Text cleaning for the BM25 corpus and user queries.

Two entry points share one normalization pipeline:

* :func:`main` — batch: reads ``data/processed/structured_text.csv`` and writes
  ``data/processed/cleaned_data.csv`` with ``[topic, cleaned_document]``.
* :func:`clean_query_text` — import from Flask / retrieval code to apply the
  exact same rules to an incoming query.

Cleaning rules (identical for docs and queries):

1. lowercase
2. keep digits
3. drop punctuation, except hyphens (``-``)
4. drop English stopwords
5. collapse whitespace

No stemming, no lemmatization.

Run batch cleaning from the repo root::

    python utils/text_cleaning.py
"""
from __future__ import annotations

import re
from pathlib import Path

import pandas as pd

IN_FILE = Path("data/processed/structured_text.csv")
OUT_DIR = Path("data/processed")
OUT_FILE = OUT_DIR / "cleaned_data.csv"

TOPIC_COLUMN = "topic"
TEXT_COLUMN = "text"
CLEANED_COLUMN = "cleaned_document"

# English stopwords — vendored from the NLTK English list so this module has
# no runtime download step. If you prefer NLTK, swap this set for
# ``nltk.corpus.stopwords.words('english')`` after calling
# ``nltk.download('stopwords')``.
STOPWORDS: frozenset[str] = frozenset({
    "i", "me", "my", "myself", "we", "our", "ours", "ourselves", "you",
    "you're", "you've", "you'll", "you'd", "your", "yours", "yourself",
    "yourselves", "he", "him", "his", "himself", "she", "she's", "her",
    "hers", "herself", "it", "it's", "its", "itself", "they", "them",
    "their", "theirs", "themselves", "what", "which", "who", "whom", "this",
    "that", "that'll", "these", "those", "am", "is", "are", "was", "were",
    "be", "been", "being", "have", "has", "had", "having", "do", "does",
    "did", "doing", "a", "an", "the", "and", "but", "if", "or", "because",
    "as", "until", "while", "of", "at", "by", "for", "with", "about",
    "against", "between", "into", "through", "during", "before", "after",
    "above", "below", "to", "from", "up", "down", "in", "out", "on", "off",
    "over", "under", "again", "further", "then", "once", "here", "there",
    "when", "where", "why", "how", "all", "any", "both", "each", "few",
    "more", "most", "other", "some", "such", "no", "nor", "not", "only",
    "own", "same", "so", "than", "too", "very", "t", "can", "will",
    "just", "don", "don't", "should", "should've", "now", "d", "ll", "m",
    "o", "re", "ve", "y", "ain", "aren", "aren't", "couldn", "couldn't",
    "didn", "didn't", "doesn", "doesn't", "hadn", "hadn't", "hasn",
    "hasn't", "haven", "haven't", "isn", "isn't", "ma", "mightn",
    "mightn't", "mustn", "mustn't", "needn", "needn't", "shan", "shan't",
    "shouldn", "shouldn't", "wasn", "wasn't", "weren", "weren't", "won",
    "won't", "wouldn", "wouldn't",
})

# After lowercasing, anything that is not an ASCII letter, digit, whitespace,
# or hyphen becomes a space. Hyphens survive so tokens like ``f-1`` or
# ``sevp-certified`` stay intact.
_STRIP_PUNCT = re.compile(r"[^a-z0-9\s-]+")
_COLLAPSE_WS = re.compile(r"\s+")


def normalize_text(text: str) -> str:
    """Lowercase, drop punctuation (keep ``-``), collapse whitespace."""
    if not text:
        return ""
    text = text.lower()
    text = _STRIP_PUNCT.sub(" ", text)
    text = _COLLAPSE_WS.sub(" ", text)
    return text.strip()


def remove_stopwords(tokens: list[str]) -> list[str]:
    """Drop stopwords and tokens that collapse to pure hyphens."""
    return [
        tok for tok in tokens
        if tok and tok.strip("-") and tok not in STOPWORDS
    ]


def clean_text(text: str) -> str:
    """Shared core: normalize → tokenize → drop stopwords → rejoin.

    Used by both the corpus pipeline and :func:`clean_query_text` so the two
    paths are guaranteed identical.
    """
    normalized = normalize_text(text)
    if not normalized:
        return ""
    tokens = normalized.split(" ")
    tokens = remove_stopwords(tokens)
    return " ".join(tokens)


def clean_document_row(row: pd.Series, text_columns: list[str]) -> str:
    """Merge a row's non-topic columns into one string, then clean."""
    parts = []
    for col in text_columns:
        val = row.get(col, "")
        if isinstance(val, str) and val:
            parts.append(val)
    merged = " ".join(parts)
    return clean_text(merged)


def clean_query_text(query: str) -> str:
    """Clean a single user query with the exact rules used on the corpus.

    Returns a space-joined string of tokens. Call ``.split()`` downstream if
    you want a token list for BM25.
    """
    if not isinstance(query, str):
        return ""
    return clean_text(query)


def build_cleaned_corpus(df: pd.DataFrame) -> pd.DataFrame:
    """Return a DataFrame with ``[topic, cleaned_document]``."""
    if TOPIC_COLUMN not in df.columns:
        raise ValueError(f"input CSV must contain a '{TOPIC_COLUMN}' column")
    text_cols = [c for c in df.columns if c != TOPIC_COLUMN]
    cleaned = df.apply(lambda r: clean_document_row(r, text_cols), axis=1)
    return pd.DataFrame({
        TOPIC_COLUMN: df[TOPIC_COLUMN].fillna("").astype(str),
        CLEANED_COLUMN: cleaned,
    })


def main() -> None:
    if not IN_FILE.exists():
        raise FileNotFoundError(f"input not found: {IN_FILE}")
    # ``keep_default_na=False`` keeps empty topic/section cells as empty
    # strings instead of NaN, so ``clean_document_row`` can skip them cleanly.
    df = pd.read_csv(IN_FILE, keep_default_na=False)
    out_df = build_cleaned_corpus(df)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(OUT_FILE, index=False)
    print(f"[ok] wrote {len(out_df)} cleaned docs to {OUT_FILE}")


if __name__ == "__main__":
    main()
