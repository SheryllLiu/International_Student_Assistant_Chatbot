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
6. WordNet lemmatize (verb → noun fallback; hyphenated / numeric tokens
   are left untouched so ``f-1``, ``on-campus``, ``i-20`` survive intact)

Run batch cleaning from the repo root::

    python utils/text_cleaning.py
"""

from __future__ import annotations

import re
from functools import lru_cache
from pathlib import Path

import pandas as pd
from nltk.stem import WordNetLemmatizer


def _ensure_wordnet() -> None:
    """Make sure the WordNet corpus is available; fetch it if missing."""
    try:
        from nltk.corpus import wordnet

        wordnet.ensure_loaded()
    except LookupError:
        import nltk

        nltk.download("wordnet", quiet=True)
        nltk.download("omw-1.4", quiet=True)


_ensure_wordnet()
_LEMMATIZER = WordNetLemmatizer()

IN_FILE = Path("data/processed/structured_text.csv")
OUT_DIR = Path("data/processed")
OUT_FILE = OUT_DIR / "cleaned_data.csv"

TOPIC_COLUMN = "topic"
TEXT_COLUMN = "text"
CLEANED_COLUMN = "cleaned_document"
# Raw text columns to carry through unchanged so the retriever can display them.
RAW_TEXT_COLUMNS = ("title", "section", "text")

# English stopwords — vendored from the NLTK English list so this module has
# no runtime download step. If you prefer NLTK, swap this set for
# ``nltk.corpus.stopwords.words('english')`` after calling
# ``nltk.download('stopwords')``.
STOPWORDS: frozenset[str] = frozenset(
    {
        "i",
        "me",
        "my",
        "myself",
        "we",
        "our",
        "ours",
        "ourselves",
        "you",
        "you're",
        "you've",
        "you'll",
        "you'd",
        "your",
        "yours",
        "yourself",
        "yourselves",
        "he",
        "him",
        "his",
        "himself",
        "she",
        "she's",
        "her",
        "hers",
        "herself",
        "it",
        "it's",
        "its",
        "itself",
        "they",
        "them",
        "their",
        "theirs",
        "themselves",
        "what",
        "which",
        "who",
        "whom",
        "this",
        "that",
        "that'll",
        "these",
        "those",
        "am",
        "is",
        "are",
        "was",
        "were",
        "be",
        "been",
        "being",
        "have",
        "has",
        "had",
        "having",
        "do",
        "does",
        "did",
        "doing",
        "a",
        "an",
        "the",
        "and",
        "but",
        "if",
        "or",
        "because",
        "as",
        "until",
        "while",
        "of",
        "at",
        "by",
        "for",
        "with",
        "about",
        "against",
        "between",
        "into",
        "through",
        "during",
        "before",
        "after",
        "above",
        "below",
        "to",
        "from",
        "up",
        "down",
        "in",
        "out",
        "on",
        "off",
        "over",
        "under",
        "again",
        "further",
        "then",
        "once",
        "here",
        "there",
        "when",
        "where",
        "why",
        "how",
        "all",
        "any",
        "both",
        "each",
        "few",
        "more",
        "most",
        "other",
        "some",
        "such",
        "no",
        "nor",
        "not",
        "only",
        "own",
        "same",
        "so",
        "than",
        "too",
        "very",
        "t",
        "can",
        "will",
        "just",
        "don",
        "don't",
        "should",
        "should've",
        "now",
        "d",
        "ll",
        "m",
        "o",
        "re",
        "ve",
        "y",
        "ain",
        "aren",
        "aren't",
        "couldn",
        "couldn't",
        "didn",
        "didn't",
        "doesn",
        "doesn't",
        "hadn",
        "hadn't",
        "hasn",
        "hasn't",
        "haven",
        "haven't",
        "isn",
        "isn't",
        "ma",
        "mightn",
        "mightn't",
        "mustn",
        "mustn't",
        "needn",
        "needn't",
        "shan",
        "shan't",
        "shouldn",
        "shouldn't",
        "wasn",
        "wasn't",
        "weren",
        "weren't",
        "won",
        "won't",
        "wouldn",
        "wouldn't",
    }
)

# After lowercasing, anything that is not an ASCII letter, digit, whitespace,
# or hyphen becomes a space. Hyphens survive so tokens like ``f-1`` or
# ``sevp-certified`` stay intact.
_STRIP_PUNCT = re.compile(r"[^a-z0-9\s-]+")
_COLLAPSE_WS = re.compile(r"\s+")

# ---------------------------------------------------------------------------
# Query-side term normalization
# ---------------------------------------------------------------------------
# Maps what a user might type → the canonical form present in the corpus.
#
# Section A (up to "sevp-certified"): unhyphenated user variants → hyphenated
#   corpus tokens (e.g. "f1" → "f-1", "on campus" → "on-campus").
# Section B: common abbreviations → the full phrases indexed in the corpus
#   (e.g. "opt" → "optional practical training").
#
# Patterns are applied longest-first so "my e verify" is consumed before
# the shorter "e verify" key can fire on the same span.
QUERY_TERM_NORMALIZATION: dict[str, str] = {
    # --- Section A: visa / status codes ---
    "f1": "f-1",
    "f 1": "f-1",
    "f2": "f-2",
    "f 2": "f-2",
    "m1": "m-1",
    "m 1": "m-1",
    "m2": "m-2",
    "m 2": "m-2",
    "j1": "j-1",
    "j 1": "j-1",
    "j2": "j-2",
    "j 2": "j-2",
    "b1": "b-1",
    "b 1": "b-1",
    "b2": "b-2",
    "b 2": "b-2",
    "h1b": "h-1b",
    "h 1b": "h-1b",
    "h 1 b": "h-1b",
    "h1-b": "h-1b",
    # --- Section A: USCIS / DHS form numbers ---
    "i20": "i-20",
    "i 20": "i-20",
    "i94": "i-94",
    "i 94": "i-94",
    "i901": "i-901",
    "i 901": "i-901",
    "i515": "i-515",
    "i 515": "i-515",
    "i515a": "i-515a",
    "i 515a": "i-515a",
    "i539": "i-539",
    "i 539": "i-539",
    "i765": "i-765",
    "i 765": "i-765",
    "i766": "i-766",
    "i 766": "i-766",
    "i797": "i-797",
    "i 797": "i-797",
    "i797a": "i-797a",
    "i 797a": "i-797a",
    "i129": "i-129",
    "i 129": "i-129",
    "i290b": "i-290b",
    "i 290b": "i-290b",
    "i983": "i-983",
    "i 983": "i-983",
    "i9": "i-9",
    "i 9": "i-9",
    "i17": "i-17",
    "i 17": "i-17",
    "w7": "w-7",
    "w 7": "w-7",
    "ds2019": "ds-2019",
    "ds 2019": "ds-2019",
    "ssa l676": "ssa-l676",
    # --- Section A: compound terms (space-separated → hyphenated) ---
    "cap gap": "cap-gap",
    "pre completion": "pre-completion",
    "post completion": "post-completion",
    "on campus": "on-campus",
    "off campus": "off-campus",
    "post secondary": "post-secondary",
    "full time": "full-time",
    "part time": "part-time",
    "up to date": "up-to-date",
    "my e verify": "mye-verify",
    "e verify": "e-verify",
    "everify": "e-verify",
    "re enter": "re-enter",
    "re entry": "re-entry",
    "transfer in": "transfer-in",
    "transfer out": "transfer-out",
    "k 12": "k-12",
    "sevp certified": "sevp-certified",
    # --- Section B: abbreviations → full phrases used in the corpus ---
    "ssn": "social security number",
    "dso": "designated school official",
    "dsos": "designated school officials",
    "sevis": "student exchange visitor information system",
    "sevp": "student exchange visitor program",
    "uscis": "united states citizenship immigration services",
    "cbp": "customs border protection",
    "dhs": "department homeland security",
    "irs": "internal revenue service",
    "itin": "individual taxpayer identification number",
    "opt": "optional practical training",
    "cpt": "curricular practical training",
    "ead": "employment authorization document",
    "ssa": "social security administration",
    "dmv": "department motor vehicles",
}

# Pre-compiled patterns (longest source phrase first) with hyphen-aware word
# boundaries so "ssa" does not fire inside "ssa-l676".
_NORM_PATTERNS: list[tuple[re.Pattern[str], str]] = [
    (
        re.compile(r"(?<![a-zA-Z0-9-])" + re.escape(src) + r"(?![a-zA-Z0-9-])"),
        dst,
    )
    for src, dst in sorted(QUERY_TERM_NORMALIZATION.items(), key=lambda kv: -len(kv[0]))
]


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
    return [tok for tok in tokens if tok and tok.strip("-") and tok not in STOPWORDS]


@lru_cache(maxsize=50_000)
def lemmatize_token(tok: str) -> str:
    """WordNet-lemmatize a single token; leave hyphenated/numeric tokens alone.

    Tries verb POS first, then noun. Verb-first handles ``working -> work`` and
    ``studies -> study``; noun fallback handles ``students -> student`` and
    ``mice -> mouse``. Hyphenated tokens (``f-1``, ``on-campus``) and any
    token containing a digit are passed through unchanged so domain-canonical
    forms don't get mangled.
    """
    if not tok or "-" in tok or any(ch.isdigit() for ch in tok):
        return tok
    lemma = _LEMMATIZER.lemmatize(tok, pos="v")
    if lemma == tok:
        lemma = _LEMMATIZER.lemmatize(tok, pos="n")
    return lemma


def lemmatize_tokens(tokens: list[str]) -> list[str]:
    """Apply :func:`lemmatize_token` to every token in ``tokens``."""
    return [lemmatize_token(t) for t in tokens]


def clean_text(text: str) -> str:
    """Shared core: normalize → tokenize → drop stopwords → lemmatize → rejoin.

    Used by both the corpus pipeline and :func:`clean_query_text` so the two
    paths are guaranteed identical. Lemmatization runs last so stopword
    removal still sees the original surface forms (e.g. ``doing`` is dropped
    as a stopword instead of being collapsed to ``do`` first).
    """
    normalized = normalize_text(text)
    if not normalized:
        return ""
    tokens = normalized.split(" ")
    tokens = remove_stopwords(tokens)
    tokens = lemmatize_tokens(tokens)
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


def normalize_query_terms(text: str) -> str:
    """Normalize immigration-domain terminology before the main cleaning step.

    Lowercases ``text`` then applies :data:`QUERY_TERM_NORMALIZATION` patterns
    (longest-first) so that:

    * Unhyphenated variants become the hyphenated corpus tokens
      (``f1`` → ``f-1``, ``on campus`` → ``on-campus``).
    * Common abbreviations expand to the full phrase in the index
      (``opt`` → ``optional practical training``).

    Hyphens are used as word-boundary guards so ``ssa`` never fires inside
    ``ssa-l676``.
    """
    text = text.lower()
    for pattern, replacement in _NORM_PATTERNS:
        text = pattern.sub(replacement, text)
    return text


def clean_query_text(query: str) -> str:
    """Normalize terms then clean with the same rules used on the corpus.

    Returns a space-joined string of tokens. Call ``.split()`` downstream if
    you want a token list for BM25.
    """
    if not isinstance(query, str):
        return ""
    return clean_text(normalize_query_terms(query))


def build_cleaned_corpus(df: pd.DataFrame) -> pd.DataFrame:
    """Return a DataFrame with cleaned and original text columns.

    Output columns (in order):
    - ``topic``            — unchanged
    - ``title``            — raw, if present in input
    - ``section``          — raw, if present in input
    - ``text``             — raw, if present in input
    - ``raw_document``     — title + section + text joined by newlines (non-empty parts only)
    - ``cleaned_document`` — normalized text used for BM25 indexing
    """
    if TOPIC_COLUMN not in df.columns:
        raise ValueError(f"input CSV must contain a '{TOPIC_COLUMN}' column")
    text_cols = [c for c in df.columns if c != TOPIC_COLUMN]
    cleaned = df.apply(lambda r: clean_document_row(r, text_cols), axis=1)

    present_raw = [c for c in RAW_TEXT_COLUMNS if c in df.columns]

    out = pd.DataFrame({TOPIC_COLUMN: df[TOPIC_COLUMN].fillna("").astype(str)})

    for col in present_raw:
        out[col] = df[col].fillna("").astype(str)

    if present_raw:
        out["raw_document"] = df.apply(
            lambda r: "\n".join(str(r[c]) for c in present_raw if str(r[c]).strip()),
            axis=1,
        )

    out[CLEANED_COLUMN] = cleaned
    return out


def main() -> None:
    """Build the cleaned corpus from ``structured_text.csv`` and write it to disk."""
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
