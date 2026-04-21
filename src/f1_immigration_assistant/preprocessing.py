"""Shared text preprocessing.

Exactly one preprocessing pipeline is used for both documents (at index time)
and queries (at retrieval time). Having one source of truth here guarantees
that the vocabulary used to build the index matches the vocabulary used to
query it.

The pipeline is intentionally simple and dependency-light:

1. lowercase
2. regex tokenize (words + digits, apostrophes preserved inside tokens)
3. drop very short tokens
4. drop English stop words (inline list, no nltk download)
5. optional Porter-style suffix trimming for a few common endings
"""

from __future__ import annotations

import re

from f1_immigration_assistant.config import DEFAULT_CONFIG, PreprocessingConfig

# Compact English stop-word list. Small on purpose; BM25 handles common words
# reasonably via IDF, so we only strip the most uninformative ones.
STOPWORDS: frozenset[str] = frozenset(
    """
    a an the and or but if while of at by for with about against between into through during
    before after above below to from up down in out on off over under again further then once
    here there when where why how all any both each few more most other some such no nor not
    only own same so than too very s t can will just don should now is are was were be been
    being have has had do does did doing this that these those i you he she it we they them
    their our my your his her its as also per etc e g ie eg ex
    """.split()
)

# Small suffix trimming table — keeps things predictable without a real stemmer.
_SUFFIXES = ("ingly", "edly", "ization", "isation", "ations", "ation", "ings", "ing", "ied", "ies", "ied", "ed", "es", "s", "ly")

_TOKEN_RE = re.compile(r"[A-Za-z][A-Za-z']*[A-Za-z]|[A-Za-z]|\d+")


def _trim_suffix(tok: str) -> str:
    """Very light suffix trimming. Returns the token unchanged if nothing fits."""

    for suf in _SUFFIXES:
        if len(tok) > len(suf) + 2 and tok.endswith(suf):
            return tok[: -len(suf)]
    return tok


def tokenize(text: str, cfg: PreprocessingConfig | None = None) -> list[str]:
    """Return the preprocessed token list for ``text``.

    Used for both indexing and querying — the single shared vocabulary path.
    """

    cfg = cfg or DEFAULT_CONFIG.preprocessing
    if not text:
        return []

    s = text.lower() if cfg.lowercase else text
    raw = _TOKEN_RE.findall(s)

    out: list[str] = []
    for tok in raw:
        if len(tok) < cfg.min_token_length:
            continue
        if cfg.remove_stopwords and tok in STOPWORDS:
            continue
        out.append(_trim_suffix(tok))
    return out


def normalize_whitespace(text: str) -> str:
    """Collapse runs of whitespace to a single space and strip the result."""

    return re.sub(r"\s+", " ", text or "").strip()
