"""BM25 retrieval over document chunks.

This is the main retrieval component of the project. We wrap ``rank-bm25``'s
``BM25Okapi`` so the rest of the code deals with :class:`DocumentChunk` and
:class:`RetrievalResult` objects instead of integer positions.

The same preprocessing pipeline (:mod:`f1_immigration_assistant.preprocessing`)
is used at index time and query time — there is only ever one vocabulary.
"""

from __future__ import annotations

import logging
import pickle
from dataclasses import dataclass
from pathlib import Path

from rank_bm25 import BM25Okapi

from f1_immigration_assistant.config import DEFAULT_CONFIG, RetrievalConfig
from f1_immigration_assistant.models import DocumentChunk, RetrievalResult
from f1_immigration_assistant.preprocessing import tokenize

logger = logging.getLogger(__name__)

_INDEX_FILENAME = "bm25_index.pkl"


@dataclass
class _Index:
    """Internal pickle-friendly bundle of the BM25 state."""

    chunks: list[DocumentChunk]
    tokenized: list[list[str]]
    bm25: BM25Okapi


class BM25Retriever:
    """BM25 retriever over a fixed set of :class:`DocumentChunk` objects."""

    def __init__(self, cfg: RetrievalConfig | None = None) -> None:
        """Create an empty retriever. Call :meth:`fit` or :meth:`load` next."""

        self.cfg = cfg or DEFAULT_CONFIG.retrieval
        self._index: _Index | None = None

    # ----- construction -----------------------------------------------------

    def fit(self, chunks: list[DocumentChunk]) -> "BM25Retriever":
        """Build the BM25 index from a list of chunks."""

        if not chunks:
            raise ValueError("BM25Retriever.fit requires at least one chunk")
        tokenized = [tokenize(c.text) for c in chunks]
        bm25 = BM25Okapi(tokenized, k1=self.cfg.bm25_k1, b=self.cfg.bm25_b)
        self._index = _Index(chunks=chunks, tokenized=tokenized, bm25=bm25)
        logger.info("BM25 index built over %d chunks", len(chunks))
        return self

    # ----- persistence ------------------------------------------------------

    def save(self, index_dir: Path | str) -> Path:
        """Persist the index to ``<index_dir>/bm25_index.pkl``."""

        if self._index is None:
            raise RuntimeError("nothing to save — call fit() first")
        index_dir = Path(index_dir)
        index_dir.mkdir(parents=True, exist_ok=True)
        path = index_dir / _INDEX_FILENAME
        with path.open("wb") as f:
            pickle.dump(self._index, f)
        logger.info("BM25 index saved to %s", path)
        return path

    def load(self, index_dir: Path | str) -> "BM25Retriever":
        """Load a previously saved index from ``<index_dir>``."""

        path = Path(index_dir) / _INDEX_FILENAME
        with path.open("rb") as f:
            self._index = pickle.load(f)
        logger.info("BM25 index loaded from %s (%d chunks)", path, len(self._index.chunks))
        return self

    # ----- query ------------------------------------------------------------

    @property
    def chunks(self) -> list[DocumentChunk]:
        """The chunks this retriever was fit on."""

        if self._index is None:
            raise RuntimeError("retriever is not fit/loaded")
        return self._index.chunks

    def search(self, query: str, top_k: int | None = None) -> list[RetrievalResult]:
        """Return the ``top_k`` highest-BM25-scoring chunks for ``query``."""

        if self._index is None:
            raise RuntimeError("retriever is not fit/loaded")
        k = top_k or self.cfg.top_k

        tokens = tokenize(query)
        if not tokens:
            return []

        scores = self._index.bm25.get_scores(tokens)
        ranked = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)[:k]
        return [
            RetrievalResult(chunk=self._index.chunks[i], score=float(s))
            for i, s in ranked
            if s > 0
        ]
