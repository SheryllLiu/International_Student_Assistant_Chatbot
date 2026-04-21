"""Lightweight data models used across the package.

Plain dataclasses — no pydantic, no inheritance trees. Each model carries only
the fields we actually use; add fields only when a real caller needs them.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class SourceDocument:
    """A parsed Georgetown OGS page."""

    url: str
    title: str
    headings: list[str]
    text: str
    pdf_links: list[str] = field(default_factory=list)


@dataclass
class DocumentChunk:
    """A retrieval-sized slice of a SourceDocument."""

    chunk_id: str
    url: str
    title: str
    heading: str
    text: str


@dataclass
class QueryAnalysis:
    """Lightweight hints extracted from a user query."""

    query: str
    keywords: list[str]
    intent: str  # one of: status, employment, travel, tax, unknown
    risk_flags: list[str] = field(default_factory=list)


@dataclass
class RetrievalResult:
    """A single retrieved chunk with its BM25 score."""

    chunk: DocumentChunk
    score: float


@dataclass
class AnswerResult:
    """The final grounded answer returned by the pipeline."""

    query: str
    answer: str
    citations: list[str]  # source URLs, deduplicated, top-first
    risk_flags: list[str] = field(default_factory=list)
    warning: str | None = None
    used_llm: bool = False  # True when OpenAI was called; False for the stub
