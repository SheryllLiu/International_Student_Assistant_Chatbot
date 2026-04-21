"""Deterministic, heading-aware chunking.

The parser gives us one big block of text per page plus the list of headings.
We split on those headings when possible (so each chunk carries a meaningful
section title), then greedy-pack the section text into ~``target_tokens``
pieces with a small overlap. If no headings are present we just pack the text.
"""

from __future__ import annotations

import logging
import re

from f1_immigration_assistant.config import DEFAULT_CONFIG, ChunkingConfig
from f1_immigration_assistant.models import DocumentChunk, SourceDocument
from f1_immigration_assistant.utils import stable_id

logger = logging.getLogger(__name__)


def _split_by_headings(text: str, headings: list[str]) -> list[tuple[str, str]]:
    """Return ``[(heading, body)]`` splits of ``text`` on known heading strings.

    If a heading does not appear verbatim in ``text`` it is skipped. The first
    block (before the first matched heading) is returned under the empty
    heading ``""``.
    """

    positions: list[tuple[int, str]] = []
    for h in headings:
        if not h:
            continue
        m = re.search(re.escape(h), text)
        if m:
            positions.append((m.start(), h))
    positions.sort()
    if not positions:
        return [("", text)]

    out: list[tuple[str, str]] = []
    first = positions[0][0]
    if first > 0:
        out.append(("", text[:first].strip()))
    for i, (start, heading) in enumerate(positions):
        end = positions[i + 1][0] if i + 1 < len(positions) else len(text)
        body = text[start + len(heading) : end].strip()
        if body:
            out.append((heading, body))
    return out


def _pack_tokens(tokens: list[str], cfg: ChunkingConfig) -> list[str]:
    """Greedy pack a token list into target-sized windows with overlap."""

    if not tokens:
        return []
    windows: list[str] = []
    step = max(1, cfg.target_tokens - cfg.overlap_tokens)
    i = 0
    while i < len(tokens):
        window = tokens[i : i + cfg.target_tokens]
        if len(window) < cfg.min_tokens and windows:
            # Too small to stand alone — merge the remainder into the previous chunk.
            windows[-1] = windows[-1] + " " + " ".join(window)
            break
        windows.append(" ".join(window))
        i += step
    return windows


def chunk_document(doc: SourceDocument, cfg: ChunkingConfig | None = None) -> list[DocumentChunk]:
    """Split one :class:`SourceDocument` into retrieval chunks."""

    cfg = cfg or DEFAULT_CONFIG.chunking
    sections = _split_by_headings(doc.text, doc.headings)

    chunks: list[DocumentChunk] = []
    for heading, body in sections:
        tokens = body.split()
        for j, window in enumerate(_pack_tokens(tokens, cfg)):
            chunk_id = stable_id(doc.url, heading, str(j))
            chunks.append(
                DocumentChunk(
                    chunk_id=chunk_id,
                    url=doc.url,
                    title=doc.title,
                    heading=heading or doc.title,
                    text=window,
                )
            )
    return chunks


def chunk_documents(
    docs: list[SourceDocument], cfg: ChunkingConfig | None = None
) -> list[DocumentChunk]:
    """Chunk a list of documents and return the flattened chunk list."""

    all_chunks: list[DocumentChunk] = []
    for doc in docs:
        all_chunks.extend(chunk_document(doc, cfg))
    logger.info("chunked %d documents into %d chunks", len(docs), len(all_chunks))
    return all_chunks
