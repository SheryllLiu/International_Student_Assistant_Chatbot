"""HTML -> SourceDocument.

Extracts a clean title, the ordered list of section headings, and the visible
body text for a single Georgetown OGS page. Navigation, scripts, styles, and
footer chrome are stripped so they do not pollute the BM25 vocabulary.
"""

from __future__ import annotations

import logging
from pathlib import Path

from bs4 import BeautifulSoup

from f1_immigration_assistant.models import SourceDocument
from f1_immigration_assistant.preprocessing import normalize_whitespace
from f1_immigration_assistant.utils import read_json

logger = logging.getLogger(__name__)

_STRIP_TAGS = ("script", "style", "nav", "header", "footer", "form", "noscript", "aside")


def parse_html(html: str, url: str, pdf_links: list[str] | None = None) -> SourceDocument:
    """Parse an HTML string into a :class:`SourceDocument`."""

    soup = BeautifulSoup(html, "html.parser")
    for tag in soup(_STRIP_TAGS):
        tag.decompose()

    main = soup.find("main") or soup.find("article") or soup.body or soup

    title_tag = soup.find("title")
    title = normalize_whitespace(title_tag.get_text()) if title_tag else ""
    if not title:
        h1 = main.find("h1")
        title = normalize_whitespace(h1.get_text()) if h1 else url

    headings = [
        normalize_whitespace(h.get_text())
        for h in main.find_all(["h1", "h2", "h3"])
        if h.get_text(strip=True)
    ]

    text = normalize_whitespace(main.get_text(separator=" "))

    return SourceDocument(
        url=url,
        title=title,
        headings=headings,
        text=text,
        pdf_links=list(pdf_links or []),
    )


def parse_raw_dir(raw_dir: Path | str) -> list[SourceDocument]:
    """Parse every ``<id>.html`` + ``<id>.json`` pair in ``raw_dir``."""

    raw_dir = Path(raw_dir)
    docs: list[SourceDocument] = []
    for html_path in sorted(raw_dir.glob("*.html")):
        meta_path = html_path.with_suffix(".json")
        meta = read_json(meta_path) if meta_path.exists() else {"url": html_path.stem}
        html = html_path.read_text(encoding="utf-8")
        docs.append(parse_html(html, url=meta.get("url", ""), pdf_links=meta.get("pdf_links", [])))
    logger.info("parsed %d documents from %s", len(docs), raw_dir)
    return docs
