"""Parse raw crawled HTML into cleaner text documents.

Think of this file as the "cleaning" step of the pipeline:

1. Read raw HTML files from ``data/raw``.
2. Remove obvious website boilerplate.
3. Pull out the useful page content.
4. Save the cleaned result as JSON for the next step.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from urllib.parse import urlparse

from bs4 import BeautifulSoup
import re

try:
    from summerizer.utils.logger import write_log
except ModuleNotFoundError:
    from logger import write_log  # type: ignore

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = PROJECT_ROOT / "data" / "raw"

# These tags usually contain page chrome or technical markup instead of the
# main written content we want to keep.
TAGS_TO_REMOVE = (
    "script",
    "style",
    "nav",
    "header",
    "footer",
    "form",
    "noscript",
    "aside",
)

# These labels appear repeatedly in the site chrome and taxonomy sections.
# Removing them keeps the parsed text focused on the page's main content.
UNWANTED_HEADINGS = {
    "Breadcrumb",
    "Related Content",
    "Tools",
    "Social Media",
}

UNWANTED_TEXT_PHRASES = (
    "Breadcrumb Home Students",
    "Related Tags:",
    "Related Content",
    "Social Media",
    "Tools",
    "Tweets by StudyinStates",
)


@dataclass
class ParsedDocument:
    """A cleaned version of one crawled HTML page."""

    source_path: str
    url: str
    title: str
    headings: list[str]
    text: str


def normalize_whitespace(text: str) -> str:
    """Turn repeated whitespace into single spaces."""

    return re.sub(r"\s+", " ", text).strip()


def get_canonical_url(soup: BeautifulSoup, fallback: str = "") -> str:
    """Read the canonical URL from the page if it exists."""

    canonical = soup.find("link", rel="canonical")
    if canonical and canonical.get("href"):
        return canonical["href"].strip()
    return fallback


def clean_links(main_content: BeautifulSoup, page_url: str) -> None:
    """Clean up links inside the content area.

    We keep the visible text of useful links, but:
    - remove links with no text at all
    - unwrap off-site links so only their text remains
    """

    if not page_url:
        return

    page_host = urlparse(page_url).netloc.replace("www.", "")

    for link in main_content.find_all("a", href=True):
        href = link["href"].strip()
        link_text = link.get_text(strip=True)

        if not link_text:
            link.decompose()
            continue

        parsed = urlparse(href)
        if parsed.netloc and parsed.netloc.replace("www.", "") != page_host:
            link.unwrap()


def remove_empty_tags(main_content: BeautifulSoup) -> None:
    """Remove tags that are empty after cleaning."""

    for tag in main_content.find_all():
        if tag.name in {"br", "hr"}:
            continue
        if not tag.get_text(strip=True) and not tag.find(["img", "table"]):
            tag.decompose()


def choose_main_content(soup: BeautifulSoup) -> BeautifulSoup:
    """Pick the part of the page most likely to contain the real content."""

    main_content = soup.find("main")
    if main_content is not None:
        return main_content

    article = soup.find("article")
    if article is not None:
        return article

    if soup.body is not None:
        return soup.body

    return soup


def get_document_title(soup: BeautifulSoup, main_content: BeautifulSoup, source_path: str) -> str:
    """Choose a title for the document.

    We try, in order:
    1. the HTML <title>
    2. the first <h1>
    3. the file name
    """

    title_tag = soup.find("title")
    if title_tag:
        title_text = normalize_whitespace(title_tag.get_text())
        if title_text:
            return title_text

    h1_tag = main_content.find("h1")
    if h1_tag:
        h1_text = normalize_whitespace(h1_tag.get_text())
        if h1_text:
            return h1_text

    return Path(source_path).stem


def get_document_headings(main_content: BeautifulSoup) -> list[str]:
    """Collect headings in the order they appear."""

    headings: list[str] = []
    seen_headings: set[str] = set()

    for tag in main_content.find_all(["h1", "h2", "h3"]):
        heading_text = normalize_whitespace(tag.get_text())
        if heading_text in UNWANTED_HEADINGS:
            continue
        if heading_text and heading_text not in seen_headings:
            headings.append(heading_text)
            seen_headings.add(heading_text)

    return headings


def remove_unwanted_text_phrases(text: str) -> str:
    """Remove short repeated boilerplate phrases from parsed text."""

    cleaned_text = text
    for phrase in UNWANTED_TEXT_PHRASES:
        cleaned_text = cleaned_text.replace(phrase, " ")
    return normalize_whitespace(cleaned_text)


def parse_html(html: str, source_path: str = "") -> ParsedDocument:
    """Parse one raw HTML string into a cleaned document."""

    soup = BeautifulSoup(html, "html.parser")

    # Step 1: remove tags that almost never hold the main page content.
    for tag in soup(TAGS_TO_REMOVE):
        tag.decompose()

    # Step 2: decide which part of the page we want to treat as the main body.
    main_content = choose_main_content(soup)
    page_url = get_canonical_url(soup)

    # Step 3: clean smaller pieces inside the chosen content area.
    clean_links(main_content, page_url)
    remove_empty_tags(main_content)

    # Step 4: pull out the structured information we care about.
    title = get_document_title(soup, main_content, source_path)
    headings = get_document_headings(main_content)
    text = normalize_whitespace(main_content.get_text(separator=" "))
    text = remove_unwanted_text_phrases(text)

    return ParsedDocument(
        source_path=source_path,
        url=page_url,
        title=title,
        headings=headings,
        text=text,
    )


def parse_file(path: Path | str) -> ParsedDocument:
    """Parse one HTML file from disk."""

    path = Path(path)
    html = path.read_text(encoding="utf-8")
    return parse_html(html, source_path=str(path))


def parse_raw_dir(raw_dir: Path | str = RAW_DIR) -> list[ParsedDocument]:
    """Parse every HTML file in a raw crawl directory.

    We deduplicate by canonical URL. That means if two raw HTML files point to
    the same underlying page, we keep only one cleaned document.
    """

    raw_dir = Path(raw_dir)
    documents: list[ParsedDocument] = []
    seen_urls: set[str] = set()

    for html_path in sorted(raw_dir.glob("*.html")):
        document = parse_file(html_path)
        if document.url and document.url in seen_urls:
            continue
        if document.url:
            seen_urls.add(document.url)
        documents.append(document)

    return documents


def write_parsed_json(
    out_path: Path | str,
    raw_dir: Path | str = RAW_DIR,
) -> Path:
    """Parse a raw directory and save the cleaned documents as JSON."""

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    write_log("parser.py started")
    parsed_documents = parse_raw_dir(raw_dir)
    json_ready_documents = [asdict(document) for document in parsed_documents]

    out_path.write_text(
        json.dumps(json_ready_documents, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    write_log(f"parser.py parsed {len(parsed_documents)} documents into {out_path}")
    return out_path


if __name__ == "__main__":
    output = write_parsed_json(PROJECT_ROOT / "data" / "parsed" / "parsed_docs.json", RAW_DIR)
    print(f"wrote parsed documents to {output}")
