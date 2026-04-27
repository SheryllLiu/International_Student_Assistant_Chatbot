"""Parse crawled Study-in-the-States HTML pages into a flat ``(topic, title, section, text)`` table."""

from __future__ import annotations

import re
from pathlib import Path
from urllib.parse import urlparse

import pandas as pd
from bs4 import BeautifulSoup, Tag

RAW_DIR = Path("data/raw")
OUT_DIR = Path("data/processed")
OUT_FILE = OUT_DIR / "structured_text.csv"

TARGET_TAGS = ("h2", "h3", "dt", "p", "li")
OUTPUT_COLUMNS = ["topic", "title", "section", "text"]

# Paragraphs that start with this prefix are taxonomy metadata injected by
# the Drupal template, not actual body content — drop them from text blocks.
RELATED_TAGS_PREFIX = "Related Tags:"


def read_html(path: Path) -> BeautifulSoup:
    """Read the file at ``path`` as HTML and return a parsed BeautifulSoup tree."""
    html = path.read_text(encoding="utf-8", errors="ignore")
    return BeautifulSoup(html, "lxml")


def extract_topic(soup: BeautifulSoup) -> str:
    """Pull the topic out of the canonical link.

    For a canonical href like ``.../students/work/obtaining-...`` the topic is
    ``work`` — the path segment immediately after ``students``. Returns an empty
    string when the link is missing or the URL doesn't match the expected shape.
    """
    link = soup.find("link", rel="canonical")
    if link is None:
        return ""
    href = link.get("href", "") or ""
    if not href:
        return ""
    try:
        path = urlparse(href).path
    except Exception:
        return ""
    parts = [p for p in path.split("/") if p]
    for i, seg in enumerate(parts):
        if seg == "students" and i + 1 < len(parts):
            return parts[i + 1]
    return ""


def clean_text(text: str) -> str:
    """Replace non-breaking spaces and collapse runs of whitespace."""
    if not text:
        return ""
    text = text.replace("\xa0", " ")
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def get_block_text(elem: Tag) -> str:
    """Return the cleaned visible text of an HTML element."""
    return clean_text(elem.get_text(" ", strip=True))


def has_ancestor(elem: Tag, tag_names: tuple[str, ...], stop_at: Tag) -> bool:
    """True if ``elem`` has an ancestor in ``tag_names`` before reaching ``stop_at``."""
    parent = elem.parent
    while parent is not None and parent is not stop_at:
        if getattr(parent, "name", None) in tag_names:
            return True
        parent = parent.parent
    return False


def parse_article(article: Tag) -> list[dict]:
    """Walk one ``<article>`` and yield rows keyed by title/section/text.

    State machine:

    - ``h2``  -> flush current block, start a new title, reset section.
    - ``h3``  -> flush current block, set the section to this h3's text,
                 enter "h3 mode" so subsequent ``dt``s merge into this block.
    - ``dt``  -> if inside h3 mode, the dt's own text is folded into the
                 current block; otherwise flush and start a new section.
                 A ``dt`` that wraps an ``h3`` is skipped (the inner h3
                 handles the boundary).
    - ``p`` / ``li`` -> append text to the current block.
    """
    rows: list[dict] = []

    current_title = ""
    current_section = ""
    current_parts: list[str] = []
    in_h3_mode = False
    title_seen = False

    def flush() -> None:
        nonlocal current_parts
        text = clean_text(" ".join(part for part in current_parts if part))
        if current_title and (text or current_section):
            rows.append(
                {
                    "title": current_title,
                    "section": current_section,
                    "text": text,
                }
            )
        current_parts = []

    for elem in article.find_all(TARGET_TAGS):
        name = elem.name

        if name == "h2":
            if title_seen:
                flush()
            current_title = get_block_text(elem)
            current_section = ""
            in_h3_mode = False
            title_seen = True
            continue

        if not title_seen:
            continue

        if name == "h3":
            flush()
            current_section = get_block_text(elem)
            in_h3_mode = True
            continue

        if name == "dt":
            # A dt that wraps an h3 is just structural noise; the nested
            # h3 will trigger the section boundary on its own iteration.
            if elem.find("h3") is not None:
                continue
            dt_text = get_block_text(elem)
            if in_h3_mode:
                if dt_text:
                    current_parts.append(dt_text)
            else:
                flush()
                current_section = dt_text
            continue

        # p or li: skip when nested inside another p/li to avoid duplication.
        if has_ancestor(elem, ("p", "li"), stop_at=article):
            continue
        text = get_block_text(elem)
        if not text:
            continue
        if text.startswith(RELATED_TAGS_PREFIX):
            continue
        current_parts.append(text)

    if title_seen:
        flush()

    return rows


def parse_file(path: Path) -> list[dict]:
    """Parse one HTML file and return its rows tagged with the page topic."""
    soup = read_html(path)
    topic = extract_topic(soup)
    article = soup.find("article")
    if article is None:
        return []
    rows = parse_article(article)
    for row in rows:
        row["topic"] = topic
    return rows


def collect_rows(raw_dir: Path) -> list[dict]:
    """Parse every ``.html`` file under ``raw_dir`` and concatenate their rows."""
    all_rows: list[dict] = []
    html_files = sorted(raw_dir.glob("*.html"))
    for html_path in html_files:
        try:
            rows = parse_file(html_path)
        except Exception as exc:
            print(f"[warn] failed to parse {html_path.name}: {exc}")
            continue
        all_rows.extend(rows)
    print(f"[info] parsed {len(html_files)} html files -> {len(all_rows)} rows")
    return all_rows


def main() -> None:
    """Parse every raw HTML page, drop duplicates, and write ``structured_text.csv``."""
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    rows = collect_rows(RAW_DIR)
    df = pd.DataFrame(rows, columns=OUTPUT_COLUMNS)
    before = len(df)
    df = df.drop_duplicates(subset=["text"]).reset_index(drop=True)
    if before != len(df):
        print(f"[info] dropped {before - len(df)} duplicate rows on 'text'")
    df.to_csv(OUT_FILE, index=False)
    print(f"[ok] wrote {len(df)} rows to {OUT_FILE}")


if __name__ == "__main__":
    main()
