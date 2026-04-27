"""Polite seed-driven crawler for ``studyinthestates.dhs.gov`` HTML pages.

Walks each seed BFS-style under a path prefix, respects ``robots.txt``, and
deduplicates pages by canonical URL and article-body content hash before
saving raw HTML to ``data/raw``.
"""

from __future__ import annotations

import contextlib
import hashlib
import logging
import posixpath
import re
import time
from collections import deque
from pathlib import Path
from urllib.parse import urldefrag, urljoin, urlparse, urlunparse
from urllib.robotparser import RobotFileParser

import requests
from bs4 import BeautifulSoup

logger = logging.getLogger("isa.crawler")

RAW_DIR = Path("data/raw")
RAW_DIR.mkdir(parents=True, exist_ok=True)

# Define seed URLs for crawling
SEEDS = [
    {
        "url": "https://studyinthestates.dhs.gov/students/",
        "source_authority": "gov",
        "institution": "DHS/SEVP",
        "path_prefix": "/students/",
        "institution_specific": False,
    }
]

# Specify crawling parameters
MAX_DEPTH = 8
MAX_PAGES_PER_SEED = 80
DELAY_SEC = 1.5
TIMEOUT = 30

HEADERS = {
    "User-Agent": "RAG-research-crawler/0.1 (contact: your-email@example.com)",
    "Accept": "text/html,application/xhtml+xml",
}

SKIP_EXT = {
    ".pdf",
    ".jpg",
    ".jpeg",
    ".png",
    ".gif",
    ".svg",
    ".css",
    ".js",
    ".zip",
    ".mp4",
    ".mp3",
    ".doc",
    ".docx",
    ".xls",
    ".xlsx",
    ".ppt",
    ".pptx",
    ".ico",
    ".woff",
    ".woff2",
    ".ttf",
}


def normalize_url(url: str) -> str:
    """Return a canonical form of ``url`` for stable dedup keys.

    Drops the fragment, forces https, lowercases the host, strips ``www.`` and
    default ports, normalizes the path, and removes a trailing slash.
    """
    url, _ = urldefrag(url)
    p = urlparse(url)
    scheme = "https" if p.scheme in ("http", "https", "") else p.scheme.lower()
    host = (p.hostname or "").lower()
    if host.startswith("www."):
        host = host[4:]
    port = p.port
    if port and not ((scheme == "http" and port == 80) or (scheme == "https" and port == 443)):
        netloc = f"{host}:{port}"
    else:
        netloc = host
    path = p.path or "/"
    path = posixpath.normpath(path)

    if not path.startswith("/"):
        path = "/" + path
    if path != "/":
        path = path.rstrip("/")
    return urlunparse((scheme, netloc, path, "", "", ""))


def slugify(url: str) -> str:
    """Build a deterministic ``host__path__hash`` filename slug for a URL."""
    url = normalize_url(url)
    p = urlparse(url)
    host = p.netloc.replace("www.", "")
    path = (p.path.strip("/").replace("/", "_") or "index")[:100]
    h = hashlib.sha1(url.encode()).hexdigest()[:8]
    return f"{host}__{path}__{h}"


def is_crawlable(url: str, seed_host: str, path_prefix: str | None) -> bool:
    """True if ``url`` is on ``seed_host``, under ``path_prefix``, and not a binary asset."""
    p = urlparse(url)
    if p.scheme not in ("http", "https"):
        return False
    if p.netloc.replace("www.", "") != seed_host.replace("www.", ""):  # make sure same host as seed
        return False
    if path_prefix:  # crawl only under the specified path prefix
        prefix = path_prefix.rstrip("/")
        if not (p.path == prefix or p.path.startswith(prefix + "/")):
            return False
    return not any(p.path.lower().endswith(ext) for ext in SKIP_EXT)


def fetch(url: str):
    """GET ``url`` and return the response if it is HTML; return None on error."""
    try:
        r = requests.get(url, headers=HEADERS, timeout=TIMEOUT)
        r.raise_for_status()
        if "text/html" not in r.headers.get("Content-Type", ""):
            return None
        return r
    except requests.RequestException as e:
        logger.warning("fetch failed %s: %s", url, e)
        return None


def extract_links(html: str, base_url: str) -> list[str]:
    """Return deduped, normalized absolute links from ``html`` (skips js/mailto/tel/#)."""
    soup = BeautifulSoup(html, "html.parser")
    out = set()
    for a in soup.find_all("a", href=True):
        href = a["href"].strip()
        if not href or href.startswith(("javascript:", "mailto:", "tel:", "#")):
            continue
        out.add(normalize_url(urljoin(base_url, href)))
    return list(out)


def _canonical_url(soup: BeautifulSoup) -> str | None:
    """Return the page's <link rel="canonical"> href, or None."""
    link = soup.find("link", rel="canonical")
    if link is None:
        return None
    href = (link.get("href") or "").strip()
    return href or None


def _article_hash(soup: BeautifulSoup) -> str | None:
    """SHA-256 of the <article> visible text, whitespace-collapsed.

    Hashing only the article body means cosmetic differences in nav / analytics
    markup don't defeat dedup for pages whose real content is identical.
    """
    article = soup.find("article")
    if article is None:
        return None
    text = re.sub(r"\s+", " ", article.get_text(" ", strip=True)).strip()
    return hashlib.sha256(text.encode("utf-8")).hexdigest() if text else None


def save_page(url: str, html: str, meta_extra: dict) -> None:
    """Write the raw HTML for ``url`` into ``RAW_DIR`` under its slug filename."""
    url = normalize_url(url)
    slug = slugify(url)
    (RAW_DIR / f"{slug}.html").write_text(html, encoding="utf-8")


def get_robots(seed_url: str) -> RobotFileParser:
    """Fetch and parse the host's ``robots.txt`` (returns an empty parser on failure)."""
    p = urlparse(seed_url)
    rp = RobotFileParser()
    rp.set_url(f"{p.scheme}://{p.netloc}/robots.txt")
    with contextlib.suppress(Exception):
        rp.read()
    return rp


def crawl_seed(seed: dict) -> None:
    """BFS-crawl one seed under its path prefix, deduping by canonical URL and body hash."""
    seed_url = normalize_url(seed["url"])
    seed_host = urlparse(seed_url).netloc
    path_prefix = seed.get("path_prefix")
    rp = get_robots(seed_url)
    ua = HEADERS["User-Agent"]

    visited: set[str] = set()
    queue: deque = deque([(seed_url, 0, None)])
    saved = 0
    # Layer-1: canonical URL dedup (first-win).
    seen_canonicals: dict[str, str] = {}
    # Layer-2: article content-hash dedup (first-win).
    seen_hashes: dict[str, str] = {}

    logger.info(
        "crawling %s  depth<=%d  max_pages=%d  prefix=%s",
        seed_url,
        MAX_DEPTH,
        MAX_PAGES_PER_SEED,
        path_prefix,
    )

    while queue and saved < MAX_PAGES_PER_SEED:
        url, depth, parent = queue.popleft()
        if url in visited:
            continue
        visited.add(url)

        if not is_crawlable(url, seed_host, path_prefix):
            continue
        if not rp.can_fetch(ua, url):
            logger.warning("robots-blocked: %s", url)
            continue

        logger.debug("[d=%d] %s", depth, url)
        resp = fetch(url)
        if resp is None:
            continue

        # --- dedup checks (parse once, reuse soup) ---
        soup = BeautifulSoup(resp.text, "html.parser")

        canonical = _canonical_url(soup)
        if canonical and canonical in seen_canonicals:
            logger.debug("skip duplicate (canonical): %s == %s", url, seen_canonicals[canonical])
            time.sleep(DELAY_SEC)
            continue

        body_hash = _article_hash(soup)
        if body_hash and body_hash in seen_hashes:
            logger.debug("skip duplicate (content hash): %s == %s", url, seen_hashes[body_hash])
            time.sleep(DELAY_SEC)
            continue
        # --- end dedup ---

        save_page(
            url,
            resp.text,
            {
                "depth": depth,
                "parent_url": parent,
                "seed_url": seed_url,
                "status_code": resp.status_code,
                "content_type": resp.headers.get("Content-Type"),
                "last_modified": resp.headers.get("Last-Modified"),
                "etag": resp.headers.get("ETag"),
                "content_length": len(resp.text),
                "source_authority": seed.get("source_authority"),
                "institution": seed.get("institution"),
                "institution_specific": seed.get("institution_specific", False),
            },
        )
        saved += 1
        logger.info("saved [%d/%d] %s", saved, MAX_PAGES_PER_SEED, url)
        if canonical:
            seen_canonicals[canonical] = url
        if body_hash:
            seen_hashes[body_hash] = url

        if depth < MAX_DEPTH:
            for link in extract_links(resp.text, url):
                if link not in visited:
                    queue.append((link, depth + 1, url))

        time.sleep(DELAY_SEC)

    logger.info("finished seed %s — saved=%d, queued_unvisited=%d", seed_url, saved, len(queue))


def main():
    """Crawl every entry in :data:`SEEDS`."""
    for seed in SEEDS:
        crawl_seed(seed)


if __name__ == "__main__":
    main()
