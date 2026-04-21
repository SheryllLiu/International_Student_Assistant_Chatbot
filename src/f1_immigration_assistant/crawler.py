"""Constrained crawler for approved Georgetown OGS pages.

This is not a general web crawler. It fetches only the URLs on the allowlist
in :mod:`f1_immigration_assistant.config`, applies an explicit denylist on top,
and saves each response as HTML plus a small JSON sidecar recording the URL
and any linked PDFs discovered on the page.
"""

from __future__ import annotations

import gzip
import logging
import time
from pathlib import Path
from typing import Iterable
from urllib.parse import urljoin, urlparse
from xml.etree import ElementTree as ET

import requests
from bs4 import BeautifulSoup

from f1_immigration_assistant.config import (
    ALLOWED_DOMAIN,
    ALLOWED_URLS,
    DENY_PATTERNS,
    ROBOTS_TXT_URL,
    SITEMAP_CANDIDATES,
    SITEMAP_DEPTH_LIMIT,
)
from f1_immigration_assistant.utils import stable_id, write_json

logger = logging.getLogger(__name__)

USER_AGENT = "f1-immigration-assistant/0.1 (course project)"
REQUEST_TIMEOUT = 20
DELAY_SECONDS = 1.0


def is_allowed(url: str) -> bool:
    """Return True if ``url`` is on the allowlist and passes the denylist."""

    if url not in ALLOWED_URLS:
        return False
    parsed = urlparse(url)
    if parsed.netloc != ALLOWED_DOMAIN:
        return False
    lower = url.lower()
    return not any(pat in lower for pat in DENY_PATTERNS)


def _extract_pdf_links(html: str, base_url: str) -> list[str]:
    """Return absolute URLs for any ``.pdf`` links on the page."""

    soup = BeautifulSoup(html, "html.parser")
    links: list[str] = []
    for a in soup.find_all("a", href=True):
        href = a["href"].strip()
        if href.lower().endswith(".pdf"):
            links.append(urljoin(base_url, href))
    return sorted(set(links))


def fetch(url: str, session: requests.Session | None = None) -> str:
    """Fetch one allowlisted URL. Raises if the URL is not allowed."""

    if not is_allowed(url):
        raise ValueError(f"URL not on allowlist: {url}")
    session = session or requests.Session()
    logger.info("fetching %s", url)
    resp = session.get(url, headers={"User-Agent": USER_AGENT}, timeout=REQUEST_TIMEOUT)
    resp.raise_for_status()
    return resp.text


def crawl(urls: Iterable[str] | None = None, out_dir: Path | str = "data/raw") -> list[Path]:
    """Crawl the allowlist (or a subset of it) and save HTML + sidecar JSON.

    Returns the list of saved HTML paths. Pages are fetched serially with a
    small polite delay; the whole allowlist is short by design.
    """

    urls = list(urls) if urls is not None else list(ALLOWED_URLS)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    saved: list[Path] = []
    session = requests.Session()
    for url in urls:
        if not is_allowed(url):
            logger.warning("skipping non-allowlisted url: %s", url)
            continue
        try:
            html = fetch(url, session=session)
        except requests.RequestException as exc:
            logger.error("fetch failed for %s: %s", url, exc)
            continue

        doc_id = stable_id(url)
        html_path = out_dir / f"{doc_id}.html"
        meta_path = out_dir / f"{doc_id}.json"
        html_path.write_text(html, encoding="utf-8")
        write_json(
            meta_path,
            {"id": doc_id, "url": url, "pdf_links": _extract_pdf_links(html, url)},
        )
        saved.append(html_path)
        time.sleep(DELAY_SECONDS)

    logger.info("crawl complete: %d pages saved to %s", len(saved), out_dir)
    return saved


# ---------------------------------------------------------------------------
# Sitemap discovery (offline, review-only)
# ---------------------------------------------------------------------------
# Everything below is an offline utility for expanding ``ALLOWED_URLS``. It is
# deliberately decoupled from :func:`crawl` above: running discovery never
# touches ``data/raw`` and never modifies the allowlist. It produces a single
# JSON artifact under ``data/discovery/`` that a human reviews to decide which
# URLs to copy into ``config.ALLOWED_URLS`` for the next production crawl.


def _fetch_raw_bytes(url: str, session: requests.Session) -> bytes:
    """Fetch raw bytes and transparently decompress gzipped sitemaps.

    Sitemaps are served either as plain XML or as gzip-compressed XML with a
    ``.xml.gz`` suffix. We detect gzip by URL suffix *or* by the gzip magic
    header ``0x1f 0x8b`` so both conventions work without extra config.
    """

    resp = session.get(
        url, headers={"User-Agent": USER_AGENT}, timeout=REQUEST_TIMEOUT
    )
    resp.raise_for_status()
    content = resp.content
    if url.endswith(".gz") or content[:2] == b"\x1f\x8b":
        content = gzip.decompress(content)
    return content


def _sitemaps_from_robots(session: requests.Session) -> list[str]:
    """Return any ``Sitemap:`` URLs declared in robots.txt, or [] on failure.

    Per the Sitemap protocol, sites are expected to advertise their sitemap(s)
    via one or more ``Sitemap: <url>`` lines in robots.txt. Honoring these
    first means we pick up the *canonical* sitemap location instead of
    guessing, and avoids missing multi-sitemap setups.
    """

    try:
        resp = session.get(
            ROBOTS_TXT_URL,
            headers={"User-Agent": USER_AGENT},
            timeout=REQUEST_TIMEOUT,
        )
        resp.raise_for_status()
    except requests.RequestException as exc:
        logger.warning("could not fetch robots.txt: %s", exc)
        return []

    sitemaps: list[str] = []
    for line in resp.text.splitlines():
        line = line.strip()
        # Case-insensitive match; value is everything after the first colon.
        if line.lower().startswith("sitemap:"):
            sitemaps.append(line.split(":", 1)[1].strip())
    return sitemaps


def _parse_sitemap(xml_bytes: bytes) -> tuple[list[str], list[str]]:
    """Parse a sitemap, returning ``(page_urls, nested_sitemap_urls)``.

    The sitemaps.org schema defines two root elements that we care about:

    * ``<urlset>`` — a flat list of page URLs (leaf sitemap).
    * ``<sitemapindex>`` — a list of nested sitemap URLs (index sitemap).

    Both use ``<loc>`` for the URL payload, so we switch on the root tag and
    bucket the ``<loc>`` text into one of the two returned lists.
    """

    try:
        root = ET.fromstring(xml_bytes)
    except ET.ParseError as exc:
        logger.error("sitemap parse failed: %s", exc)
        return [], []

    # Strip XML namespace from tag names: "{http://...}urlset" -> "urlset".
    root_tag = root.tag.rsplit("}", 1)[-1]
    locs: list[str] = []
    for elem in root.iter():
        if elem.tag.rsplit("}", 1)[-1] == "loc" and elem.text:
            locs.append(elem.text.strip())

    if root_tag == "sitemapindex":
        return [], locs
    # Default to treating locs as page URLs; covers <urlset> and any
    # non-standard root that still uses <loc> for pages.
    return locs, []


def _classify(url: str, allow_set: set[str]) -> str:
    """Return which review bucket a URL belongs to.

    Returns one of: ``"in_allowlist"``, ``"denylisted"``, ``"pdf_candidate"``,
    ``"html_candidate"``, or ``"off_domain"``. Centralizing this keeps the
    classification rules in one place and makes it trivial to unit test.
    """

    parsed = urlparse(url)
    if parsed.netloc != ALLOWED_DOMAIN:
        # Some sitemap indices reference sister subdomains; we skip those
        # silently because the project scope is a single domain.
        return "off_domain"
    if url in allow_set:
        return "in_allowlist"
    lower = url.lower()
    if any(pat in lower for pat in DENY_PATTERNS):
        return "denylisted"
    # PDFs are surfaced separately because the current parser.py handles HTML
    # only; accepting a PDF into the allowlist requires an explicit decision
    # to either add PDF parsing or leave that URL out of scope.
    if parsed.path.lower().endswith(".pdf"):
        return "pdf_candidate"
    return "html_candidate"


def discover_urls(
    out_dir: Path | str = "data/discovery",
    seed_sitemaps: Iterable[str] | None = None,
) -> dict:
    """Enumerate all sitemap URLs and classify them for allowlist review.

    This routine does **not** modify ``ALLOWED_URLS``. It walks every sitemap
    on the site, classifies each URL against the current allowlist and
    denylist, and writes the result to ``<out_dir>/discovery.json``. A human
    then reviews ``html_candidates`` and ``pdf_candidates`` and decides which
    entries to copy into ``config.ALLOWED_URLS`` for the next crawl.

    Sitemap source priority (first hit wins, rest are ignored only if they
    overlap): sitemaps declared in robots.txt, then any ``seed_sitemaps``
    provided by the caller, then the static fallback ``SITEMAP_CANDIDATES``.
    Nested sitemap indices are followed breadth-first up to
    ``SITEMAP_DEPTH_LIMIT`` levels to bound worst-case work.
    """

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    session = requests.Session()

    # Priority-ordered initial queue: robots.txt > explicit seeds > fallbacks.
    initial: list[str] = _sitemaps_from_robots(session)
    if seed_sitemaps:
        initial.extend(seed_sitemaps)
    initial.extend(SITEMAP_CANDIDATES)

    seen_sitemaps: set[str] = set()
    all_page_urls: set[str] = set()
    queue: list[str] = initial

    # Breadth-first traversal of sitemap indices, bounded by depth so a
    # pathological recursive sitemap cannot hang the discovery run.
    for depth in range(SITEMAP_DEPTH_LIMIT):
        if not queue:
            break
        next_queue: list[str] = []
        for sm_url in queue:
            if sm_url in seen_sitemaps:
                continue
            seen_sitemaps.add(sm_url)
            logger.info("fetching sitemap (depth=%d): %s", depth, sm_url)
            try:
                xml_bytes = _fetch_raw_bytes(sm_url, session)
            except requests.RequestException as exc:
                logger.warning("sitemap fetch failed for %s: %s", sm_url, exc)
                continue
            pages, nested = _parse_sitemap(xml_bytes)
            all_page_urls.update(pages)
            next_queue.extend(nested)
            time.sleep(DELAY_SECONDS)
        queue = next_queue

    # Classify every discovered URL into one review bucket.
    allow_set = set(ALLOWED_URLS)
    in_allowlist: list[str] = []
    denylisted: list[dict[str, str]] = []
    html_candidates: list[str] = []
    pdf_candidates: list[str] = []

    for url in sorted(all_page_urls):
        bucket = _classify(url, allow_set)
        if bucket == "in_allowlist":
            in_allowlist.append(url)
        elif bucket == "denylisted":
            # Store the first matching denylist fragment so review is auditable.
            lower = url.lower()
            matched = next(pat for pat in DENY_PATTERNS if pat in lower)
            denylisted.append({"url": url, "matched": matched})
        elif bucket == "pdf_candidate":
            pdf_candidates.append(url)
        elif bucket == "html_candidate":
            html_candidates.append(url)
        # "off_domain" is intentionally dropped without recording.

    result = {
        "stats": {
            "total_discovered": len(all_page_urls),
            "in_allowlist": len(in_allowlist),
            "denylisted": len(denylisted),
            "html_candidates": len(html_candidates),
            "pdf_candidates": len(pdf_candidates),
        },
        "sitemaps_fetched": sorted(seen_sitemaps),
        "in_allowlist": in_allowlist,
        "denylisted": denylisted,
        "html_candidates": html_candidates,
        "pdf_candidates": pdf_candidates,
    }
    write_json(out_dir / "discovery.json", result)
    logger.info(
        "discovery complete: %d URLs total, %d already approved, "
        "%d denylisted, %d html candidates, %d pdf candidates",
        len(all_page_urls),
        len(in_allowlist),
        len(denylisted),
        len(html_candidates),
        len(pdf_candidates),
    )
    return result
