# crawl.py
from urllib.parse import urlparse, urlunparse, urldefrag, urljoin
from urllib.robotparser import RobotFileParser
from pathlib import Path
from collections import deque
from datetime import datetime, timezone

import hashlib
import posixpath
import re
import requests
import time
from bs4 import BeautifulSoup

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
    ".pdf", ".jpg", ".jpeg", ".png", ".gif", ".svg",
    ".css", ".js", ".zip", ".mp4", ".mp3",
    ".doc", ".docx", ".xls", ".xlsx", ".ppt", ".pptx",
    ".ico", ".woff", ".woff2", ".ttf",
}

# standardizeed URL upon crawling
def normalize_url(url: str) -> str:
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

# how to name raw data files
def slugify(url: str) -> str:
    url = normalize_url(url)
    p = urlparse(url)
    host = p.netloc.replace("www.", "")
    path = (p.path.strip("/").replace("/", "_") or "index")[:100]
    h = hashlib.sha1(url.encode()).hexdigest()[:8]
    return f"{host}__{path}__{h}"

# determine if a URL is crawlable 
def is_crawlable(url: str, seed_host: str, path_prefix: str | None) -> bool:
    p = urlparse(url)
    if p.scheme not in ("http", "https"):
        return False
    if p.netloc.replace("www.", "") != seed_host.replace("www.", ""): # make sure same host as seed
        return False
    if path_prefix:  # crawl only under the specified path prefix
        prefix = path_prefix.rstrip("/")
        if not (p.path == prefix or p.path.startswith(prefix + "/")):
            return False
    if any(p.path.lower().endswith(ext) for ext in SKIP_EXT): 
        return False
    return True


def fetch(url: str):
    try:
        r = requests.get(url, headers=HEADERS, timeout=TIMEOUT)
        r.raise_for_status()
        if "text/html" not in r.headers.get("Content-Type", ""):
            return None
        return r
    except requests.RequestException as e:
        print(f"   ! fetch failed: {e}")
        return None


def extract_links(html: str, base_url: str) -> list[str]:
    soup = BeautifulSoup(html, "html.parser")
    out = set()
    for a in soup.find_all("a", href=True):
        href = a["href"].strip()
        if not href or href.startswith(("javascript:", "mailto:", "tel:", "#")):
            continue
        abs_url = urljoin(base_url, href)
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


# save raw HTML
def save_page(url: str, html: str, meta_extra: dict) -> None:
    url = normalize_url(url)
    slug = slugify(url)
    (RAW_DIR / f"{slug}.html").write_text(html, encoding="utf-8")
  
# see if the URL is allowed to be crawled by robots.txt
def get_robots(seed_url: str) -> RobotFileParser:
    p = urlparse(seed_url)
    rp = RobotFileParser()
    rp.set_url(f"{p.scheme}://{p.netloc}/robots.txt")
    try:
        rp.read()
    except Exception:
        pass
    return rp

# crawling logic
def crawl_seed(seed: dict) -> None:
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

    print(f"\n=== {seed_url}")
    print(f"    depth<={MAX_DEPTH}, max_pages={MAX_PAGES_PER_SEED}, prefix={path_prefix}")

    while queue and saved < MAX_PAGES_PER_SEED:
        url, depth, parent = queue.popleft()
        if url in visited:
            continue
        visited.add(url)

        if not is_crawlable(url, seed_host, path_prefix):
            continue
        if not rp.can_fetch(ua, url):
            print(f"   robots-blocked: {url}")
            continue

        print(f"[d={depth}] {url}")
        resp = fetch(url)
        if resp is None:
            continue

        # --- dedup checks (parse once, reuse soup) ---
        soup = BeautifulSoup(resp.text, "html.parser")

        canonical = _canonical_url(soup)
        if canonical and canonical in seen_canonicals:
            print(f"   skip duplicate (canonical): {url} == {seen_canonicals[canonical]}")
            time.sleep(DELAY_SEC)
            continue

        body_hash = _article_hash(soup)
        if body_hash and body_hash in seen_hashes:
            print(f"   skip duplicate (content hash): {url} == {seen_hashes[body_hash]}")
            time.sleep(DELAY_SEC)
            continue
        # --- end dedup ---

        save_page(url, resp.text, {
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
        })
        saved += 1
        if canonical:
            seen_canonicals[canonical] = url
        if body_hash:
            seen_hashes[body_hash] = url

        if depth < MAX_DEPTH:
            for link in extract_links(resp.text, url):
                if link not in visited:
                    queue.append((link, depth + 1, url))

        time.sleep(DELAY_SEC)

    print(f"saved={saved}, queued_unvisited={len(queue)}")


def main():
    for seed in SEEDS:
        crawl_seed(seed)


if __name__ == "__main__":
    main()