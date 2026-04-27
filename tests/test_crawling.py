"""Unit tests for ``rag_chatbot.utils.data_crawling`` (no network)."""

from __future__ import annotations

from rag_chatbot.utils import data_crawling


class TestNormalizeUrl:
    """URL canonicalization for stable dedup keys."""

    def test_strips_fragment(self):
        assert data_crawling.normalize_url("https://x.com/a#frag") == "https://x.com/a"

    def test_strips_www(self):
        assert data_crawling.normalize_url("https://www.x.com/a") == "https://x.com/a"

    def test_lowercases_host_keeps_path_case(self):
        # Host is lowercased but the path's case is preserved.
        assert data_crawling.normalize_url("HTTPS://X.COM/Foo") == "https://x.com/Foo"

    def test_force_upgrades_http_to_https(self):
        # The function rewrites http→https unconditionally; documented here
        # so future changes to that behavior surface as a test failure.
        assert data_crawling.normalize_url("http://x.com/a") == "https://x.com/a"

    def test_strips_default_https_port(self):
        assert data_crawling.normalize_url("https://x.com:443/a") == "https://x.com/a"

    def test_preserves_nondefault_port(self):
        assert data_crawling.normalize_url("https://x.com:8080/a") == "https://x.com:8080/a"

    def test_trailing_slash_collapsed_to_root(self):
        # "/" stays as "/", but "/a/" becomes "/a".
        assert data_crawling.normalize_url("https://x.com/") == "https://x.com/"
        assert data_crawling.normalize_url("https://x.com/a/") == "https://x.com/a"


class TestSlugify:
    """Deterministic ``host__path__hash`` filename slug."""

    def test_deterministic(self):
        url = "https://studyinthestates.dhs.gov/students/work/foo"
        assert data_crawling.slugify(url) == data_crawling.slugify(url)

    def test_includes_host_and_path(self):
        slug = data_crawling.slugify("https://www.example.com/a/b/c")
        assert slug.startswith("example.com__a_b_c__")

    def test_index_when_path_empty(self):
        slug = data_crawling.slugify("https://example.com/")
        assert "__index__" in slug

    def test_includes_8_char_hash_suffix(self):
        slug = data_crawling.slugify("https://example.com/a")
        suffix = slug.rsplit("__", 1)[-1]
        assert len(suffix) == 8


class TestIsCrawlable:
    """Host / path-prefix / file-extension gating for crawlable URLs."""

    def test_blocks_offhost(self):
        assert not data_crawling.is_crawlable(
            "https://other.com/students/page",
            seed_host="studyinthestates.dhs.gov",
            path_prefix="/students/",
        )

    def test_blocks_outside_prefix(self):
        assert not data_crawling.is_crawlable(
            "https://studyinthestates.dhs.gov/schools/page",
            seed_host="studyinthestates.dhs.gov",
            path_prefix="/students/",
        )

    def test_allows_under_prefix(self):
        assert data_crawling.is_crawlable(
            "https://studyinthestates.dhs.gov/students/work/opt",
            seed_host="studyinthestates.dhs.gov",
            path_prefix="/students/",
        )

    def test_blocks_skip_extensions(self):
        assert not data_crawling.is_crawlable(
            "https://studyinthestates.dhs.gov/students/file.pdf",
            seed_host="studyinthestates.dhs.gov",
            path_prefix="/students/",
        )

    def test_blocks_non_http_scheme(self):
        assert not data_crawling.is_crawlable(
            "ftp://studyinthestates.dhs.gov/students/x",
            seed_host="studyinthestates.dhs.gov",
            path_prefix="/students/",
        )


class TestExtractLinks:
    """Link extraction from rendered HTML, with junk-scheme filtering."""

    def test_resolves_relative_and_absolute(self):
        html = """
        <a href="/students/a">A</a>
        <a href="https://example.com/students/b">B</a>
        <a href="#frag">skip</a>
        <a href="mailto:x@y.com">skip</a>
        <a href="javascript:void(0)">skip</a>
        """
        links = data_crawling.extract_links(html, "https://example.com/page")
        # /students/a resolves against the base; both should be normalized.
        assert "https://example.com/students/a" in links
        assert "https://example.com/students/b" in links
        # Fragment / mailto / js links must be filtered.
        assert all("mailto:" not in u for u in links)
        assert all("javascript:" not in u for u in links)

    def test_dedupes(self):
        html = '<a href="/a">1</a><a href="/a">2</a>'
        links = data_crawling.extract_links(html, "https://example.com/")
        assert links.count("https://example.com/a") == 1
