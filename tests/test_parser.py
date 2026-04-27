"""Unit tests for ``rag_chatbot.utils.html_parser``."""

from __future__ import annotations

from bs4 import BeautifulSoup

from rag_chatbot.utils import html_parser


class TestHtmlParserCleanText:
    """Whitespace + nbsp normalization helper."""

    def test_replaces_nbsp_and_collapses(self):
        assert html_parser.clean_text("hello\xa0\xa0world  foo") == "hello world foo"

    def test_empty_returns_empty(self):
        assert html_parser.clean_text("") == ""


class TestExtractTopic:
    """Pulling the topic segment out of a page's canonical link."""

    def _soup(self, href: str) -> BeautifulSoup:
        return BeautifulSoup(f'<link rel="canonical" href="{href}">', "lxml")

    def test_segment_after_students(self):
        soup = self._soup("https://studyinthestates.dhs.gov/students/work/some-page")
        assert html_parser.extract_topic(soup) == "work"

    def test_no_canonical_returns_empty(self):
        soup = BeautifulSoup("<html></html>", "lxml")
        assert html_parser.extract_topic(soup) == ""

    def test_students_at_end_returns_empty(self):
        # "students" is the last segment, no following segment to use as topic.
        soup = self._soup("https://studyinthestates.dhs.gov/students")
        assert html_parser.extract_topic(soup) == ""

    def test_url_without_students_segment(self):
        soup = self._soup("https://example.com/about/team")
        assert html_parser.extract_topic(soup) == ""


class TestParseArticle:
    """Heading/section state machine that turns one ``<article>`` into rows."""

    def test_h2_h3_p_state_machine(self):
        html = """
        <article>
          <h2>Working in the US</h2>
          <p>Intro paragraph.</p>
          <h3>OPT</h3>
          <p>OPT details.</p>
          <p>More OPT details.</p>
          <h3>CPT</h3>
          <p>CPT details.</p>
        </article>
        """
        article = BeautifulSoup(html, "lxml").find("article")
        rows = html_parser.parse_article(article)

        # We should get one row per (title, section) combo, plus the intro
        # which has empty section.
        sections = [(r["title"], r["section"]) for r in rows]
        assert ("Working in the US", "") in sections
        assert ("Working in the US", "OPT") in sections
        assert ("Working in the US", "CPT") in sections

        # The OPT block should accumulate both paragraphs into one text field.
        opt_row = next(r for r in rows if r["section"] == "OPT")
        assert "OPT details." in opt_row["text"]
        assert "More OPT details." in opt_row["text"]

    def test_skips_related_tags_paragraphs(self):
        html = """
        <article>
          <h2>Title</h2>
          <p>Real content.</p>
          <p>Related Tags: foo, bar</p>
        </article>
        """
        article = BeautifulSoup(html, "lxml").find("article")
        rows = html_parser.parse_article(article)
        assert len(rows) == 1
        assert "Related Tags" not in rows[0]["text"]
        assert "Real content." in rows[0]["text"]

    def test_ignores_content_before_first_h2(self):
        html = """
        <article>
          <p>Orphan paragraph before any h2 — should be dropped.</p>
          <h2>Title</h2>
          <p>Kept.</p>
        </article>
        """
        article = BeautifulSoup(html, "lxml").find("article")
        rows = html_parser.parse_article(article)
        assert len(rows) == 1
        assert rows[0]["text"] == "Kept."
