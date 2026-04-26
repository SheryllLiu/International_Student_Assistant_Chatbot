"""Unit tests for the data-processing helpers in ``rag_chatbot/utils/``.

Covers the pure functions that transform text / HTML / URLs / DataFrames into
the shapes the retriever expects. Network calls and on-disk crawling are not
exercised — only the deterministic data layer.

Run from repo root::

    pytest tests/test_utils.py -q
"""
from __future__ import annotations

import math

import pandas as pd
import pytest
from bs4 import BeautifulSoup

from rag_chatbot.utils import data_crawling, html_parser, inverted_index, text_cleaning



# Test fortext_cleaning.py

class TestNormalizeText:
    def test_lowercases(self):
        assert text_cleaning.normalize_text("HeLLo") == "hello"

    def test_strips_punctuation_keeps_hyphen(self):
        # commas/periods become spaces; hyphen survives so "f-1" stays intact.
        assert text_cleaning.normalize_text("F-1, visa.") == "f-1 visa"

    def test_collapses_whitespace(self):
        assert text_cleaning.normalize_text("a   b\t\nc") == "a b c"

    def test_keeps_digits(self):
        assert text_cleaning.normalize_text("Form I-20 (2024)") == "form i-20 2024"

    def test_empty_string_returns_empty(self):
        assert text_cleaning.normalize_text("") == ""


class TestRemoveStopwords:
    def test_drops_common_stopwords(self):
        assert text_cleaning.remove_stopwords(["the", "student", "is", "here"]) == ["student"]

    def test_drops_pure_hyphen_tokens(self):
        # "-" and "--" collapse to empty after strip("-") and must be removed.
        assert text_cleaning.remove_stopwords(["-", "f-1", "--"]) == ["f-1"]

    def test_keeps_non_stopwords(self):
        assert text_cleaning.remove_stopwords(["visa", "passport"]) == ["visa", "passport"]


class TestLemmatizeToken:
    def test_pass_through_hyphenated(self):
        assert text_cleaning.lemmatize_token("on-campus") == "on-campus"

    def test_pass_through_numeric(self):
        # Anything containing a digit is left alone (e.g. "f-1", "i20").
        assert text_cleaning.lemmatize_token("i20") == "i20"

    def test_verb_lemma(self):
        # "studies" → "study" via verb POS.
        assert text_cleaning.lemmatize_token("studies") == "study"

    def test_noun_fallback(self):
        # "students" should fall back to noun lemma "student".
        assert text_cleaning.lemmatize_token("students") == "student"

    def test_empty_token(self):
        assert text_cleaning.lemmatize_token("") == ""


class TestCleanText:
    def test_full_pipeline(self):
        # "The students are studying" → drop "the/are" stopwords, lemmatize.
        out = text_cleaning.clean_text("The students are studying.")
        assert out == "student study"

    def test_preserves_domain_hyphenated_terms(self):
        out = text_cleaning.clean_text("F-1 visa for on-campus jobs")
        # stopwords "for" gone; hyphenated tokens untouched.
        assert "f-1" in out.split()
        assert "on-campus" in out.split()

    def test_blank_input_returns_empty(self):
        assert text_cleaning.clean_text("   ") == ""


class TestNormalizeQueryTerms:
    def test_unhyphenated_visa_code(self):
        assert "f-1" in text_cleaning.normalize_query_terms("apply for f1 visa")

    def test_form_number_with_space(self):
        assert "i-20" in text_cleaning.normalize_query_terms("need an I 20 form")

    def test_abbreviation_expansion(self):
        out = text_cleaning.normalize_query_terms("how to apply for OPT?")
        assert "optional practical training" in out

    def test_longest_match_wins(self):
        # "my e verify" must be consumed before the shorter "e verify" key
        # would fire on the same span.
        out = text_cleaning.normalize_query_terms("my e verify status")
        assert "mye-verify" in out
        # And "e-verify" should NOT also appear from a second pass on the same span.
        assert out.count("verify") == 1

    def test_hyphen_word_boundary_protects_inner_match(self):
        # "ssa" should NOT fire inside "ssa-l676".
        out = text_cleaning.normalize_query_terms("file ssa-l676 today")
        assert "ssa-l676" in out
        assert "social security administration" not in out


class TestCleanQueryText:
    def test_normalizes_then_cleans(self):
        # "f1 opt" → "f-1 optional practical training" → cleaned tokens.
        # "training" lemmatizes to "train" (verb POS tried first).
        out = text_cleaning.clean_query_text("apply for f1 opt").split()
        assert "f-1" in out
        assert "optional" in out
        assert "practical" in out
        assert "train" in out
        assert "for" not in out  # stopword removed

    def test_non_string_returns_empty(self):
        assert text_cleaning.clean_query_text(None) == ""  # type: ignore[arg-type]
        assert text_cleaning.clean_query_text(123) == ""  # type: ignore[arg-type]


class TestCleanDocumentRow:
    def test_concatenates_text_columns(self):
        row = pd.Series({"topic": "work", "title": "OPT", "text": "Students apply."})
        out = text_cleaning.clean_document_row(row, ["title", "text"])
        # "OPT" is uppercased, becomes lowercased token; "Students" → "student".
        tokens = out.split()
        assert "opt" in tokens
        assert "student" in tokens

    def test_skips_missing_and_non_string(self):
        row = pd.Series({"title": "Hello", "section": None, "text": 42})
        out = text_cleaning.clean_document_row(row, ["title", "section", "text"])
        # Only "Hello" is a non-empty string; the cleaned form is "hello".
        assert out == "hello"


class TestBuildCleanedCorpus:
    def test_output_columns_and_content(self):
        df = pd.DataFrame(
            [
                {"topic": "work", "title": "OPT Basics", "section": "Intro", "text": "Students may apply."},
                {"topic": "travel", "title": "Re-entry", "section": "", "text": "Carry your I-20."},
            ]
        )
        out = text_cleaning.build_cleaned_corpus(df)

        # Schema
        for col in ("topic", "title", "section", "text", "raw_document", "cleaned_document"):
            assert col in out.columns

        # Raw text columns are preserved verbatim.
        assert out.loc[0, "title"] == "OPT Basics"
        assert out.loc[1, "text"] == "Carry your I-20."

        # raw_document joins non-empty raw fields with newlines (skipping empty section).
        assert "Re-entry" in out.loc[1, "raw_document"]
        assert "Carry your I-20." in out.loc[1, "raw_document"]
        assert out.loc[1, "raw_document"].count("\n") == 1  # only 2 non-empty parts

        # cleaned_document is the merged & normalized text.
        cleaned_tokens = out.loc[0, "cleaned_document"].split()
        assert "student" in cleaned_tokens
        assert "opt" in cleaned_tokens

        # Hyphenated/numeric tokens survive cleaning.
        assert "i-20" in out.loc[1, "cleaned_document"].split()

    def test_missing_topic_column_raises(self):
        df = pd.DataFrame([{"title": "x", "text": "y"}])
        with pytest.raises(ValueError, match="topic"):
            text_cleaning.build_cleaned_corpus(df)



#  Test for html_parser.py 

class TestHtmlParserCleanText:
    def test_replaces_nbsp_and_collapses(self):
        assert html_parser.clean_text("hello\xa0\xa0world  foo") == "hello world foo"

    def test_empty_returns_empty(self):
        assert html_parser.clean_text("") == ""


class TestExtractTopic:
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



# Test forinverted_index.py

class TestTokenizeField:
    def test_string_input(self):
        # "Students studying" → cleaned → ["student", "study"]
        assert inverted_index.tokenize_field("Students studying") == ["student", "study"]

    def test_non_string_returns_empty(self):
        assert inverted_index.tokenize_field(None) == []
        assert inverted_index.tokenize_field(42) == []

    def test_empty_string_returns_empty(self):
        assert inverted_index.tokenize_field("") == []


class TestComputeIdf:
    def test_idf_is_non_negative(self):
        idf = inverted_index.compute_idf({"a": 1, "b": 5}, N=10)
        assert all(v >= 0 for v in idf.values())

    def test_rarer_term_has_higher_idf(self):
        idf = inverted_index.compute_idf({"common": 9, "rare": 1}, N=10)
        assert idf["rare"] > idf["common"]

    def test_idf_value(self):
        # Spot-check the formula: log(1 + (N - df + 0.5) / (df + 0.5))
        idf = inverted_index.compute_idf({"x": 2}, N=5)
        expected = math.log(1 + (5 - 2 + 0.5) / (2 + 0.5))
        assert idf["x"] == pytest.approx(expected)


class TestBuildFieldIndex:
    def test_basic_shape(self):
        tokens_by_doc = {
            0: ["a", "b", "a"],
            1: ["b", "c"],
        }
        sub = inverted_index.build_field_index(tokens_by_doc)

        assert sub["N"] == 2
        assert sub["doc_len"] == {0: 3, 1: 2}
        assert sub["avgdl"] == pytest.approx(2.5)
        # "a" only appears in doc 0; "b" in both; "c" in doc 1.
        assert sub["doc_freq"]["a"] == 1
        assert sub["doc_freq"]["b"] == 2
        assert sub["doc_freq"]["c"] == 1
        # Postings carry (doc_id, term_freq).
        assert sub["inverted_index"]["a"] == [(0, 2)]
        assert sorted(sub["inverted_index"]["b"]) == [(0, 1), (1, 1)]

    def test_empty_corpus(self):
        sub = inverted_index.build_field_index({})
        assert sub["N"] == 0
        assert sub["avgdl"] == 0.0
        assert sub["doc_len"] == {}
        assert sub["inverted_index"] == {}


class TestBuildIndexes:
    def test_full_pipeline_small_df(self):
        df = pd.DataFrame(
            [
                {"topic": "work", "title": "OPT Basics", "text": "students apply"},
                {"topic": "travel", "title": "Re-entry", "text": "carry i-20"},
            ]
        )
        idx = inverted_index.build_indexes(df, k1=1.5, b=0.3)

        assert idx["N"] == 2
        assert idx["k1"] == 1.5
        assert idx["b"] == 0.3
        assert set(idx["fields"].keys()) == {"topic", "title", "text"}

        # doc_store carries every input column.
        assert idx["doc_store"][0]["topic"] == "work"
        assert idx["doc_store"][1]["title"] == "Re-entry"

        # The "text" field index should have a non-empty vocabulary.
        text_field = idx["fields"]["text"]
        assert len(text_field["inverted_index"]) >= 1
        assert text_field["N"] == 2



# Test fordata_crawling.py (no network)

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
