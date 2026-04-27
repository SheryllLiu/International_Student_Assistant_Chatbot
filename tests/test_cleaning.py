"""Unit tests for ``rag_chatbot.utils.text_cleaning``."""

from __future__ import annotations

import pandas as pd
import pytest

from rag_chatbot.utils import text_cleaning


class TestNormalizeText:
    """Lowercase + punctuation/whitespace normalization rules."""

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
    """Stopword and pure-hyphen token filtering."""

    def test_drops_common_stopwords(self):
        assert text_cleaning.remove_stopwords(["the", "student", "is", "here"]) == ["student"]

    def test_drops_pure_hyphen_tokens(self):
        # "-" and "--" collapse to empty after strip("-") and must be removed.
        assert text_cleaning.remove_stopwords(["-", "f-1", "--"]) == ["f-1"]

    def test_keeps_non_stopwords(self):
        assert text_cleaning.remove_stopwords(["visa", "passport"]) == ["visa", "passport"]


class TestLemmatizeToken:
    """WordNet lemmatization with hyphen/digit pass-through."""

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
    """End-to-end text-cleaning pipeline."""

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
    """Domain-specific query rewriting (visa codes, form numbers, abbreviations)."""

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
    """Query-side wrapper that normalizes terms then runs the cleaning pipeline."""

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
    """Per-row column merging + cleaning."""

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
    """Top-level corpus-builder that produces the BM25 input DataFrame."""

    def test_output_columns_and_content(self):
        df = pd.DataFrame(
            [
                {
                    "topic": "work",
                    "title": "OPT Basics",
                    "section": "Intro",
                    "text": "Students may apply.",
                },
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
