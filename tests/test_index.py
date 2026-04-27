"""Unit tests for ``rag_chatbot.utils.inverted_index``."""

from __future__ import annotations

import math

import pandas as pd
import pytest

from rag_chatbot.utils import inverted_index


class TestTokenizeField:
    """Field tokenization (string-only, empty/non-string short-circuits)."""

    def test_string_input(self):
        # "Students studying" → cleaned → ["student", "study"]
        assert inverted_index.tokenize_field("Students studying") == ["student", "study"]

    def test_non_string_returns_empty(self):
        assert inverted_index.tokenize_field(None) == []
        assert inverted_index.tokenize_field(42) == []

    def test_empty_string_returns_empty(self):
        assert inverted_index.tokenize_field("") == []


class TestComputeIdf:
    """BM25 inverse-document-frequency formula."""

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
    """Posting-list / doc-length / avgdl construction for one BM25 sub-index."""

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
    """End-to-end per-field index build from a DataFrame."""

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
