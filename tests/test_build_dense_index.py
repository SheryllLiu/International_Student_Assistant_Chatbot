"""Unit tests for the pure helpers in ``rag_chatbot/information_retrieval/build_dense_index.py``.

Covers only the deterministic data-transformation layer.  SentenceTransformer
encoding and FAISS I/O are not exercised here.

Run from repo root::

    pytest tests/test_build_dense_index.py -q
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pandas as pd

from rag_chatbot.utils.build_dense_index import (
    build_corpus,
    check_token_lengths,
)


# Test for build_dense_index.py (no network)
class TestBuildCorpus:
    """Corpus assembly: column joining, empty-row dropping, row_id resetting."""

    def _df(self, rows: list[dict]) -> pd.DataFrame:
        base = {"topic": "work", "title": "", "section": "", "text": ""}
        return pd.DataFrame([{**base, **r} for r in rows])

    def test_output_columns(self):
        df = self._df([{"title": "OPT", "text": "Students apply."}])
        out = build_corpus(df)
        for col in ("row_id", "topic", "title", "section", "text", "natural_text"):
            assert col in out.columns

    def test_joins_title_section_text(self):
        df = self._df([{"title": "OPT", "section": "Work Auth", "text": "Students apply."}])
        out = build_corpus(df)
        assert out.loc[0, "natural_text"] == "OPT. Work Auth. Students apply."

    def test_skips_empty_section(self):
        # empty section must not produce a double period
        df = self._df([{"title": "OPT", "section": "", "text": "Students apply."}])
        out = build_corpus(df)
        assert out.loc[0, "natural_text"] == "OPT. Students apply."

    def test_skips_whitespace_only_parts(self):
        df = self._df([{"title": "  ", "section": "", "text": "Carry I-20."}])
        out = build_corpus(df)
        assert out.loc[0, "natural_text"] == "Carry I-20."

    def test_drops_fully_empty_rows(self):
        # a row where all TEXT_COLS are empty should be removed
        df = self._df(
            [
                {"title": "OPT", "text": "Good row."},
                {"title": "", "section": "", "text": ""},
            ]
        )
        out = build_corpus(df)
        assert len(out) == 1
        assert out.loc[0, "title"] == "OPT"

    def test_row_id_sequential_after_drop(self):
        # row_id reflects the *post-drop* reset index
        df = self._df(
            [
                {"title": "", "section": "", "text": ""},  # dropped
                {"title": "F-1", "text": "Entry visa."},
                {"title": "OPT", "text": "Work permit."},
            ]
        )
        out = build_corpus(df)
        assert list(out["row_id"]) == [0, 1]

    def test_handles_nan_input(self):
        df = pd.DataFrame(
            [{"topic": "travel", "title": None, "section": float("nan"), "text": "Re-entry."}]
        )
        out = build_corpus(df)
        assert "Re-entry." in out.loc[0, "natural_text"]

    def test_topic_column_preserved(self):
        df = self._df([{"topic": "health", "title": "Insurance", "text": "Required."}])
        out = build_corpus(df)
        assert out.loc[0, "topic"] == "health"

    def test_multiple_rows_all_kept(self):
        df = self._df(
            [
                {"title": "A", "text": "First."},
                {"title": "B", "text": "Second."},
                {"title": "C", "text": "Third."},
            ]
        )
        out = build_corpus(df)
        assert len(out) == 3

    def test_only_expected_columns_returned(self):
        # extra columns in the input should NOT leak into the output
        df = self._df([{"title": "OPT", "text": "apply."}])
        df["extra_col"] = "noise"
        out = build_corpus(df)
        assert "extra_col" not in out.columns


def _mock_model(lengths: list[int]) -> MagicMock:
    """Return a fake SentenceTransformer whose tokenizer returns controlled lengths."""
    model = MagicMock()
    model.tokenizer.encode.side_effect = [list(range(n)) for n in lengths]
    return model


class TestCheckTokenLengths:
    """Token-length audit + warning behavior for the BGE 512-token cap."""

    def test_all_within_limit_logs_info(self, caplog):
        import logging

        with caplog.at_level(logging.INFO, logger="isa.index.dense"):
            model = _mock_model([10, 50, 511])
            check_token_lengths(["a", "b", "c"], model)
        assert any("all documents fit" in r.message for r in caplog.records)
        assert not any(r.levelno >= logging.WARNING for r in caplog.records)

    def test_over_limit_logs_warning(self, caplog):
        import logging

        with caplog.at_level(logging.WARNING, logger="isa.index.dense"):
            model = _mock_model([10, 600])
            check_token_lengths(["short", "very long doc"], model)
        warnings = [r.message for r in caplog.records if r.levelno >= logging.WARNING]
        assert any("1 docs exceed" in w for w in warnings)

    def test_reports_correct_max_length(self, caplog):
        import logging

        with caplog.at_level(logging.INFO, logger="isa.index.dense"):
            model = _mock_model([42, 300, 100])
            check_token_lengths(["a", "b", "c"], model)
        assert any("max: 300" in r.message for r in caplog.records)

    def test_exactly_at_limit_is_not_warned(self, caplog):
        import logging

        with caplog.at_level(logging.WARNING, logger="isa.index.dense"):
            model = _mock_model([512])
            check_token_lengths(["borderline doc"], model)
        assert not any(r.levelno >= logging.WARNING for r in caplog.records)
