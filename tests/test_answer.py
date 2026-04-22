import json
from pathlib import Path

from summerizer.utils.answer import (
    AnswerResult,
    build_answer_text,
    build_citations,
    choose_supporting_sentences,
    score_sentence_against_query,
    write_answer_json,
)
from summerizer.utils.retriever import RetrievalResult


def make_result(title: str, url: str, text: str, score: float = 1.0) -> RetrievalResult:
    return RetrievalResult(
        chunk_id="chunk_1",
        url=url,
        title=title,
        chunk_index=0,
        text=text,
        score=score,
    )


def test_choose_supporting_sentences_takes_useful_sentences():
    results = [
        make_result(
            "Work Rules",
            "https://example.com/work",
            "F-1 students may work on campus. Students may not work more than 20 hours per week while school is in session.",
        )
    ]

    sentences = choose_supporting_sentences(
        "How many hours can I work on campus during the semester?",
        results,
        max_sentences=2,
    )

    assert len(sentences) == 2
    assert "20 hours per week" in sentences[1]


def test_score_sentence_against_query_prefers_more_relevant_sentence():
    query = "How many hours can I work on campus during the semester?"
    generic_sentence = "In order to apply, talk to your DSO and follow the instructions."
    relevant_sentence = (
        "If you participate in on-campus employment, you may not work more than "
        "20 hours per week when school is in session."
    )

    generic_score = score_sentence_against_query(query, generic_sentence)
    relevant_score = score_sentence_against_query(query, relevant_sentence)

    assert relevant_score > generic_score


def test_build_citations_deduplicates_urls():
    results = [
        make_result("A", "https://example.com/page", "One."),
        make_result("B", "https://example.com/page", "Two."),
        make_result("C", "https://example.com/other", "Three."),
    ]

    citations = build_citations(results)

    assert citations == [
        "https://example.com/page",
        "https://example.com/other",
    ]


def test_build_answer_text_handles_empty_results():
    answer = build_answer_text("What are the work rules?", [])

    assert "could not find a strong match" in answer


def test_build_answer_text_uses_retrieved_sentences():
    results = [
        make_result(
            "Work Rules",
            "https://example.com/work",
            "In order to apply, talk to your DSO. Students may not work more than 20 hours per week while school is in session.",
        )
    ]

    answer = build_answer_text("How many hours can I work?", results)

    assert "Based on the retrieved documents:" in answer
    assert "20 hours per week" in answer


def test_write_answer_json_saves_output(tmp_path: Path):
    result = AnswerResult(
        query="Sample question",
        answer="Sample answer",
        citations=["https://example.com/page"],
        supporting_chunks=[],
    )
    out_path = tmp_path / "answers" / "answer.json"

    written_path = write_answer_json(result, out_path)

    assert written_path == out_path
    assert out_path.exists()

    contents = json.loads(out_path.read_text(encoding="utf-8"))
    assert contents["query"] == "Sample question"
    assert contents["answer"] == "Sample answer"
