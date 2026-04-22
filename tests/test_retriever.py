import json
from pathlib import Path

from summerizer.utils.retriever import (
    build_and_save_index,
    build_index,
    load_chunks,
    load_index,
    search_index,
    search_saved_index,
    tokenize,
    write_search_results_json,
)


def test_tokenize_removes_common_stopwords():
    tokens = tokenize("The student is working in the United States")

    assert "the" not in tokens
    assert "is" not in tokens
    assert "student" in tokens
    assert "working" in tokens


def test_build_index_requires_chunks():
    try:
        build_index([])
        assert False, "expected ValueError for empty chunks"
    except ValueError:
        assert True


def test_load_chunks_reads_json(tmp_path: Path):
    chunks_path = tmp_path / "chunks.json"
    chunks_path.write_text(
        json.dumps([{"chunk_id": "a", "text": "hello world"}]),
        encoding="utf-8",
    )

    chunks = load_chunks(chunks_path)

    assert len(chunks) == 1
    assert chunks[0]["chunk_id"] == "a"


def test_search_index_returns_relevant_chunk():
    chunks = [
        {
            "chunk_id": "chunk_1",
            "url": "https://example.com/work",
            "title": "Working in the United States",
            "chunk_index": 0,
            "text": "F-1 students may work on campus up to 20 hours during the semester.",
        },
        {
            "chunk_id": "chunk_2",
            "url": "https://example.com/travel",
            "title": "Travel Rules",
            "chunk_index": 0,
            "text": "Travel signatures are required before reentering the United States.",
        },
    ]

    saved_index = build_index(chunks)
    results = search_index("on campus work hours", saved_index, top_k=2)

    assert len(results) >= 1
    assert results[0].chunk_id == "chunk_1"
    assert "20 hours" in results[0].text


def test_build_and_save_index_then_load_and_search(tmp_path: Path):
    chunks_path = tmp_path / "chunks.json"
    index_path = tmp_path / "index" / "bm25_index.pkl"

    chunks_path.write_text(
        json.dumps(
            [
                {
                    "chunk_id": "chunk_1",
                    "url": "https://example.com/ssn",
                    "title": "Social Security Number",
                    "chunk_index": 0,
                    "text": "Students may need a Social Security Number to be paid for on-campus work.",
                }
            ]
        ),
        encoding="utf-8",
    )

    written_path = build_and_save_index(chunks_path, index_path)
    saved_index = load_index(index_path)
    results = search_saved_index("social security number", index_path, top_k=3)

    assert written_path == index_path
    assert index_path.exists()
    assert len(saved_index.chunks) == 1
    assert len(results) == 1
    assert results[0].title == "Social Security Number"


def test_write_search_results_json_saves_results(tmp_path: Path):
    chunks_path = tmp_path / "chunks.json"
    index_path = tmp_path / "index" / "bm25_index.pkl"
    out_path = tmp_path / "results" / "search_results.json"

    chunks_path.write_text(
        json.dumps(
            [
                {
                    "chunk_id": "chunk_1",
                    "url": "https://example.com/visa",
                    "title": "Visa Info",
                    "chunk_index": 0,
                    "text": "Apply for a visa at a U.S. embassy or consulate.",
                }
            ]
        ),
        encoding="utf-8",
    )

    build_and_save_index(chunks_path, index_path)
    written_path = write_search_results_json(
        query="visa application",
        out_path=out_path,
        index_path=index_path,
        top_k=5,
    )

    assert written_path == out_path
    assert out_path.exists()

    contents = json.loads(out_path.read_text(encoding="utf-8"))
    assert len(contents) == 1
    assert contents[0]["title"] == "Visa Info"
