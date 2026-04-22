import json
from pathlib import Path

from summerizer.utils.chunker import (
    build_text_chunks,
    chunk_document,
    chunk_parsed_documents,
    count_words,
    split_into_sentences,
    write_chunks_json,
)


def test_split_into_sentences_breaks_text_into_sentences():
    text = "First sentence. Second sentence! Third sentence?"

    sentences = split_into_sentences(text)

    assert sentences == [
        "First sentence.",
        "Second sentence!",
        "Third sentence?",
    ]


def test_build_text_chunks_groups_sentences_under_word_limit():
    text = "One two three. Four five six. Seven eight nine. Ten eleven twelve."

    chunks = build_text_chunks(text, max_words=6)

    assert chunks == [
        "One two three. Four five six.",
        "Seven eight nine. Ten eleven twelve.",
    ]


def test_build_text_chunks_splits_very_long_sentence():
    text = " ".join(f"word{i}" for i in range(1, 16))

    chunks = build_text_chunks(text, max_words=5)

    assert len(chunks) == 3
    assert all(count_words(chunk) <= 5 for chunk in chunks)


def test_chunk_document_returns_chunks_with_metadata():
    document = {
        "source_path": "data/raw/example.html",
        "url": "https://example.com/page",
        "title": "Example Page",
        "headings": ["Intro", "Rules"],
        "text": "First sentence here. Second sentence here. Third sentence here.",
    }

    chunks = chunk_document(document, max_words=4)

    assert len(chunks) >= 2
    assert chunks[0].title == "Example Page"
    assert chunks[0].url == "https://example.com/page"
    assert chunks[0].headings == ["Intro", "Rules"]
    assert chunks[0].chunk_id.startswith("example_page_chunk_")


def test_chunk_parsed_documents_reads_json_and_returns_chunks(tmp_path: Path):
    parsed_docs_path = tmp_path / "parsed_docs.json"
    parsed_docs_path.write_text(
        json.dumps(
            [
                {
                    "source_path": "doc1.html",
                    "url": "https://example.com/1",
                    "title": "Doc One",
                    "headings": ["A"],
                    "text": "Alpha beta gamma. Delta epsilon zeta.",
                },
                {
                    "source_path": "doc2.html",
                    "url": "https://example.com/2",
                    "title": "Doc Two",
                    "headings": ["B"],
                    "text": "Eta theta iota. Kappa lambda mu.",
                },
            ]
        ),
        encoding="utf-8",
    )

    chunks = chunk_parsed_documents(parsed_docs_path, max_words=10)

    assert len(chunks) == 2
    assert [chunk.title for chunk in chunks] == ["Doc One", "Doc Two"]


def test_write_chunks_json_saves_chunked_output(tmp_path: Path):
    parsed_docs_path = tmp_path / "parsed_docs.json"
    out_path = tmp_path / "chunks" / "chunks.json"

    parsed_docs_path.write_text(
        json.dumps(
            [
                {
                    "source_path": "doc1.html",
                    "url": "https://example.com/page",
                    "title": "Chunk Me",
                    "headings": ["Heading"],
                    "text": "This is sentence one. This is sentence two.",
                }
            ]
        ),
        encoding="utf-8",
    )

    written_path = write_chunks_json(out_path, parsed_docs_path, max_words=5)

    assert written_path == out_path
    assert out_path.exists()

    contents = json.loads(out_path.read_text(encoding="utf-8"))
    assert len(contents) >= 1
    assert contents[0]["title"] == "Chunk Me"
    assert "chunk_id" in contents[0]
