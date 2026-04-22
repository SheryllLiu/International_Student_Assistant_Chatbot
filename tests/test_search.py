from summerizer.utils.search import format_result
from summerizer.utils.retriever import RetrievalResult


def test_format_result_includes_main_fields():
    result = RetrievalResult(
        chunk_id="chunk_1",
        url="https://example.com/page",
        title="Example Title",
        chunk_index=2,
        text="Example chunk text.",
        score=3.14159,
    )

    formatted = format_result(result, rank=1)

    assert "Result 1" in formatted
    assert "Title: Example Title" in formatted
    assert "URL: https://example.com/page" in formatted
    assert "Score: 3.1416" in formatted
    assert "Chunk Index: 2" in formatted
    assert "Text: Example chunk text." in formatted
