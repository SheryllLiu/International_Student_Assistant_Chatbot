"""Split parsed documents into smaller chunks for retrieval.

Think of this file as the "make the documents searchable" step.

The parser gives us one cleaned document per page, but search usually works
better on smaller pieces than on a whole long page. This file turns each
parsed document into several shorter chunks.
"""

from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path

try:
    from summerizer.utils.logger import write_log
except ModuleNotFoundError:
    from logger import write_log  # type: ignore

PROJECT_ROOT = Path(__file__).resolve().parents[2]
PARSED_DOCS_PATH = PROJECT_ROOT / "data" / "parsed" / "parsed_docs.json"
CHUNKS_PATH = PROJECT_ROOT / "data" / "chunks" / "chunks.json"

# A chunk should be large enough to contain useful context, but not so large
# that one chunk turns back into a whole page.
DEFAULT_MAX_WORDS = 120


@dataclass
class DocumentChunk:
    """One smaller piece of a parsed document."""

    chunk_id: str
    source_path: str
    url: str
    title: str
    headings: list[str]
    chunk_index: int
    text: str
    word_count: int


def normalize_whitespace(text: str) -> str:
    """Turn repeated whitespace into single spaces."""

    return re.sub(r"\s+", " ", text).strip()


def count_words(text: str) -> int:
    """Count the words in a piece of text."""

    cleaned_text = normalize_whitespace(text)
    if not cleaned_text:
        return 0
    return len(cleaned_text.split())


def split_into_sentences(text: str) -> list[str]:
    """Split text into rough sentence-sized pieces.

    This is a simple punctuation-based splitter. It is not perfect English
    sentence parsing, but it is easy to understand and good enough here.
    """

    cleaned_text = normalize_whitespace(text)
    if not cleaned_text:
        return []

    sentence_parts = re.split(r"(?<=[.!?])\s+", cleaned_text)
    return [part.strip() for part in sentence_parts if part.strip()]


def split_long_text_by_words(text: str, max_words: int) -> list[str]:
    """Break one long text block into smaller word-based pieces."""

    words = normalize_whitespace(text).split()
    if not words:
        return []

    pieces: list[str] = []
    start = 0
    while start < len(words):
        end = start + max_words
        pieces.append(" ".join(words[start:end]))
        start = end
    return pieces


def build_text_chunks(text: str, max_words: int = DEFAULT_MAX_WORDS) -> list[str]:
    """Group sentences into chunks that stay under the word limit.

    The rule is simple:
    - keep adding sentences to the current chunk
    - if the chunk would get too long, start a new chunk
    - if one sentence is too long by itself, split it by words
    """

    sentences = split_into_sentences(text)
    if not sentences:
        return []

    chunks: list[str] = []
    current_chunk_parts: list[str] = []
    current_chunk_word_count = 0

    for sentence in sentences:
        sentence_word_count = count_words(sentence)

        # A single sentence can sometimes be huge. In that case, break it into
        # smaller pieces so we still stay near the word limit.
        if sentence_word_count > max_words:
            if current_chunk_parts:
                chunks.append(" ".join(current_chunk_parts))
                current_chunk_parts = []
                current_chunk_word_count = 0

            chunks.extend(split_long_text_by_words(sentence, max_words))
            continue

        # If this sentence would push us over the size limit, save the current
        # chunk and begin a new one.
        if current_chunk_word_count + sentence_word_count > max_words and current_chunk_parts:
            chunks.append(" ".join(current_chunk_parts))
            current_chunk_parts = [sentence]
            current_chunk_word_count = sentence_word_count
        else:
            current_chunk_parts.append(sentence)
            current_chunk_word_count += sentence_word_count

    if current_chunk_parts:
        chunks.append(" ".join(current_chunk_parts))

    return [normalize_whitespace(chunk) for chunk in chunks if normalize_whitespace(chunk)]


def load_parsed_documents(parsed_docs_path: Path | str = PARSED_DOCS_PATH) -> list[dict]:
    """Load the parsed document JSON file."""

    parsed_docs_path = Path(parsed_docs_path)
    return json.loads(parsed_docs_path.read_text(encoding="utf-8"))


def make_chunk_id(title: str, chunk_index: int) -> str:
    """Create a predictable chunk ID from the title and position."""

    safe_title = re.sub(r"[^a-zA-Z0-9]+", "_", title).strip("_").lower()
    if not safe_title:
        safe_title = "document"
    return f"{safe_title}_chunk_{chunk_index}"


def chunk_document(document: dict, max_words: int = DEFAULT_MAX_WORDS) -> list[DocumentChunk]:
    """Turn one parsed document into a list of smaller chunks."""

    text_chunks = build_text_chunks(document.get("text", ""), max_words=max_words)
    output_chunks: list[DocumentChunk] = []
    title = document.get("title", "")
    source_path = document.get("source_path", "")
    url = document.get("url", "")
    headings = list(document.get("headings", []))

    for chunk_index, chunk_text in enumerate(text_chunks):
        output_chunks.append(
            DocumentChunk(
                chunk_id=make_chunk_id(title, chunk_index),
                source_path=source_path,
                url=url,
                title=title,
                headings=headings,
                chunk_index=chunk_index,
                text=chunk_text,
                word_count=count_words(chunk_text),
            )
        )

    return output_chunks


def chunk_parsed_documents(
    parsed_docs_path: Path | str = PARSED_DOCS_PATH,
    max_words: int = DEFAULT_MAX_WORDS,
) -> list[DocumentChunk]:
    """Chunk every parsed document in the parsed JSON file."""

    parsed_documents = load_parsed_documents(parsed_docs_path)
    all_chunks: list[DocumentChunk] = []

    for document in parsed_documents:
        all_chunks.extend(chunk_document(document, max_words=max_words))

    return all_chunks


def write_chunks_json(
    out_path: Path | str = CHUNKS_PATH,
    parsed_docs_path: Path | str = PARSED_DOCS_PATH,
    max_words: int = DEFAULT_MAX_WORDS,
) -> Path:
    """Chunk the parsed documents and save them as JSON."""

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    write_log("chunker.py started")
    chunks = chunk_parsed_documents(parsed_docs_path, max_words=max_words)
    json_ready_chunks = [asdict(chunk) for chunk in chunks]

    out_path.write_text(
        json.dumps(json_ready_chunks, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    write_log(f"chunker.py created {len(chunks)} chunks in {out_path}")
    return out_path


if __name__ == "__main__":
    output = write_chunks_json()
    print(f"wrote chunks to {output}")
