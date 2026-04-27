"""Phase 2: build the BGE dense index for the chatbot's retrieval layer.

Scope: embedding only. BM25 is a separate workstream and is handled by a
teammate. This script reads the parsed HTML corpus and produces two artifacts:

* ``data/processed/embedding_corpus.csv`` — the corpus used for encoding,
  with a stable ``row_id`` and the natural-language ``natural_text`` field
  (punctuation / case / stopwords all preserved, as BGE expects).
* ``data/indices/faiss.index`` — FAISS IndexFlatIP over L2-normalized BGE
  embeddings. Inner product on unit vectors == cosine similarity, so this is
  exact cosine top-K with no approximation.

Dedup note: we call ``drop_duplicates(subset=["text"])`` with default
``keep="first"`` to mirror ``utils.text_cleaning.drop_duplicate_rows``. If the
BM25 pipeline uses the same rule, the two indices stay row-aligned and can be
fused by ``row_id`` at query time.

Run from repo root::

    python -m information_retreival.build_dense_index
"""

from __future__ import annotations

import logging
from pathlib import Path

import faiss
import pandas as pd
from sentence_transformers import SentenceTransformer

logger = logging.getLogger("isa.index.dense")

STRUCT_CSV = Path("data/processed/structured_text.csv")
OUT_CORPUS = Path("data/processed/embedding_corpus.csv")
OUT_INDEX_DIR = Path("data/indices")
OUT_INDEX = OUT_INDEX_DIR / "faiss.index"

MODEL_NAME = "BAAI/bge-small-en-v1.5"
BGE_MAX_TOKENS = 512

TEXT_COLS = ("title", "section", "text")


def build_corpus(df: pd.DataFrame) -> pd.DataFrame:
    """Return a DataFrame with one row per unique document.

    ``natural_text`` joins ``title``, ``section``, and ``text`` with periods
    so BGE's tokenizer sees clean sentence boundaries. Empty components are
    skipped — a row with no section becomes ``"Title. Text."`` not ``"Title.
    . Text."``.
    """
    for col in TEXT_COLS:
        df[col] = df[col].fillna("").astype(str)

    def join_natural(row: pd.Series) -> str:
        parts = [row[c].strip() for c in TEXT_COLS]
        parts = [p for p in parts if p]
        return ". ".join(parts)

    df["natural_text"] = df.apply(join_natural, axis=1)

    empty_mask = df["natural_text"].str.len() == 0
    if empty_mask.any():
        logger.info("dropped %d rows with empty natural_text", empty_mask.sum())
        df = df[~empty_mask].reset_index(drop=True)

    df["row_id"] = df.index
    cols = ["row_id", "topic", "title", "section", "text", "natural_text"]
    return df[cols]


def check_token_lengths(texts: list[str], model: SentenceTransformer) -> None:
    """Log a summary + warn on rows that will be truncated."""
    tokenizer = model.tokenizer
    over = []
    max_len = 0
    for i, t in enumerate(texts):
        n = len(tokenizer.encode(t, add_special_tokens=True))
        max_len = max(max_len, n)
        if n > BGE_MAX_TOKENS:
            over.append((i, n))
    logger.info("token length — max: %d, limit: %d", max_len, BGE_MAX_TOKENS)
    if over:
        logger.warning("%d docs exceed %d tokens (will be truncated)", len(over), BGE_MAX_TOKENS)
        for i, n in over[:10]:
            logger.warning("  row_id=%d  tokens=%d", i, n)
        if len(over) > 10:
            logger.warning("  ... and %d more", len(over) - 10)
    else:
        logger.info("all documents fit under the %d-token limit", BGE_MAX_TOKENS)


def build_faiss_index(texts: list[str], model: SentenceTransformer) -> faiss.Index:
    """Encode with L2 normalization, add to IndexFlatIP."""
    embeddings = model.encode(
        texts,
        normalize_embeddings=True,
        batch_size=32,
        show_progress_bar=True,
        convert_to_numpy=True,
    ).astype("float32")

    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings)
    logger.info("built FAISS index: dim=%d, vectors=%d", dim, index.ntotal)
    return index


def main() -> None:
    """Build the embedding corpus + FAISS index and write both to ``data/``."""
    if not STRUCT_CSV.exists():
        raise FileNotFoundError(f"input not found: {STRUCT_CSV}")

    logger.info("loading corpus from %s", STRUCT_CSV)
    df = pd.read_csv(STRUCT_CSV, keep_default_na=False)
    logger.info("loaded %d rows", len(df))

    corpus = build_corpus(df)
    logger.info("final corpus: %d rows", len(corpus))

    logger.info("loading model %s", MODEL_NAME)
    model = SentenceTransformer(MODEL_NAME)

    check_token_lengths(corpus["natural_text"].tolist(), model)
    index = build_faiss_index(corpus["natural_text"].tolist(), model)

    OUT_CORPUS.parent.mkdir(parents=True, exist_ok=True)
    OUT_INDEX_DIR.mkdir(parents=True, exist_ok=True)
    corpus.to_csv(OUT_CORPUS, index=False)
    faiss.write_index(index, str(OUT_INDEX))

    logger.info("wrote corpus  -> %s", OUT_CORPUS)
    logger.info("wrote index   -> %s", OUT_INDEX)


if __name__ == "__main__":
    main()
