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
 
    python retrieval/build_dense_index.py
"""
from __future__ import annotations
 
from pathlib import Path
 
import faiss
import pandas as pd
from sentence_transformers import SentenceTransformer
 
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
    before = len(df)
    df = df.drop_duplicates(subset=["text"]).reset_index(drop=True)
    dropped = before - len(df)
    if dropped:
        print(f"[info] dropped {dropped} duplicate rows on 'text'")
 
    for col in TEXT_COLS:
        df[col] = df[col].fillna("").astype(str)
 
    def join_natural(row: pd.Series) -> str:
        parts = [row[c].strip() for c in TEXT_COLS]
        parts = [p for p in parts if p]
        return ". ".join(parts)
 
    df["natural_text"] = df.apply(join_natural, axis=1)
 
    empty_mask = df["natural_text"].str.len() == 0
    if empty_mask.any():
        print(f"[info] dropped {empty_mask.sum()} rows with empty natural_text")
        df = df[~empty_mask].reset_index(drop=True)
 
    df["row_id"] = df.index
    cols = ["row_id", "topic", "title", "section", "text", "natural_text"]
    return df[cols]
 
 
def check_token_lengths(texts: list[str], model: SentenceTransformer) -> None:
    """Print a one-line summary + warn on rows that will be truncated."""
    tokenizer = model.tokenizer
    over = []
    max_len = 0
    for i, t in enumerate(texts):
        n = len(tokenizer.encode(t, add_special_tokens=True))
        max_len = max(max_len, n)
        if n > BGE_MAX_TOKENS:
            over.append((i, n))
    print(f"[info] token length — max: {max_len}, limit: {BGE_MAX_TOKENS}")
    if over:
        print(f"[warn] {len(over)} docs exceed {BGE_MAX_TOKENS} tokens (will be truncated):")
        for i, n in over[:10]:
            print(f"       row_id={i}  tokens={n}")
        if len(over) > 10:
            print(f"       ... and {len(over) - 10} more")
    else:
        print("[ok] all documents fit under the 512-token limit")
 
 
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
    print(f"[ok] built FAISS index: dim={dim}, vectors={index.ntotal}")
    return index
 
 
def main() -> None:
    if not STRUCT_CSV.exists():
        raise FileNotFoundError(f"input not found: {STRUCT_CSV}")
 
    df = pd.read_csv(STRUCT_CSV, keep_default_na=False)
    print(f"[info] loaded {len(df)} rows from {STRUCT_CSV}")
 
    corpus = build_corpus(df)
    print(f"[info] final corpus: {len(corpus)} rows")
 
    model = SentenceTransformer(MODEL_NAME)
    print(f"[info] loaded model {MODEL_NAME}")
 
    check_token_lengths(corpus["natural_text"].tolist(), model)
    index = build_faiss_index(corpus["natural_text"].tolist(), model)
 
    OUT_CORPUS.parent.mkdir(parents=True, exist_ok=True)
    OUT_INDEX_DIR.mkdir(parents=True, exist_ok=True)
    corpus.to_csv(OUT_CORPUS, index=False)
    faiss.write_index(index, str(OUT_INDEX))
 
    print(f"[ok] wrote corpus  -> {OUT_CORPUS}")
    print(f"[ok] wrote index   -> {OUT_INDEX}")
 
 
if __name__ == "__main__":
    main()

