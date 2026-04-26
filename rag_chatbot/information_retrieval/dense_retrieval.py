"""Dense semantic retriever over BGE embeddings + FAISS.

Loads the FAISS index produced by ``build_dense_index.py`` and the paired
``embedding_corpus.csv``. At query time the query is encoded with the BGE
query instruction prefix, L2-normalized, and fed to FAISS; results are the
top-k documents by cosine similarity (exact — ``IndexFlatIP`` on unit
vectors is not an approximation).

``row_id`` in ``embedding_corpus.csv`` equals the FAISS row index, which in
turn equals the ``doc_id`` used by ``BM25Retriever`` (both pipelines read the
same de-duplicated 148-row source). That shared integer key is what
``HybridRetriever`` fuses on.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import faiss
import pandas as pd
from sentence_transformers import SentenceTransformer

from rag_chatbot.information_retrieval.build_dense_index import MODEL_NAME

DEFAULT_INDEX_PATH = "data/indices/faiss.index"
DEFAULT_CORPUS_PATH = "data/processed/embedding_corpus.csv"

# BGE v1.5 recommends prepending this instruction to the *query* only.
# Documents were encoded without a prefix in build_dense_index.py — keep it
# asymmetric, that is how the model was trained.
QUERY_INSTRUCTION = "Represent this sentence for searching relevant passages: "

RESULT_FIELDS = ("topic", "title", "section", "text")


class DenseRetriever:
    """Cosine top-k retriever over the BGE FAISS index."""

    def __init__(
        self,
        index_path: str | Path = DEFAULT_INDEX_PATH,
        corpus_path: str | Path = DEFAULT_CORPUS_PATH,
        model_name: str = MODEL_NAME,
    ):
        self.index = faiss.read_index(str(index_path))
        self.corpus = pd.read_csv(corpus_path, keep_default_na=False)
        self.model = SentenceTransformer(model_name)

    def search(self, query: str, top_k: int = 10) -> list[dict[str, Any]]:
        """Return the top-``top_k`` documents for ``query`` by cosine score."""
        if not isinstance(query, str) or not query.strip():
            return []

        emb = self.model.encode(
            [QUERY_INSTRUCTION + query],
            normalize_embeddings=True,
            convert_to_numpy=True,
        ).astype("float32")

        scores, ids = self.index.search(emb, top_k)

        results: list[dict[str, Any]] = []
        for score, row_id in zip(scores[0], ids[0], strict=True):
            # FAISS pads with -1 when the index has fewer than top_k vectors.
            if row_id < 0:
                continue
            row = self.corpus.iloc[int(row_id)]
            out: dict[str, Any] = {
                "doc_id": int(row_id),
                "final_score": float(score),
                "dense_score": float(score),
            }
            for field in RESULT_FIELDS:
                if field in row:
                    out[field] = row[field]
            results.append(out)
        return results
