"""Top-level pipeline: crawl -> parse -> chunk -> index -> retrieve -> answer.

One class, a handful of methods, no framework. The pipeline owns the BM25
retriever and the OpenAI generator so callers (CLI, notebooks, tests) have a
single obvious entrypoint.
"""

from __future__ import annotations

import logging
from pathlib import Path

from f1_immigration_assistant.bm25_retriever import BM25Retriever
from f1_immigration_assistant.chunker import chunk_documents
from f1_immigration_assistant.config import DEFAULT_CONFIG, Config
from f1_immigration_assistant.crawler import crawl
from f1_immigration_assistant.evaluation import EvalReport, evaluate
from f1_immigration_assistant.generator import OpenAIGenerator
from f1_immigration_assistant.models import AnswerResult, DocumentChunk, RetrievalResult
from f1_immigration_assistant.parser import parse_raw_dir
from f1_immigration_assistant.query_analysis import analyze_query

logger = logging.getLogger(__name__)


class Pipeline:
    """End-to-end F-1 Immigration Assistant pipeline."""

    def __init__(
        self,
        config: Config | None = None,
        generator: OpenAIGenerator | None = None,
    ) -> None:
        """Create a pipeline. Inject ``generator`` (e.g. with a mock client) in tests."""

        self.config = config or DEFAULT_CONFIG
        self.retriever = BM25Retriever(self.config.retrieval)
        self.generator = generator or OpenAIGenerator(self.config.generation)

    # ----- indexing ---------------------------------------------------------

    def crawl(self, out_dir: Path | str | None = None) -> list[Path]:
        """Crawl the allowlisted Georgetown OGS pages to ``out_dir``."""

        return crawl(out_dir=out_dir or self.config.raw_dir)

    def build_index(
        self,
        raw_dir: Path | str | None = None,
        index_dir: Path | str | None = None,
    ) -> list[DocumentChunk]:
        """Parse raw HTML, chunk it, fit BM25, and persist the index."""

        raw = Path(raw_dir or self.config.raw_dir)
        out = Path(index_dir or self.config.index_dir)

        docs = parse_raw_dir(raw)
        if not docs:
            raise RuntimeError(f"no documents found in {raw} — run `crawl` first")
        chunks = chunk_documents(docs, self.config.chunking)
        self.retriever.fit(chunks)
        self.retriever.save(out)
        return chunks

    def load_index(self, index_dir: Path | str | None = None) -> "Pipeline":
        """Load a previously built BM25 index."""

        self.retriever.load(Path(index_dir or self.config.index_dir))
        return self

    def fit_chunks(self, chunks: list[DocumentChunk]) -> "Pipeline":
        """Build the BM25 index directly from in-memory chunks (test helper)."""

        self.retriever.fit(chunks)
        return self

    # ----- querying ---------------------------------------------------------

    def retrieve(self, query: str, top_k: int | None = None) -> list[RetrievalResult]:
        """Run BM25 retrieval for ``query``."""

        return self.retriever.search(query, top_k=top_k)

    def answer(self, query: str, top_k: int | None = None) -> AnswerResult:
        """Retrieve evidence, then synthesize a grounded answer."""

        analysis = analyze_query(query)
        results = self.retrieve(query, top_k=top_k)
        logger.info(
            "query='%s' intent=%s risks=%s retrieved=%d",
            query, analysis.intent, analysis.risk_flags, len(results),
        )
        return self.generator.generate_answer(
            query=query,
            retrieved_chunks=results,
            risk_flags=analysis.risk_flags,
        )

    # ----- evaluation -------------------------------------------------------

    def evaluate(
        self,
        queries_path: Path | str,
        k: int | None = None,
    ) -> EvalReport:
        """Evaluate the retriever on a small judged query set."""

        return evaluate(self.retriever, queries_path, k=k or self.config.retrieval.top_k)
