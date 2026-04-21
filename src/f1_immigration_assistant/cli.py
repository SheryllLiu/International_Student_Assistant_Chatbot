"""Click-based CLI with a small number of obvious entrypoints.

Commands:

    f1a crawl        Fetch the allowlisted Georgetown OGS pages.
    f1a build-index  Parse crawled HTML, chunk it, and build the BM25 index.
    f1a retrieve     Run BM25 retrieval for a query and print the top hits.
    f1a answer       Retrieve + synthesize a grounded answer.
    f1a evaluate     Run Precision@k / Recall@k / mAP on the judged query set.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import click

from f1_immigration_assistant.config import DEFAULT_CONFIG
from f1_immigration_assistant.logging_config import setup_logging
from f1_immigration_assistant.pipeline import Pipeline


@click.group()
@click.option("--log-level", default="INFO", show_default=True, help="Python logging level.")
def main(log_level: str) -> None:
    """Georgetown F-1 Immigration Compliance Assistant."""

    setup_logging(log_level)


@main.command()
@click.option("--out", "out_dir", type=click.Path(path_type=Path), default=DEFAULT_CONFIG.raw_dir,
              show_default=True, help="Where to save crawled HTML.")
def crawl(out_dir: Path) -> None:
    """Fetch the allowlisted Georgetown OGS pages."""

    saved = Pipeline().crawl(out_dir=out_dir)
    click.echo(f"saved {len(saved)} pages to {out_dir}")


@main.command("build-index")
@click.option("--raw", "raw_dir", type=click.Path(path_type=Path), default=DEFAULT_CONFIG.raw_dir,
              show_default=True)
@click.option("--out", "index_dir", type=click.Path(path_type=Path), default=DEFAULT_CONFIG.index_dir,
              show_default=True)
def build_index(raw_dir: Path, index_dir: Path) -> None:
    """Parse crawled HTML, chunk it, and build the BM25 index."""

    chunks = Pipeline().build_index(raw_dir=raw_dir, index_dir=index_dir)
    click.echo(f"indexed {len(chunks)} chunks -> {index_dir}")


@main.command()
@click.argument("query")
@click.option("--index", "index_dir", type=click.Path(path_type=Path),
              default=DEFAULT_CONFIG.index_dir, show_default=True)
@click.option("-k", "top_k", type=int, default=DEFAULT_CONFIG.retrieval.top_k, show_default=True)
def retrieve(query: str, index_dir: Path, top_k: int) -> None:
    """Run BM25 retrieval for QUERY and print the top hits."""

    pipeline = Pipeline().load_index(index_dir)
    for i, r in enumerate(pipeline.retrieve(query, top_k=top_k), start=1):
        click.echo(f"{i:>2}. score={r.score:.3f}  {r.chunk.url}  |  {r.chunk.heading}")


@main.command()
@click.argument("query")
@click.option("--index", "index_dir", type=click.Path(path_type=Path),
              default=DEFAULT_CONFIG.index_dir, show_default=True)
@click.option("-k", "top_k", type=int, default=DEFAULT_CONFIG.retrieval.top_k, show_default=True)
def answer(query: str, index_dir: Path, top_k: int) -> None:
    """Retrieve evidence and synthesize a grounded answer."""

    pipeline = Pipeline().load_index(index_dir)
    result = pipeline.answer(query, top_k=top_k)
    click.echo(result.answer)
    if result.warning:
        click.echo(f"\n⚠ {result.warning}")
    if result.citations:
        click.echo("\nSources:")
        for url in result.citations:
            click.echo(f"  - {url}")
    click.echo(f"\n(used_llm={result.used_llm})")


@main.command()
@click.option("--index", "index_dir", type=click.Path(path_type=Path),
              default=DEFAULT_CONFIG.index_dir, show_default=True)
@click.option("--queries", "queries_path", type=click.Path(path_type=Path),
              default=Path("data/eval/queries.json"), show_default=True)
@click.option("-k", type=int, default=DEFAULT_CONFIG.retrieval.top_k, show_default=True)
def evaluate(index_dir: Path, queries_path: Path, k: int) -> None:
    """Run Precision@k / Recall@k / mAP on the judged query set."""

    pipeline = Pipeline().load_index(index_dir)
    report = pipeline.evaluate(queries_path, k=k)
    click.echo(
        json.dumps(
            {
                "k": report.k,
                "precision_at_k": round(report.precision_at_k, 4),
                "recall_at_k": round(report.recall_at_k, 4),
                "mean_ap": round(report.mean_ap, 4),
                "n_queries": len(report.per_query),
            },
            indent=2,
        )
    )


if __name__ == "__main__":  # pragma: no cover
    logging.getLogger().setLevel(logging.INFO)
    main()
