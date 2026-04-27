"""CLI entry point for the International Student Assistant.

Usage::

    uv run isa --help
    uv run isa serve
    uv run isa serve --port 8080 --debug
    uv run isa evaluate
    uv run isa evaluate --no-summarizer
    uv run isa evaluate --no-ir
"""

from __future__ import annotations

import logging

import click

_LOG_FORMAT = "%(asctime)s  %(name)-30s  %(levelname)-8s  %(message)s"
_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"


@click.group()
@click.option(
    "--log-level",
    default="INFO",
    type=click.Choice(["DEBUG", "INFO", "WARNING", "ERROR"], case_sensitive=False),
    show_default=True,
    help="Log verbosity for isa.* loggers.",
)
@click.pass_context
def main(ctx: click.Context, log_level: str) -> None:
    """International Student Immigration Assistant."""
    logging.basicConfig(format=_LOG_FORMAT, datefmt=_DATE_FORMAT)
    logging.getLogger("isa").setLevel(log_level.upper())


@main.command()
@click.option("--host", default="127.0.0.1", show_default=True, help="Host to bind.")
@click.option("--port", default=8080, show_default=True, help="Port to bind.")
@click.option("--debug", is_flag=True, default=False, help="Enable Flask debug mode.")
def serve(host: str, port: int, debug: bool) -> None:
    """Start the web demo (BM25 + dense hybrid retrieval + summarizer)."""
    from rag_chatbot.web import app  # noqa: PLC0415

    click.echo(f"Starting server at http://{host}:{port}/  (Ctrl+C to quit)")
    app.run(host=host, port=port, debug=debug)


@main.command()
@click.option(
    "--ir/--no-ir", default=True, show_default=True, help="Run IR evaluation (BM25 vs Hybrid)."
)
@click.option(
    "--summarizer/--no-summarizer",
    default=True,
    show_default=True,
    help="Run summarizer evaluation (ROUGE + BERTScore).",
)
def evaluate(ir: bool, summarizer: bool) -> None:
    """Evaluate retrieval and/or summarizer quality."""
    if not ir and not summarizer:
        raise click.UsageError("At least one of --ir or --summarizer must be enabled.")

    if ir:
        click.echo("\n── IR Evaluation (BM25 vs Hybrid) ──────────────────────────────")
        from rag_chatbot.eval.eval_IR import main as ir_main  # noqa: PLC0415

        ir_main()

    if summarizer:
        click.echo("\n── Summarizer Evaluation (ROUGE + BERTScore) ───────────────────")
        from rag_chatbot.eval.eval_summarizer import main as sum_main  # noqa: PLC0415

        sum_main()
