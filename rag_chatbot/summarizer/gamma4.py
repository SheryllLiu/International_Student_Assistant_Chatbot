"""Local Ollama-backed summarizer that turns retrieved passages into a short answer."""

from __future__ import annotations

import logging

import requests

logger = logging.getLogger("isa.summarizer")


class Gamma4Summarizer:
    """Summarize retrieved passages by calling a locally running Ollama model."""

    def __init__(self):
        """Configure the Ollama endpoint and model name."""
        # The class name reflects the project plan; the actual Ollama tag is gemma4:e2b.
        self.model = "gemma4:e2b"
        self.url = "http://localhost:11434/api/generate"

    def summarize(self, query: str, results: list[dict]) -> str:
        """Generate a short answer to ``query`` grounded in ``results``.

        Falls back to the top result's text if the Ollama request fails.
        """
        if not results:
            logger.warning("summarize called with empty results for query %r", query)
            return "No results were found, so there is nothing to summarize."

        context = self.build_context(results)

        prompt = f"""
You are a helpful summarizer.

Answer the user's question using ONLY the retrieved information below.
Do not make up facts.
Keep the answer short and clear.

Question:
{query}

Retrieved information:
{context}

Summary:
"""

        try:
            response = requests.post(
                self.url,
                json={
                    "model": self.model,
                    "prompt": prompt,
                    "stream": False,
                },
                timeout=60,
            )

            response.raise_for_status()
            data = response.json()
            return data.get("response", "").strip()

        except Exception as e:
            logger.warning("Ollama request failed for query %r: %s", query, e)
            first_result = results[0]
            text = first_result.get("text") or first_result.get("raw_document") or ""
            return "Ollama is not running, so here is the top result instead:\n\n" + text[:500]

    def build_context(self, results: list[dict]) -> str:
        """Format the top-3 retrieval hits into a compact prompt context block."""
        context_parts = []

        # Use only the top 3 results to keep the prompt simple.
        for result in results[:3]:
            doc_id = result.get("doc_id", "")
            title = result.get("title", "")
            section = result.get("section", "")
            text = result.get("text") or result.get("raw_document") or ""

            one_result_text = (
                f"Document ID: {doc_id}\nTitle: {title}\nSection: {section}\nText: {text[:800]}\n"
            )

            context_parts.append(one_result_text)

        return "\n---\n".join(context_parts)
