"""Final answer-generation layer.

Isolation is intentional: all OpenAI-specific code lives here. The rest of the
package only sees :meth:`OpenAIGenerator.generate_answer`. If ``OPENAI_API_KEY``
is not set (or the import fails), we fall back to a deterministic local stub
so the pipeline and test suite keep working without network access.
"""

from __future__ import annotations

import logging
import os
from typing import TYPE_CHECKING

from f1_immigration_assistant.config import DEFAULT_CONFIG, GenerationConfig
from f1_immigration_assistant.models import AnswerResult, RetrievalResult

if TYPE_CHECKING:  # pragma: no cover — type-only import
    from openai import OpenAI

logger = logging.getLogger(__name__)

_SYSTEM_PROMPT = (
    "You are a careful assistant for Georgetown University F-1 international students. "
    "Answer ONLY from the provided evidence snippets. "
    "If the evidence does not contain the answer, say so and suggest the user contact the "
    "Office of Global Services — do not invent facts or cite outside sources. "
    "Keep answers concise, practical, and specific to F-1 students. "
    "Do not give personalized legal or tax advice; refer the user to OGS for those cases."
)

_RISK_NOTICE = (
    "Your question mentions a situation that may affect F-1 status. "
    "Contact Georgetown OGS promptly to confirm next steps before acting."
)


def _format_evidence(retrieved: list[RetrievalResult]) -> str:
    """Render retrieved chunks as a numbered evidence block for the prompt."""

    lines: list[str] = []
    for i, r in enumerate(retrieved, start=1):
        lines.append(f"[{i}] {r.chunk.title} — {r.chunk.heading}\nURL: {r.chunk.url}\n{r.chunk.text}")
    return "\n\n".join(lines)


def _dedupe_citations(retrieved: list[RetrievalResult]) -> list[str]:
    """Return source URLs in retrieval order, deduplicated."""

    seen: set[str] = set()
    out: list[str] = []
    for r in retrieved:
        if r.chunk.url and r.chunk.url not in seen:
            seen.add(r.chunk.url)
            out.append(r.chunk.url)
    return out


class OpenAIGenerator:
    """Grounded answer generator backed by the OpenAI Chat Completions API.

    Falls back to a local stub that summarizes the top retrieved chunk when no
    API key is present. Tests mock the OpenAI client.
    """

    def __init__(
        self,
        cfg: GenerationConfig | None = None,
        client: "OpenAI | None" = None,
    ) -> None:
        """Construct the generator. Pass ``client`` to inject a mock in tests."""

        self.cfg = cfg or DEFAULT_CONFIG.generation
        self._client = client
        self._api_key = os.environ.get(self.cfg.api_key_env)

    # ------------------------------------------------------------------ utils

    def _get_client(self) -> "OpenAI | None":
        """Lazy-construct and return the OpenAI client, or None if unusable."""

        if self._client is not None:
            return self._client
        if not self._api_key:
            return None
        try:
            from openai import OpenAI
        except ImportError:  # pragma: no cover — openai is a declared dep
            logger.warning("openai SDK not importable; using stub generator")
            return None
        self._client = OpenAI(api_key=self._api_key)
        return self._client

    # ------------------------------------------------------------------ main

    def generate_answer(
        self,
        query: str,
        retrieved_chunks: list[RetrievalResult],
        risk_flags: list[str] | None = None,
    ) -> AnswerResult:
        """Return a grounded :class:`AnswerResult` for ``query``."""

        risk_flags = list(risk_flags or [])
        citations = _dedupe_citations(retrieved_chunks)
        warning = _RISK_NOTICE if risk_flags else None

        if not retrieved_chunks:
            return AnswerResult(
                query=query,
                answer=(
                    "I could not find relevant information in the approved Georgetown OGS "
                    "pages. Please contact the Office of Global Services directly."
                ),
                citations=[],
                risk_flags=risk_flags,
                warning=warning,
                used_llm=False,
            )

        client = self._get_client()
        if client is None:
            answer_text = self._stub_answer(query, retrieved_chunks)
            return AnswerResult(
                query=query,
                answer=answer_text,
                citations=citations,
                risk_flags=risk_flags,
                warning=warning,
                used_llm=False,
            )

        answer_text = self._llm_answer(client, query, retrieved_chunks)
        return AnswerResult(
            query=query,
            answer=answer_text,
            citations=citations,
            risk_flags=risk_flags,
            warning=warning,
            used_llm=True,
        )

    # ---------------------------------------------------------------- backends

    def _llm_answer(
        self,
        client: "OpenAI",
        query: str,
        retrieved: list[RetrievalResult],
    ) -> str:
        """Call the OpenAI Chat Completions API and return the answer text."""

        evidence = _format_evidence(retrieved)
        user_prompt = (
            f"Question: {query}\n\n"
            f"Evidence (from approved Georgetown OGS pages):\n{evidence}\n\n"
            "Write a short grounded answer (2-5 sentences). End with a line 'Sources:' "
            "listing the URLs you used, one per line."
        )

        try:
            resp = client.chat.completions.create(
                model=self.cfg.model,
                temperature=self.cfg.temperature,
                max_tokens=self.cfg.max_output_tokens,
                messages=[
                    {"role": "system", "content": _SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt},
                ],
            )
            return resp.choices[0].message.content.strip()
        except Exception as exc:  # noqa: BLE001 — surface as a graceful fallback
            logger.warning("OpenAI call failed, using stub answer: %s", exc)
            return self._stub_answer(query, retrieved)

    def _stub_answer(self, query: str, retrieved: list[RetrievalResult]) -> str:
        """Deterministic fallback used when no API key is available."""

        top = retrieved[0].chunk
        snippet = top.text.strip()
        if len(snippet) > 500:
            snippet = snippet[:500].rsplit(" ", 1)[0] + "…"
        return (
            f"Based on Georgetown OGS guidance ({top.heading or top.title}): {snippet}\n\n"
            f"See: {top.url}"
        )
