"""Georgetown F-1 Immigration Compliance Assistant package."""

from f1_immigration_assistant.models import (
    AnswerResult,
    DocumentChunk,
    QueryAnalysis,
    RetrievalResult,
    SourceDocument,
)
from f1_immigration_assistant.pipeline import Pipeline

__all__ = [
    "AnswerResult",
    "DocumentChunk",
    "Pipeline",
    "QueryAnalysis",
    "RetrievalResult",
    "SourceDocument",
]

__version__ = "0.1.0"
