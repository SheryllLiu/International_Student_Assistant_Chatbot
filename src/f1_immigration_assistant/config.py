"""Central configuration for the F-1 Immigration Assistant.

Single source of truth for the URL allowlist, domain denylist, text processing
defaults, chunking parameters, retrieval defaults, and the default OpenAI
model name. Do not scatter tunables across the codebase.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

ALLOWED_DOMAIN = "internationalservices.georgetown.edu"

# Approved Georgetown OGS pages. Only URLs in this list are ever fetched.
ALLOWED_URLS: list[str] = [
    "https://internationalservices.georgetown.edu/students/regulations/f-1/",
    "https://internationalservices.georgetown.edu/students/employment/",
    "https://internationalservices.georgetown.edu/students/employment/f-1-on-campus-employment/",
    "https://internationalservices.georgetown.edu/students/employment/cpt/",
    "https://internationalservices.georgetown.edu/students/employment/pre-completion-opt/",
    "https://internationalservices.georgetown.edu/students/employment/post-completion-opt/",
    "https://internationalservices.georgetown.edu/students/employment/post-completion-opt/post-completion-opt-faqs/",
    "https://internationalservices.georgetown.edu/students/employment/optstem/",
    "https://internationalservices.georgetown.edu/travel/travel-out-reentry-f1-j1-students/",
    "https://internationalservices.georgetown.edu/tax/",
    "https://internationalservices.georgetown.edu/tax/file/",
    "https://internationalservices.georgetown.edu/tax/tax-faq/",
    "https://internationalservices.georgetown.edu/tax/noincome/",
    "https://internationalservices.georgetown.edu/resources/key-terms/",
]

# URL fragments we refuse to crawl or index even if they are on the domain.
DENY_PATTERNS: list[str] = [
    "/scholars/",
    "j-1",
    "newsletter",
    "events",
    "immigration-updates",
]

# ---------------------------------------------------------------------------
# Sitemap discovery (offline, review-only)
# ---------------------------------------------------------------------------
# These settings are consumed only by :func:`crawler.discover_urls`. The
# production crawl path (:func:`crawler.crawl`) still reads exclusively from
# ``ALLOWED_URLS`` above and is unaffected by anything in this block.
#
# ``SITEMAP_CANDIDATES`` are fallback guesses used if robots.txt does not
# declare a ``Sitemap:`` directive. They are tried in order.
SITEMAP_CANDIDATES: list[str] = [
    f"https://{ALLOWED_DOMAIN}/sitemap.xml",
    f"https://{ALLOWED_DOMAIN}/wp-sitemap.xml",
    f"https://{ALLOWED_DOMAIN}/sitemap_index.xml",
]

# Nested sitemap indices are followed at most this many levels deep. Guards
# against pathological or adversarial sitemap bombs; 3 is plenty in practice.
SITEMAP_DEPTH_LIMIT: int = 3

ROBOTS_TXT_URL: str = f"https://{ALLOWED_DOMAIN}/robots.txt"


@dataclass(frozen=True)
class PreprocessingConfig:
    """Tunables for the shared query/document preprocessor."""

    lowercase: bool = True
    remove_stopwords: bool = True
    min_token_length: int = 2


@dataclass(frozen=True)
class ChunkingConfig:
    """Heading-aware chunker parameters (token-counted)."""

    target_tokens: int = 220
    overlap_tokens: int = 40
    min_tokens: int = 40


@dataclass(frozen=True)
class RetrievalConfig:
    """BM25 retrieval defaults."""

    top_k: int = 5
    bm25_k1: float = 1.5
    bm25_b: float = 0.75


@dataclass(frozen=True)
class GenerationConfig:
    """OpenAI answer-generation defaults.

    If ``OPENAI_API_KEY`` is not set, the pipeline falls back to a local stub
    generator so everything still works without network access.
    """

    model: str = "gpt-5-mini"
    temperature: float = 0.0
    max_output_tokens: int = 600
    api_key_env: str = "OPENAI_API_KEY"


@dataclass(frozen=True)
class Config:
    """Bundle of all configs used across the package."""

    preprocessing: PreprocessingConfig = field(default_factory=PreprocessingConfig)
    chunking: ChunkingConfig = field(default_factory=ChunkingConfig)
    retrieval: RetrievalConfig = field(default_factory=RetrievalConfig)
    generation: GenerationConfig = field(default_factory=GenerationConfig)

    raw_dir: Path = Path("data/raw")
    index_dir: Path = Path("data/index")
    eval_dir: Path = Path("data/eval")
    sample_dir: Path = Path("data/sample")
    discovery_dir: Path = Path("data/discovery")


DEFAULT_CONFIG = Config()
