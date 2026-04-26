# Georgetown F-1 Immigration Compliance Assistant

A focused Information Retrieval + RAG package that answers compliance questions for
Georgetown University **F-1 international students** using only approved Georgetown
Office of Global Services (OGS) public pages.

The heart of the package is a clean BM25 retrieval pipeline; OpenAI is used only for
the final answer-synthesis step.

## Team

- TEAM_MEMBER_1 (full name)
- TEAM_MEMBER_2 (full name)
- TEAM_MEMBER_3 (full name)
- TEAM_MEMBER_4 (full name)

> Replace the placeholders above with the full names of all team members before
> submission.

## Scope

**In scope** (F-1 students only):

- Maintaining legal F-1 status
- Employment authorization during study and around graduation (on-campus, CPT, OPT, STEM OPT)
- Travel and re-entry compliance
- F-1-relevant tax filing guidance **as provided by Georgetown OGS**

**Out of scope** (by design):

- J-1 students or scholars
- H-1B / O-1 / green-card topics
- Generic USCIS or IRS content
- Personalized legal or tax advice
- Any source outside the approved Georgetown OGS allowlist

The corpus is restricted to a short allowlist of OGS URLs (see
`src/f1_immigration_assistant/config.py`). Everything else is explicitly denied.

## Why BM25

BM25 is the strongest classical retriever for a small, well-curated corpus like this.
It is fast, interpretable, requires no training, and fits cleanly with the course
content. Retrieval quality matters more than fancy generation here, so we keep the
retrieval pipeline as the core of the project and use the LLM only to phrase the
final grounded answer.

## Architecture

```
            approved OGS URLs
                   │
            crawler.py  ──►  raw HTML  ──►  parser.py  ──►  SourceDocument
                                                                 │
                                                          chunker.py
                                                                 │
                                                          DocumentChunk[]
                                                                 │
                       preprocessing.py  ────────────►   bm25_retriever.py
                                                                 │
                                              query_analysis.py  │
                                                       │         │
                                                       ▼         ▼
                                                        pipeline.py
                                                             │
                                                     ┌───────┴───────┐
                                                     ▼               ▼
                                                evaluation.py   generator.py
                                                                  (OpenAI)
```

A program architecture diagram (made with draw.io or similar software) will be
added below when exported.

> **\[ARCHITECTURE DIAGRAM PLACEHOLDER\]** — replace this line with
> `![architecture](architecture.png)` once `architecture.png` is committed.

## Installation

This project uses **uv** for dependency and environment management. A conda
`environment.yml` is also provided.

### With uv (recommended)

```bash
# install uv if needed: https://docs.astral.sh/uv/
uv venv
source .venv/bin/activate
uv pip install -e ".[dev]"
```

### With conda

```bash
conda env create -f environment.yml
conda activate f1-immigration-assistant
```

### OpenAI key (optional)

Answer generation uses the OpenAI API when an API key is present:

```bash
export OPENAI_API_KEY="sk-..."
```

If the key is not set, the package falls back to a deterministic local stub
generator so the full pipeline (and the test suite) still runs without network
access.

## Quick start

### 1. Crawl approved OGS pages

```bash
f1a crawl --out data/raw
```

Only URLs on the allowlist in `config.py` are fetched. J-1 / scholar / external
pages are never fetched.

### 2. Build the BM25 index

```bash
f1a build-index --raw data/raw --out data/index
```

### 3. Ask a question

```bash
f1a answer "How many hours can I work on campus during the semester?"
```

### 4. Run the evaluation

```bash
f1a evaluate --index data/index --queries data/eval/queries.json
```

A one-shot demo script that runs the whole pipeline on bundled sample data is
also available:

```bash
python scripts/run_demo.py
```

## Evaluation Metrics:
Precision@5
Recall@5
MAP
F1 Score
ROUGE
BERTScore

## Package layout

```
final_group_project/
├── pyproject.toml            # modern packaging, ruff + pytest config
├── environment.yml           # conda alternative
├── README.md
├── data/
│   ├── sample/               # tiny bundled HTML for offline demo / tests
│   └── eval/                 # small query set + relevance judgments
├── scripts/                  # thin wrappers around the CLI commands
│   ├── crawl_site.py
│   ├── build_index.py
│   ├── evaluate.py
│   └── run_demo.py
├── src/f1_immigration_assistant/
│   ├── config.py             # allowlist URLs, retrieval defaults
│   ├── logging_config.py     # one-call logging setup
│   ├── models.py             # dataclasses: SourceDocument, DocumentChunk, ...
│   ├── preprocessing.py      # single shared tokenize/normalize pipeline
│   ├── crawler.py            # constrained allowlist crawler
│   ├── parser.py             # HTML -> SourceDocument
│   ├── chunker.py            # heading-aware chunking
│   ├── bm25_retriever.py     # main retriever (rank-bm25)
│   ├── query_analysis.py     # lightweight intent / risk hints
│   ├── evaluation.py         # Precision@k, Recall@k, optional mAP
│   ├── generator.py          # OpenAI RAG layer with local stub fallback
│   ├── pipeline.py           # top-level end-to-end workflow
│   ├── cli.py                # click-based CLI
│   └── utils.py              # small shared helpers
└── tests/                    # pytest suite
```

## Testing

```bash
pytest
```

Tests use bundled sample documents in `data/sample/` and mock the OpenAI client,
so no network access or API key is required.

## Reproducibility

From a fresh clone:

```bash
uv venv && source .venv/bin/activate
uv pip install -e ".[dev]"
pytest
python scripts/run_demo.py
```

That is intended to be enough for an instructor or a new team member to
reproduce the core behavior and the evaluation.
