# Internation Students Chatbot

A focused Information Retrieval + RAG package that answers compliance questions for
Georgetown University **F-1 international students** using only approved Georgetown
Office of Global Services (OGS) public pages.

The heart of the package is a clean BM25 retrieval pipeline; OpenAI is used only for
the final answer-synthesis step.

## Team

- Tianwei(Sophie) Shi
- Shuchen(Sheryll) Liu
- Hima Bathula
- Robert George


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
...................
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

## Package layout

```

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
