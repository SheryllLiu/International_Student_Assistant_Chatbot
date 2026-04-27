# F-1 Immigration Assistant



A beginner friendly web application (Flask app) that answers F-1 questions
by combining: 

- BM25 retreival
- Dense Retreival with BGE embeddings + FAISS
- Hybrid retreival using Reciprocal Rank Fustion (RRF)
- Local summarization with Ollama and gemma4:e2b

The app takes a user question, retreives the most relevant documents
from the corpus, and then uses the top results to generate a short
grounded summary. 


## Team

- Tianwei Shi
- Sheryll Liu
- Robert George
- Hima Bathula

# Project Flow

1. The user enters a question through the Web App
2. HybridRetreiver searches the corpus using BM25, Dense Retreival
3. Two ranked lists are fused with RRF
4. Top 3 retreived results are passed to the summarizer 
5. The summarizer sends the query plus retreived evidence to Ollama
6. The Web App displays a short summary, as well as the retreived documents + scores

# Current Structure 
- app.py is the Flask web app and UI
- information_retreival/ Retrieval pipeline
- Summarizer/utils/gamma4.py Ollama summarizer using gemma4:e2b
- data/processed data, indices, and raw files 

# Requirements
- Python 3
- Ollama
- Model: gemma4:e2b

## Python packages used by the project 
- flask
- requests
- pandas
- faiss-cpu
- sentence-transformers
  
# Install Ollama
curl -fsSL https://ollama.com/install.sh | sh

## Then pull the model: 
ollama pull gemma4:e2b

## You can test it with:
ollama run gemma4:e2b

## Install Python Dependencies
python3 -m venv .venv
source .venv/bin/activate
python -m pip install flask requests pandas faiss-cpu sentence-transformers

# Run the App
python3 app.py

## Then open url









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



