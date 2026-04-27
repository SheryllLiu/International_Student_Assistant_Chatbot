"""Minimal Flask demo for the Hybrid (BM25 + Dense, RRF) retriever.

Run from the repo root::

    python app.py

Then open http://127.0.0.1:8080/ in a browser.

The page has one input box, one submit button, and a results area. All
retrieval logic lives in ``information_retreival.hybrid_retrieval.HybridRetriever``;
this file is just the HTTP surface.

Each result shows the fused RRF score plus the per-retriever rank/score so
you can see *why* a doc is ranked where it is — e.g. "BM25 missed this
(rank=None) but dense put it at rank 1" is a concrete sign that hybrid
fusion is pulling its weight.
"""

from __future__ import annotations

import re

from flask import Flask, render_template_string, request
from markupsafe import Markup, escape

from rag_chatbot.information_retrieval.hybrid_retrieval import HybridRetriever
from rag_chatbot.summarizer.gamma4 import Gamma4Summarizer

app = Flask(__name__)

# Load both sub-indexes once at import time so every request reuses the
# same in-memory state instead of re-loading per query.
retriever = HybridRetriever()
summarizer = Gamma4Summarizer()


def _md_to_html(text: str) -> Markup:
    """Convert a small subset of Markdown to safe HTML (bold, italic, newlines)."""
    s = str(escape(text))
    s = re.sub(r"\*\*\*(.*?)\*\*\*", r"<strong><em>\1</em></strong>", s, flags=re.DOTALL)
    s = re.sub(r"\*\*(.*?)\*\*", r"<strong>\1</strong>", s, flags=re.DOTALL)
    s = re.sub(r"\*(.*?)\*", r"<em>\1</em>", s, flags=re.DOTALL)
    s = s.replace("\n", "<br>")
    return Markup(s)


PAGE = """<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>Chatbot For International Student</title>
  <style>
    * { box-sizing: border-box; }

    body {
      font-family: system-ui, -apple-system, "Segoe UI", sans-serif;
      margin: 0;
      padding: 0;
      background: #ffffff;
      color: #222;
    }

    /* Header */
    header {
      background: #79d1da;
      padding: 2.2em 1em;
      text-align: center;
    }
    header h1 {
      margin: 0;
      font-size: 2.4em;
      font-weight: 700;
      letter-spacing: 0.5px;
    }
    header h1 .for { color: #ffffff; }
    header h1 .target { color: #ffffff; }

    /* Main area */
    main {
      max-width: 640px;
      margin: 0 auto;
      padding: 2.5em 1.5em 4em;
    }

    .prompt {
      font-size: 1.1em;
      color: #444;
      margin: 0 0 1.5em;
      line-height: 1.5;
    }

    form.search {
      display: flex;
      flex-direction: column;
      align-items: center;
      gap: 0.9em;
    }
    form.search input[type="text"] {
      width: 100%;
      max-width: 600px;
      padding: 0.85em 1em;
      font-size: 1em;
      border: 1.5px solid #d4d4d4;
      border-radius: 8px;
      outline: none;
      transition: border-color 0.15s ease;
    }
    form.search input[type="text"]:focus {
      border-color: #32a9b6;
    }
    form.search button {
      background: #32a9b6;
      color: #ffffff;
      border: none;
      padding: 0.7em 2.4em;
      font-size: 1em;
      font-weight: 600;
      border-radius: 8px;
      cursor: pointer;
      transition: background 0.15s ease;
    }
    form.search button:hover {
      background: #2a8e99;
    }

    /* Results */
    .results-section {
      margin-top: 3em;
    }
    .results-section h2 {
      color: #32a9b6;
      font-size: 1.4em;
      margin-bottom: 0.6em;
    }
    .empty-msg {
      color: #888;
      font-style: italic;
    }

    /* Collapsible details */
    details.raw-results {
      margin-top: 1.8em;
      border: 1px solid #e2e2e2;
      border-radius: 8px;
      padding: 0.6em 1em;
      background: #fafafa;
    }
    details.raw-results summary {
      cursor: pointer;
      font-weight: 600;
      color: #32a9b6;
      padding: 0.3em 0;
      user-select: none;
    }
    details.raw-results summary:hover {
      color: #2a8e99;
    }
    details.raw-results ol {
      margin-top: 1em;
      padding-left: 1.4em;
    }
    details.raw-results li {
      margin-bottom: 1.5em;
    }
    .scores {
      color: #555;
      font-size: 0.9em;
    }
    .missing {
      color: #aaa;
    }
    code {
      background: #f0f0f0;
      padding: 0.1em 0.35em;
      border-radius: 3px;
      font-size: 0.92em;
    }
  </style>
</head>
<body>
  <header>
    <h1><span class="for">Chatbot For</span> <span class="target">International Student</span></h1>
  </header>

  <main>
    <p class="prompt">
      Hi! If you're preparing to study in the U.S. or already here as an international student, feel free to ask me anything.
    </p>

    <form class="search" method="post" action="/">
      <input type="text" id="q" name="query"
             value="{{ query|e }}"
             placeholder="Put your question here"
             autofocus>
      <button type="submit">Search</button>
    </form>

    {% if submitted and query and results %}
      <section class="results-section">
        <h2>Can this give you some guidance?</h2>
        <p style="line-height:1.7; border: 1px solid #32a9b6; border-radius: 6px; padding: 1em 1.2em;">{{ summary_html }}</p>

        <details class="raw-results">
          <summary>Show retrieved documents and scores ({{ results|length }})</summary>
          <ol>
            {% for r in results %}
              <li>
                <p>
                  <strong>RRF score:</strong> {{ '%.4f' % r.final_score }}
                  &nbsp;<strong>doc_id:</strong> {{ r.doc_id }}
                  &nbsp;<strong>topic:</strong> {{ r.topic|e }}
                </p>
                <p class="scores">
                  <strong>BM25:</strong>
                  {% if r.bm25_rank %}
                    rank={{ r.bm25_rank }}, score={{ '%.3f' % r.bm25_score }}
                  {% else %}
                    <span class="missing">not retrieved</span>
                  {% endif %}
                  &nbsp;|&nbsp;
                  <strong>Dense:</strong>
                  {% if r.dense_rank %}
                    rank={{ r.dense_rank }}, score={{ '%.3f' % r.dense_score }}
                  {% else %}
                    <span class="missing">not retrieved</span>
                  {% endif %}
                </p>
                {% if r.title %}<p><strong>title:</strong> {{ r.title|e }}</p>{% endif %}
                {% if r.section %}<p><strong>section:</strong> {{ r.section|e }}</p>{% endif %}
                {% if r.text %}
                  <p><strong>text:</strong> {{ r.text|e }}</p>
                {% elif r.raw_document %}
                  <p><strong>text:</strong> {{ r.raw_document|e }}</p>
                {% endif %}
              </li>
            {% endfor %}
          </ol>
        </details>
      </section>
    {% endif %}
  </main>
</body>
</html>
"""


@app.route("/", methods=["GET", "POST"])
def index():
    """Render the search page; on POST, run hybrid retrieval and summarization."""
    query = ""
    results = []
    summary = ""
    submitted = False

    if request.method == "POST":
        submitted = True
        query = (request.form.get("query") or "").strip()

        if query:
            results = retriever.search(query, top_k=5)
            summary = summarizer.summarize(query, results)

    return render_template_string(
        PAGE,
        query=query,
        results=results,
        summary_html=_md_to_html(summary),
        submitted=submitted,
    )


if __name__ == "__main__":
    app.run(debug=True)
