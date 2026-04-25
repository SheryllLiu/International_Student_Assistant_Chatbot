"""Minimal Flask demo for the Hybrid (BM25 + Dense, RRF) retriever.

Run from the repo root::

    python app.py

Then open http://127.0.0.1:5000/ in a browser.

The page has one input box, one submit button, and a results area. All
retrieval logic lives in ``information_retreival.hybrid_retrieval.HybridRetriever``;
this file is just the HTTP surface.

Each result shows the fused RRF score plus the per-retriever rank/score so
you can see *why* a doc is ranked where it is — e.g. "BM25 missed this
(rank=None) but dense put it at rank 1" is a concrete sign that hybrid
fusion is pulling its weight.
"""
from __future__ import annotations

from flask import Flask, render_template_string, request

from information_retreival.hybrid_retrieval import HybridRetriever

from summarizer.utils.gamma4 import Gamma4Summarizer

app = Flask(__name__)

# Load both sub-indexes once at import time so every request reuses the
# same in-memory state instead of re-loading per query.
retriever = HybridRetriever()
summarizer = Gamma4Summarizer()

PAGE = """<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>Hybrid Retrieval Demo (BM25 + Dense, RRF)</title>
  <style>
    body { font-family: system-ui, sans-serif; max-width: 900px; margin: 2em auto; padding: 0 1em; }
    li { margin-bottom: 1.5em; }
    .scores { color: #555; font-size: 0.9em; }
    .missing { color: #aaa; }
    code { background: #f4f4f4; padding: 0.1em 0.3em; border-radius: 3px; }
  </style>
</head>
<body>
  <h1>Hybrid Retrieval Demo</h1>
  <p><small>BM25 + BGE dense, fused by Reciprocal Rank Fusion (k=60).</small></p>

  <form method="post" action="/">
    <label for="q">Query:</label>
    <input type="text" id="q" name="query" value="{{ query|e }}" size="60" autofocus>
    <button type="submit">Search</button>
  </form>

  {% if submitted %}
    {% if not query %}
      <p><em>Please enter a query.</em></p>
    {% elif not results %}
      <p><em>No results found.</em></p>
    {% else %}
      <h2>Summary</h2>
      <p>{{ summary|e }}</p>
      <h2>Results for: <code>{{ query|e }}</code></h2>
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
    {% endif %}
  {% endif %}
</body>
</html>
"""


@app.route("/", methods=["GET", "POST"])
def index():
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
        summary=summary,
        submitted=submitted,
    )


if __name__ == "__main__":
    app.run(debug=True)
