"""Minimal Flask demo for the BM25 retriever.

Run from the repo root::

    python app.py

Then open http://127.0.0.1:5000/ in a browser.

The page has one input box, one submit button, and a results area. All
retrieval logic lives in ``utils.information_retrieval.BM25Retriever``; this
file is just the HTTP surface.
"""
from __future__ import annotations

from flask import Flask, render_template_string, request

from information_retreival.information_retrieval import BM25Retriever

app = Flask(__name__)

# Load the index once at import time so every request reuses the same
# in-memory state instead of unpickling per query.
retriever = BM25Retriever()

PAGE = """<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>BM25 Demo</title>
</head>
<body>
  <h1>BM25 Demo</h1>

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
      <h2>Results for: <code>{{ query|e }}</code></h2>
      <ol>
        {% for r in results %}
          <li>
            <p>
              <strong>score:</strong> {{ '%.4f' % r.score }}
              &nbsp;<strong>doc_id:</strong> {{ r.doc_id }}
              &nbsp;<strong>topic:</strong> {{ r.topic|e }}
            </p>
            {% if r.title %}<p><strong>title:</strong> {{ r.title|e }}</p>{% endif %}
            {% if r.section %}<p><strong>section:</strong> {{ r.section|e }}</p>{% endif %}
            {% if r.text %}
              <p><strong>text:</strong> {{ r.text|e }}</p>
            {% else %}
              <p><strong>cleaned_document:</strong> {{ r.cleaned_document|e }}</p>
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
    results: list = []
    submitted = False
    if request.method == "POST":
        submitted = True
        query = (request.form.get("query") or "").strip()
        if query:
            results = retriever.search(query, top_k=2)
    return render_template_string(
        PAGE, query=query, results=results, submitted=submitted
    )


if __name__ == "__main__":
    app.run(debug=True)
