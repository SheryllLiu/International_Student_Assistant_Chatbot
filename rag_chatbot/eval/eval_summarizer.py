"""Evaluate summarizer output against gold references using ROUGE and BERTScore."""

import json
from pathlib import Path

from bert_score import score as bertscore
from rouge_score import rouge_scorer

from rag_chatbot.information_retrieval.hybrid_retrieval import HybridRetriever
from rag_chatbot.summarizer.gamma4 import Gamma4Summarizer

DATA_PATH = Path("data/eval/summaries.json")
TOP_K = 5


def load_data():
    """Load the gold (query, reference) pairs from ``DATA_PATH``."""
    with open(DATA_PATH) as f:
        return json.load(f)


def compute_rouge(predictions, references):
    """Return mean ROUGE-1/2/L F1 scores over the prediction/reference pairs."""
    scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)
    scores = {"rouge1": [], "rouge2": [], "rougeL": []}
    for pred, ref in zip(predictions, references, strict=True):
        s = scorer.score(ref, pred)
        scores["rouge1"].append(s["rouge1"].fmeasure)
        scores["rouge2"].append(s["rouge2"].fmeasure)
        scores["rougeL"].append(s["rougeL"].fmeasure)
    return {k: sum(v) / len(v) for k, v in scores.items()}


def compute_bertscore(predictions, references):
    """Return mean BERTScore precision/recall/F1 over the prediction/reference pairs."""
    P, R, F1 = bertscore(predictions, references, lang="en")
    return {
        "precision": P.mean().item(),
        "recall": R.mean().item(),
        "f1": F1.mean().item(),
    }


def main():
    """Run the summarizer over the gold queries and print ROUGE + BERTScore."""
    data = load_data()

    retriever = HybridRetriever()
    summarizer = Gamma4Summarizer()

    predictions = []
    references = []

    for d in data:
        query = d["query"]
        reference = d["reference"]

        hits = retriever.search(query, top_k=TOP_K)
        prediction = summarizer.summarize(query, hits)

        predictions.append(prediction)
        references.append(reference)

    rouge_results = compute_rouge(predictions, references)
    bert_results = compute_bertscore(predictions, references)

    print("\nROUGE:")
    for k, v in rouge_results.items():
        print(f"{k}: {v:.4f}")

    print("\nBERTScore:")
    for k, v in bert_results.items():
        print(f"{k}: {v:.4f}")


if __name__ == "__main__":
    main()
