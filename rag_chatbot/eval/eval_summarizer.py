
import json
from pathlib import Path
from rouge_score import rouge_scorer
from bert_score import score as bertscore

#Get the retriever and summarizer
from rag_chatbot.information_retrieval.hybrid_retrieval import HybridRetriever
from rag_chatbot.summarizer.gamma4 import Gamma4Summarizer

DATA_PATH = Path('data/eval/summaries.json')
TOP_K = 5

def load_data():
    with open(DATA_PATH, "r") as f:
        return json.load(f)


def compute_rouge(predictions, references):
    scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)
    scores = {"rouge1": [], "rouge2": [], "rougeL": []}
    for pred, ref in zip(predictions, references):
        s = scorer.score(ref, pred)
        scores["rouge1"].append(s["rouge1"].fmeasure)
        scores["rouge2"].append(s["rouge2"].fmeasure)
        scores["rougeL"].append(s["rougeL"].fmeasure)
    return {k: sum(v) / len(v) for k, v in scores.items()}


def compute_bertscore(predictions, references):
    P, R, F1 = bertscore(predictions, references, lang="en")
    return {
        "precision": P.mean().item(),
        "recall": R.mean().item(),
        "f1": F1.mean().item(),
    }


def main():
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
