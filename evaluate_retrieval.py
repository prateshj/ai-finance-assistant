
import json
from pathlib import Path
from statistics import mean
from retrieval import retrieve

BASE_DIR = Path(__file__).resolve().parent

DATASET_PATH = (
    BASE_DIR / "evaluation" / "golden_dataset.json"
)

RESULTS_PATH = (
    BASE_DIR / "evaluation" / "retrieval_results.json"
)

K_VALUES = [1, 3, 5]


def calculate_metrics(results, expected_sources, k):

    # Extract source filenames from top K chunks
    ranked_sources = [
        result["chunk"]["source"]
        for result in results[:k]
    ]

    # Remove duplicate sources, preserving order
    retrieved_sources = list(
        dict.fromkeys(ranked_sources)
    )

    expected = set(expected_sources)
    retrieved = set(retrieved_sources)

    relevant = retrieved.intersection(expected)

    # Precision
    precision = (
        len(relevant) / len(retrieved)
        if retrieved else 0.0
    )

    # Recall
    recall = (
        len(relevant) / len(expected)
        if expected else 0.0
    )

    # Hit Rate
    hit_rate = float(bool(relevant))

    # Reciprocal Rank (original chunk ranking)
    reciprocal_rank = 0.0

    for rank, source in enumerate(
        ranked_sources, start=1
    ):
        if source in expected:
            reciprocal_rank = 1.0 / rank
            break

    return {
        "precision": precision,
        "recall": recall,
        "hit_rate": hit_rate,
        "reciprocal_rank": reciprocal_rank,
        "retrieved_sources": retrieved_sources
    }


def evaluate_retrieval():

    with open(DATASET_PATH, encoding="utf-8") as f:
        dataset = json.load(f)

    all_results = []

    for item in dataset:

        if not item["answerable"]:
            continue

        question = item["question"]
        expected = item["expected_sources"]

        print(f"\nQuestion {item['question_id']}")
        print(question)

        # Retrieve once, evaluate multiple K values
        results = retrieve(
            question,
            top_k=max(K_VALUES)
        )

        question_result = {
            "question_id": item["question_id"],
            "question": question,
            "expected_sources": expected,
            "metrics": {}
        }

        for k in K_VALUES:

            metrics = calculate_metrics(
                results, expected, k
            )

            question_result["metrics"][str(k)] = metrics

            print(
                f"K={k} | "
                f"Precision={metrics['precision']:.2f} | "
                f"Recall={metrics['recall']:.2f} | "
                f"Hit={metrics['hit_rate']:.0f} | "
                f"RR={metrics['reciprocal_rank']:.2f}"
            )

        all_results.append(question_result)

    if not all_results:
        print("No answerable questions to evaluate.")
        return

    summary = {}

    print("\n" + "=" * 60)
    print("RETRIEVAL EVALUATION SUMMARY")
    print("=" * 60)

    for k in K_VALUES:

        scores = [
            result["metrics"][str(k)]
            for result in all_results
        ]

        summary[str(k)] = {
            "precision": mean(
                s["precision"] for s in scores
            ),
            "recall": mean(
                s["recall"] for s in scores
            ),
            "hit_rate": mean(
                s["hit_rate"] for s in scores
            ),
            "mrr": mean(
                s["reciprocal_rank"] for s in scores
            )
        }

        print(f"\nTop K = {k}")

        for metric, value in summary[str(k)].items():
            print(f"{metric}: {value:.4f}")

    with open(RESULTS_PATH, "w", encoding="utf-8") as f:
        json.dump(
            {
                "summary": summary,
                "question_results": all_results
            },
            f,
            indent=4
        )

    print(f"\nResults saved to: {RESULTS_PATH}")


if __name__ == "__main__":
    evaluate_retrieval()
