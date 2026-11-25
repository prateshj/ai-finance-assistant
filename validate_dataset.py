
import json
from pathlib import Path
from ingestion import load_documents

BASE_DIR = Path(__file__).resolve().parent
DATASET_PATH = BASE_DIR / "evaluation" / "golden_dataset.json"


def normalize(text):
    return " ".join(text.lower().split())


def validate_dataset():

    with open(DATASET_PATH, encoding="utf-8") as f:
        dataset = json.load(f)

    documents = {
        doc["source"]: doc["text"]
        for doc in load_documents()
    }

    seen_questions = set()
    valid_count = 0

    required_fields = [
        "question_id",
        "question",
        "reference_answer",
        "supporting_evidence",
        "expected_sources",
        "answerable"
    ]

    for item in dataset:

        errors = []
        question_id = item.get("question_id", "Unknown")

        # Check required fields
        for field in required_fields:
            if field not in item or item[field] in (None, "", []):
                errors.append(f"Missing field: {field}")

        # Check duplicate questions
        question = normalize(item.get("question", ""))

        if question in seen_questions:
            errors.append("Duplicate question")

        seen_questions.add(question)

        # Check source documents
        sources = item.get("expected_sources", [])

        for source in sources:
            if source not in documents:
                errors.append(f"Invalid source: {source}")

        # Check supporting evidence
        evidence = normalize(
            item.get("supporting_evidence", "")
        )

        evidence_found = bool(evidence) and any(
            evidence in normalize(documents[source])
            for source in sources
            if source in documents
        )

        if not evidence_found:
            errors.append("Supporting evidence not found")

        if errors:
            print(f"Question {question_id}: FAILED")

            for error in errors:
                print(f"  - {error}")

        else:
            valid_count += 1
            print(f"Question {question_id}: PASSED")

    print("\nVALIDATION SUMMARY")
    print(f"Total questions: {len(dataset)}")
    print(f"Passed: {valid_count}")
    print(f"Failed: {len(dataset) - valid_count}")


if __name__ == "__main__":
    validate_dataset()
