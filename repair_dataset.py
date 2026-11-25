
import json
import re
import shutil
from pathlib import Path
from openai import OpenAI
from dotenv import load_dotenv
from ingestion import load_documents

load_dotenv()
client = OpenAI()

BASE_DIR = Path(__file__).resolve().parent

DATASET_PATH = (
    BASE_DIR / "evaluation" / "golden_dataset.json"
)


def normalize(text):
    return " ".join(text.lower().split())


def repair_question(item, document):

    # Assign IDs to original source passages
    passages = [
        p.strip()
        for p in re.split(r"\n\s*\n", document)
        if p.strip()
    ]

    numbered_passages = "\n\n".join(
        f"Passage {i}: {text}"
        for i, text in enumerate(passages)
    )

    prompt = f"""
    Evaluate the question and reference answer
    against the source passages.

    Question:
    {item["question"]}

    Reference answer:
    {item["reference_answer"]}

    Source passages:
    {numbered_passages}

    Select one passage that fully supports
    the answer.

    Correct the reference answer if necessary.
    Do not introduce unsupported information.

    If no passage adequately supports an answer,
    return passage_id as null.

    Return JSON with:
    - passage_id
    - reference_answer
    """

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "user", "content": prompt}
        ],
        response_format={"type": "json_object"},
        temperature=0
    )

    result = json.loads(
        response.choices[0].message.content
    )

    passage_id = result.get("passage_id")
    answer = result.get("reference_answer")

    if (
        type(passage_id) is not int
        or not 0 <= passage_id < len(passages)
        or not isinstance(answer, str)
        or not answer.strip()
    ):
        return False

    # Extract original text directly.
    # Do not use LLM-generated quotations.
    item["supporting_evidence"] = passages[passage_id]

    item["reference_answer"] = answer

    return True


def repair_dataset():

    with open(DATASET_PATH, encoding="utf-8") as f:
        dataset = json.load(f)

    documents = {
        doc["source"]: doc["text"]
        for doc in load_documents()
    }

    repaired = 0
    unchanged = 0
    failed = 0

    # Back up the original dataset
    shutil.copy2(
        DATASET_PATH,
        DATASET_PATH.with_suffix(".json.bak")
    )

    for item in dataset:

        sources = item["expected_sources"]

        evidence = normalize(
            item["supporting_evidence"]
        )

        evidence_found = bool(evidence) and any(
            evidence in normalize(documents[source])
            for source in sources
            if source in documents
        )

        if evidence_found:
            unchanged += 1
            continue

        print(
            f"Repairing question {item['question_id']}"
        )

        source = sources[0] if sources else None

        if source not in documents:
            failed += 1
            continue

        try:
            success = repair_question(
                item,
                documents[source]
            )

            if success:
                repaired += 1
                print("  Evidence replaced")
            else:
                failed += 1
                print("  Manual review required")

        except Exception as e:
            failed += 1
            print(f"  Error: {e}")

    with open(DATASET_PATH, "w", encoding="utf-8") as f:
        json.dump(
            dataset, f, indent=4,
            ensure_ascii=False
        )

    print("\nREPAIR SUMMARY")
    print(f"Total questions: {len(dataset)}")
    print(f"Unchanged: {unchanged}")
    print(f"Repaired: {repaired}")
    print(f"Failed: {failed}")


if __name__ == "__main__":
    repair_dataset()
