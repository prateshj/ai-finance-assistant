
import json
from pathlib import Path
from openai import OpenAI
from dotenv import load_dotenv
from ingestion import load_documents

load_dotenv()

client = OpenAI()

BASE_DIR = Path(__file__).resolve().parent
OUTPUT_PATH = BASE_DIR / "evaluation" / "golden_dataset.json"


def generate_questions(document):

    prompt = f"""
    Generate 5 diverse evaluation questions
    based strictly on the document below.

    Include factual and conceptual questions.

    For each question, provide:
    - question
    - reference_answer
    - supporting_evidence (exact quote from document)

    Return a JSON object containing a
    "questions" array.

    Document:
    {document["text"]}
    """

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "user", "content": prompt}
        ],
        response_format={"type": "json_object"},
        temperature=0
    )

    return json.loads(
        response.choices[0].message.content
    )["questions"]


def generate_dataset():

    documents = load_documents()
    dataset = []

    for document in documents:

        print(f"Processing: {document['source']}")

        questions = generate_questions(document)

        for item in questions:

            dataset.append({
                "question_id": len(dataset) + 1,
                "question": item["question"],
                "reference_answer": item["reference_answer"],
                "supporting_evidence": item["supporting_evidence"],
                "expected_sources": [document["source"]],
                "answerable": True
            })

    OUTPUT_PATH.parent.mkdir(
        parents=True, exist_ok=True
    )

    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(dataset, f, indent=4, ensure_ascii=False)

    print(f"Generated {len(dataset)} questions")


if __name__ == "__main__":
    generate_dataset()
