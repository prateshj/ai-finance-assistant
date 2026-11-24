import os
import json
import faiss
import numpy as np

from pathlib import Path
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

client = OpenAI(timeout=30.0, max_retries=0)

BASE_DIR = Path(__file__).resolve().parent

INDEX_PATH = BASE_DIR / "vector_store" / "finance.index"
CHUNKS_PATH = BASE_DIR / "vector_store" / "chunks.json"

EMBEDDING_MODEL = os.getenv(
    "EMBEDDING_MODEL", "text-embedding-3-small"
)

# # Load existing FAISS index
# index = faiss.read_index(
#     str(BASE_DIR / "faiss.index")
# )

# # Load chunk metadata
# with open(BASE_DIR / "chunks.json",
#           "r", encoding="utf-8") as f:
#     chunks = json.load(f)

index = faiss.read_index(str(INDEX_PATH))

with open(CHUNKS_PATH, "r", encoding="utf-8") as f:
    chunks = json.load(f)

def retrieve(query, top_k=3):

    # Generate query embedding
    response = client.embeddings.create(
        model=EMBEDDING_MODEL,
        input=query
    )

    query_vector = np.array(
        [response.data[0].embedding],
        dtype="float32"
    )

    # Search FAISS
    scores, indices = index.search(
        query_vector, min(top_k, index.ntotal)
    )

    results = []

    for score, idx in zip(scores[0], indices[0]):
        if idx >= 0:
            results.append({
                "chunk": chunks[idx],
                "score": float(score)
            })

    return results


if __name__ == "__main__":

    question = input("Ask a question: ")

    results = retrieve(question)

    for rank, result in enumerate(results, 1):
        print(f"\nResult {rank}")
        print(f"Score: {result['score']:.4f}")
        print(result["chunk"])
        print("-" * 50)