
import os
import json
from pathlib import Path

import faiss
import numpy as np
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

client = OpenAI(timeout=30.0, max_retries=0)

EMBEDDING_MODEL = os.getenv(
    "OPENAI_EMBEDDING_MODEL",
    "text-embedding-3-small"
)

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
INDEX_DIR = BASE_DIR / "vector_store"


def load_chunks(chunk_size=150, overlap=30):
    chunks = []

    for file in sorted(DATA_DIR.glob("*.txt")):
        text = file.read_text(encoding="utf-8-sig")
        words = text.split()

        for start in range(0, len(words), chunk_size - overlap):
            chunk = " ".join(words[start:start + chunk_size])

            chunks.append({
                "id": len(chunks),
                "text": chunk,
                "source": file.name
            })

            if start + chunk_size >= len(words):
                break

    return chunks


def create_embeddings(chunks):
    texts = [chunk["text"] for chunk in chunks]

    response = client.embeddings.create(
        model=EMBEDDING_MODEL,
        input=texts
    )

    vectors = [
        item.embedding
        for item in sorted(response.data, key=lambda x: x.index)
    ]

    return np.array(vectors, dtype="float32")


def build_index():
    chunks = load_chunks()

    if not chunks:
        raise ValueError("No text chunks found in data folder.")

    print(f"Loaded {len(chunks)} chunks.")
    print("Generating embeddings...", flush=True)

    embeddings = create_embeddings(chunks)

    # Normalize vectors for cosine similarity
    faiss.normalize_L2(embeddings)

    dimension = embeddings.shape[1]

    index = faiss.IndexFlatIP(dimension)
    index.add(embeddings)

    INDEX_DIR.mkdir(exist_ok=True)

    faiss.write_index(
        index,
        str(INDEX_DIR / "finance.index")
    )

    with open(
        INDEX_DIR / "chunks.json",
        "w",
        encoding="utf-8"
    ) as file:
        json.dump(chunks, file, indent=2, ensure_ascii=False)

    print(f"Embedding shape: {embeddings.shape}")
    print(f"Indexed vectors: {index.ntotal}")
    print("FAISS index saved successfully!")


if __name__ == "__main__":
    build_index()
