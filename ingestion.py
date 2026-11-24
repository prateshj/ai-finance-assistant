
from pathlib import Path

DATA_DIR = Path(__file__).parent / "data"

CHUNK_SIZE = 150
CHUNK_OVERLAP = 30


def load_documents():
    documents = []

    for file in sorted(DATA_DIR.glob("*.txt")):
        text = file.read_text(encoding="utf-8-sig").strip()

        if text:
            documents.append({
                "source": file.name,
                "text": text
            })

    return documents


def chunk_documents(documents):
    chunks = []

    for doc in documents:
        words = doc["text"].split()
        step = CHUNK_SIZE - CHUNK_OVERLAP

        for start in range(0, len(words), step):
            end = start + CHUNK_SIZE

            chunks.append({
                "chunk_id": len(chunks),
                "source": doc["source"],
                "text": " ".join(words[start:end])
            })

            if end >= len(words):
                break

    return chunks


if __name__ == "__main__":
    documents = load_documents()
    chunks = chunk_documents(documents)

    print(f"Documents loaded: {len(documents)}")
    print(f"Total chunks: {len(chunks)}")

    for chunk in chunks:
        print("\n" + "-" * 50)
        print(f"Chunk ID: {chunk['chunk_id']}")
        print(f"Source: {chunk['source']}")
        print(chunk["text"])
