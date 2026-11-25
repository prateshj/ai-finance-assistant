
import json
import hashlib
from pathlib import Path
from collections import defaultdict
from datetime import datetime, timezone

from ingestion import load_documents, chunk_documents
import numpy as np
import faiss
import shutil
from build_index import create_embeddings

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
INDEX_DIR = BASE_DIR / "vector_store"

REGISTRY_PATH = INDEX_DIR / "document_registry.json"

CHUNKS_PATH = INDEX_DIR / "chunk.json"
if not CHUNKS_PATH.exists():
    CHUNKS_PATH = INDEX_DIR / "chunks.json"


def calculate_hash(file_path):
    return hashlib.sha256(
        file_path.read_bytes()
    ).hexdigest()


def detect_changes():

    with open(REGISTRY_PATH, encoding="utf-8") as f:
        registry = json.load(f)

    current_files = {
        file.name: calculate_hash(file)
        for file in DATA_DIR.glob("*.txt")
    }

    changes = {
        "new": [],
        "modified": [],
        "deleted": [],
        "unchanged": []
    }

    for filename, content_hash in current_files.items():

        if filename not in registry:
            changes["new"].append(filename)

        elif content_hash != registry[filename]["content_hash"]:
            changes["modified"].append(filename)

        else:
            changes["unchanged"].append(filename)

    for filename in registry:
        if filename not in current_files:
            changes["deleted"].append(filename)

    return changes



def initialize_registry():

    if REGISTRY_PATH.exists():
        print("Registry already exists.")
        return

    with open(CHUNKS_PATH, encoding="utf-8") as f:
        existing_chunks = json.load(f)

    current_chunks = chunk_documents(load_documents())

    # Verify documents match the existing index metadata
    old = [(c["source"], c["text"]) for c in existing_chunks]
    new = [(c["source"], c["text"]) for c in current_chunks]

    if old != new:
        raise ValueError(
            "Current documents do not match saved chunks."
        )

    grouped = defaultdict(list)

    for position, chunk in enumerate(existing_chunks):
        grouped[chunk["source"]].append(position)

    registry = {}

    for filename, positions in grouped.items():

        registry[filename] = {
            "document_id": filename,
            "content_hash": calculate_hash(
                DATA_DIR / filename
            ),
            "chunk_count": len(positions),
            "vector_positions": positions,
            "registered_at": datetime.now(
                timezone.utc
            ).isoformat()
        }

        print(
            f"Registered: {filename} "
            f"({len(positions)} chunks)"
        )

    

def update_documents(changes):

    if not any(changes[key] for key in
               ["new", "modified", "deleted"]):
        print("No document changes detected.")
        return

    index_path = INDEX_DIR / "finance.index"
    index = faiss.read_index(str(index_path))

    with open(CHUNKS_PATH, encoding="utf-8") as f:
        chunks = json.load(f)

    with open(REGISTRY_PATH, encoding="utf-8") as f:
        registry = json.load(f)

    if index.ntotal != len(chunks):
        raise ValueError("Index and metadata mismatch.")

    # Identify outdated documents
    remove_files = set(
        changes["modified"] + changes["deleted"]
    )

    remove_ids = np.array([
        i for i, chunk in enumerate(chunks)
        if chunk["source"] in remove_files
    ], dtype="int64")

    # Prepare new and modified documents
    add_files = set(
        changes["new"] + changes["modified"]
    )

    documents = [
        doc for doc in load_documents()
        if doc["source"] in add_files
    ]

    new_chunks = chunk_documents(documents)

    # Generate embeddings only for changed content
    embeddings = None

    if new_chunks:
        embeddings = create_embeddings(new_chunks)
        faiss.normalize_L2(embeddings)

        if embeddings.shape[1] != index.d:
            raise ValueError("Embedding dimension mismatch.")

    # Remove outdated vectors and metadata
    if len(remove_ids):
        removed = index.remove_ids(remove_ids)

        if removed != len(remove_ids):
            raise ValueError("Vector removal mismatch.")

        chunks = [
            chunk for chunk in chunks
            if chunk["source"] not in remove_files
        ]

    # Add new vectors
    if new_chunks:
        index.add(embeddings)
        chunks.extend(new_chunks)

    # Reassign positional IDs
    for i, chunk in enumerate(chunks):
        chunk["id"] = i
        chunk["chunk_id"] = i

    # Rebuild document registry
    grouped = defaultdict(list)

    for i, chunk in enumerate(chunks):
        grouped[chunk["source"]].append(i)

    updated_registry = {}

    for filename, positions in grouped.items():

        updated_registry[filename] = {
            "document_id": filename,
            "content_hash": calculate_hash(
                DATA_DIR / filename
            ),
            "chunk_count": len(positions),
            "vector_positions": positions,
            "registered_at": (
                registry[filename]["registered_at"]
                if filename in changes["unchanged"]
                else datetime.now(timezone.utc).isoformat()
            )
        }

    if index.ntotal != len(chunks):
        raise ValueError("Final metadata mismatch.")

    # Back up existing files
    for path in [
        index_path, CHUNKS_PATH, REGISTRY_PATH
    ]:
        shutil.copy2(path, str(path) + ".bak")

    # Save updated index and metadata
    faiss.write_index(index, str(index_path))

    with open(CHUNKS_PATH, "w", encoding="utf-8") as f:
        json.dump(chunks, f, indent=2)

    with open(REGISTRY_PATH, "w", encoding="utf-8") as f:
        json.dump(updated_registry, f, indent=2)

    print(f"Removed vectors: {len(remove_ids)}")
    print(f"Added vectors: {len(new_chunks)}")
    print(f"Total vectors: {index.ntotal}")
    print("Index updated successfully!")



    with open(REGISTRY_PATH, "w", encoding="utf-8") as f:
        json.dump(registry, f, indent=2)

    print(f"\nTotal documents: {len(updated_registry)}")
    print(f"Total chunks: {len(chunks)}")
    print("Document registry created successfully!")



if __name__ == "__main__":

    initialize_registry()

    changes = detect_changes()

    update_documents(changes)
