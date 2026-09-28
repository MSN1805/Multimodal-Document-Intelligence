import os
import json
import pickle
import hashlib

import faiss
import numpy as np


VECTOR_DB_DIR = "vector_db"


def create_faiss_index(embeddings):
    embeddings = np.asarray(embeddings, dtype="float32")
    if embeddings.ndim != 2 or embeddings.shape[0] == 0:
        raise ValueError("Embeddings must be a non-empty 2D array.")

    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(embeddings)
    return index


def search_faiss(index, query_embedding, top_k=3):
    query_embedding = np.asarray(query_embedding, dtype="float32")

    if index is None or index.ntotal == 0:
        return np.array([[]]), np.array([[]], dtype=int)

    top_k = min(top_k, index.ntotal)
    return index.search(query_embedding, top_k)


def create_document_id(file_name, pdf_bytes):
    file_hash = hashlib.sha256(pdf_bytes).hexdigest()[:16]

    safe_name = os.path.splitext(file_name)[0]
    safe_name = "".join(
        character if character.isalnum() else "_"
        for character in safe_name
    )

    return f"{safe_name}_{file_hash}"


def get_document_directory(document_id):
    return os.path.join(VECTOR_DB_DIR, document_id)


def save_vector_database(index, chunks, file_name=None, pdf_bytes=None):
    if not file_name or pdf_bytes is None:
        raise ValueError("file_name and pdf_bytes are required.")

    os.makedirs(VECTOR_DB_DIR, exist_ok=True)

    document_id = create_document_id(file_name, pdf_bytes)
    directory = get_document_directory(document_id)
    os.makedirs(directory, exist_ok=True)

    faiss.write_index(index, os.path.join(directory, "index.faiss"))

    with open(os.path.join(directory, "chunks.pkl"), "wb") as file:
        pickle.dump(chunks, file)

    metadata = {
        "document_id": document_id,
        "file_name": file_name,
        "number_of_chunks": len(chunks),
        "number_of_vectors": index.ntotal,
        "embedding_dimension": index.d,
        "file_hash": hashlib.sha256(pdf_bytes).hexdigest()
    }

    with open(
        os.path.join(directory, "metadata.json"),
        "w",
        encoding="utf-8"
    ) as file:
        json.dump(metadata, file, indent=4)

    return document_id


def load_vector_database(document_id):
    directory = get_document_directory(document_id)

    index_path = os.path.join(directory, "index.faiss")
    chunks_path = os.path.join(directory, "chunks.pkl")
    metadata_path = os.path.join(directory, "metadata.json")

    if not os.path.exists(index_path) or not os.path.exists(chunks_path):
        return None, None, None

    index = faiss.read_index(index_path)

    with open(chunks_path, "rb") as file:
        chunks = pickle.load(file)

    metadata = None

    if os.path.exists(metadata_path):
        with open(metadata_path, "r", encoding="utf-8") as file:
            metadata = json.load(file)

    return index, chunks, metadata


def load_multiple_documents(document_ids):
    loaded = []

    for document_id in document_ids:
        index, chunks, metadata = load_vector_database(document_id)

        if index is not None and chunks is not None:
            loaded.append({
                "document_id": document_id,
                "index": index,
                "chunks": chunks,
                "metadata": metadata or {}
            })

    return loaded


def list_documents():
    if not os.path.exists(VECTOR_DB_DIR):
        return []

    documents = []

    for item in os.listdir(VECTOR_DB_DIR):
        directory = os.path.join(VECTOR_DB_DIR, item)

        if not os.path.isdir(directory):
            continue

        metadata_path = os.path.join(directory, "metadata.json")

        if not os.path.exists(metadata_path):
            continue

        try:
            with open(metadata_path, "r", encoding="utf-8") as file:
                metadata = json.load(file)

            documents.append(metadata)

        except Exception:
            continue

    documents.sort(key=lambda x: x.get("file_name", "").lower())
    return documents


def document_exists(document_id):
    directory = get_document_directory(document_id)

    return (
        os.path.exists(os.path.join(directory, "index.faiss"))
        and os.path.exists(os.path.join(directory, "chunks.pkl"))
    )


def delete_document(document_id):
    directory = get_document_directory(document_id)

    if not os.path.isdir(directory):
        return False

    import shutil
    shutil.rmtree(directory)
    return True


def vector_database_exists():
    return len(list_documents()) > 0
