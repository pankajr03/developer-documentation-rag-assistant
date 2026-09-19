from pathlib import Path

import chromadb

COLLECTION_NAME = "developer_docs"
CHROMA_PATH = Path(__file__).resolve().parent.parent / "data" / "chroma"


def _get_collection():
    client = chromadb.PersistentClient(path=str(CHROMA_PATH))
    return client.get_or_create_collection(name=COLLECTION_NAME)


def store_chunks(chunks, embeddings, source):
    """Upsert document chunks, embeddings, and metadata into ChromaDB."""
    if not chunks:
        raise ValueError("Cannot store an empty chunk list.")
    if not embeddings:
        raise ValueError("Cannot store an empty embedding list.")
    if len(chunks) != len(embeddings):
        raise ValueError(
            "The number of chunks must match the number of embeddings.")
    if not isinstance(source, str) or not source.strip():
        raise ValueError("source must be a non-empty string.")

    ids = [f"{source}_chunk_{index}" for index in range(len(chunks))]
    metadatas = [
        {"source": source, "chunk_index": index}
        for index in range(len(chunks))
    ]

    collection = _get_collection()
    collection.upsert(
        ids=ids,
        embeddings=embeddings,
        documents=chunks,
        metadatas=metadatas,
    )

    return len(chunks)


def get_collection_count():
    """Return the number of records currently stored in the collection."""
    return _get_collection().count()
