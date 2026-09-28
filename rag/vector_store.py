import logging
from pathlib import Path
from typing import Any, Optional, Sequence

import chromadb

from rag.loader import document_id_from_name

logger = logging.getLogger(__name__)

COLLECTION_NAME = "developer_docs"
CHROMA_PATH = Path(__file__).resolve().parent.parent / "data" / "chroma"


def _get_collection():
    client = chromadb.PersistentClient(path=str(CHROMA_PATH))
    return client.get_or_create_collection(name=COLLECTION_NAME)


def _normalize_chunk(chunk: Any, chunk_index: int) -> dict[str, Any]:
    """Accept either a plain string chunk or a chunk dict with metadata."""
    if isinstance(chunk, str):
        return {"text": chunk, "page": None, "section": None,
                "chunk_index": chunk_index}

    if isinstance(chunk, dict) and isinstance(chunk.get("text"), str):
        return {
            "text": chunk["text"],
            "page": chunk.get("page"),
            "section": chunk.get("section"),
            "chunk_index": chunk.get("chunk_index", chunk_index),
        }

    raise ValueError("Every chunk must be a string or a dict containing text.")


def build_chunk_id(document_id: str, page: Optional[int], chunk_index: int) -> str:
    """Build a unique chunk id such as speed_schedule_p3_c2."""
    page_part = f"_p{page}" if page is not None else ""
    return f"{document_id}{page_part}_c{chunk_index}"


def store_chunks(
    chunks: Sequence[Any],
    embeddings: Sequence[Sequence[float]],
    source: str,
    document_id: Optional[str] = None,
    content_hash: Optional[str] = None,
) -> int:
    """Upsert chunks, embeddings, and source metadata into ChromaDB.

    content_hash (optional) is stored on every chunk so the same file can be
    recognised later, even when it is uploaded under a different name.
    """
    if not chunks:
        raise ValueError("Cannot store an empty chunk list.")
    if not embeddings:
        raise ValueError("Cannot store an empty embedding list.")
    if len(chunks) != len(embeddings):
        raise ValueError(
            "The number of chunks must match the number of embeddings.")
    if not isinstance(source, str) or not source.strip():
        raise ValueError("source must be a non-empty string.")

    document_id = document_id or document_id_from_name(source) or "document"
    normalized = [
        _normalize_chunk(chunk, index) for index, chunk in enumerate(chunks)
    ]

    ids: list[str] = []
    documents: list[str] = []
    metadatas: list[dict[str, Any]] = []

    for chunk in normalized:
        chunk_id = build_chunk_id(
            document_id, chunk["page"], chunk["chunk_index"])
        metadata: dict[str, Any] = {
            "source": source,
            "document_id": document_id,
            "chunk_id": chunk_id,
            "chunk_index": chunk["chunk_index"],
        }
        # Chroma rejects None values, so optional fields are omitted entirely
        # when the document does not provide them.
        if chunk["page"] is not None:
            metadata["page"] = chunk["page"]
        if chunk["section"]:
            metadata["section"] = chunk["section"]
        if content_hash:
            metadata["content_hash"] = content_hash

        ids.append(chunk_id)
        documents.append(chunk["text"])
        metadatas.append(metadata)

    collection = _get_collection()
    collection.upsert(
        ids=ids,
        embeddings=embeddings,
        documents=documents,
        metadatas=metadatas,
    )
    logger.info(
        "Stored %d chunks for document_id=%s in collection %s.",
        len(ids),
        document_id,
        COLLECTION_NAME,
    )

    return len(ids)


def get_collection_count() -> int:
    """Return the number of records currently stored in the collection."""
    return _get_collection().count()


def reset_collection() -> None:
    """Delete the whole collection so documents can be re-indexed cleanly."""
    client = chromadb.PersistentClient(path=str(CHROMA_PATH))
    try:
        client.delete_collection(name=COLLECTION_NAME)
    except Exception as error:  # collection may not exist yet
        logger.warning("Could not delete collection: %s", error)
    client.get_or_create_collection(name=COLLECTION_NAME)
    logger.info("Collection %s was reset.", COLLECTION_NAME)
