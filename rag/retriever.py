import logging
from typing import Any

from rag.embeddings import create_embedding
from rag.vector_store import COLLECTION_NAME, _get_collection

logger = logging.getLogger(__name__)

RetrievedChunk = dict[str, Any]


def retrieve_chunks(question: str, top_k: int = 3) -> list[RetrievedChunk]:
    """Return the nearest stored chunks, with their source metadata."""
    if not isinstance(question, str) or not question.strip():
        raise ValueError("Question cannot be empty.")
    if not isinstance(top_k, int) or isinstance(top_k, bool) or top_k <= 0:
        raise ValueError("top_k must be a positive integer.")

    collection = _get_collection()
    record_count = collection.count()
    if record_count == 0:
        raise ValueError(
            "No documentation has been indexed yet. "
            "Upload and store a document first."
        )

    question_embedding = create_embedding(question)
    result_count = min(top_k, record_count)

    try:
        results = collection.query(
            query_embeddings=[question_embedding],
            n_results=result_count,
            include=["documents", "metadatas", "distances"],
        )
    except Exception as error:
        raise RuntimeError(
            f"Could not query the {COLLECTION_NAME} ChromaDB collection."
        ) from error

    documents = results.get("documents", [[]])[0]
    metadatas = results.get("metadatas", [[]])[0]
    distances = results.get("distances", [[]])[0]

    # Optional metadata stays None when a record does not carry it; it is
    # never guessed. Records indexed before Stage 7 have no page or chunk_id.
    retrieved = [
        {
            "text": text,
            "source": metadata.get("source", "Unknown source"),
            "document_id": metadata.get("document_id"),
            "chunk_id": metadata.get("chunk_id"),
            "chunk_index": metadata.get("chunk_index"),
            "page": metadata.get("page"),
            "section": metadata.get("section"),
            "distance": distance,
        }
        for text, metadata, distance in zip(documents, metadatas, distances)
    ]
    logger.info("Retrieved %d chunks for the question.", len(retrieved))

    return retrieved
