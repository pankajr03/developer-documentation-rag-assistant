from rag.embeddings import create_embedding
from rag.vector_store import COLLECTION_NAME, _get_collection


def retrieve_chunks(question, top_k=3):
    """Return the nearest stored chunks for a natural-language question."""
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

    return [
        {
            "text": text,
            "source": metadata.get("source", "Unknown source"),
            "chunk_index": metadata.get("chunk_index", "Unknown"),
            "distance": distance,
        }
        for text, metadata, distance in zip(documents, metadatas, distances)
    ]
