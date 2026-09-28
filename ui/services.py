"""Service boundary between the Streamlit pages and the application modules.

Pages call these functions instead of the rag/ and services/ modules directly,
so every error reaching the interface is a short, user-safe message. Nothing
here imports Streamlit, which keeps it testable with plain fakes.

Logging records only exception type names, never messages. Messages from
external libraries can echo request data or partial keys.
"""

import hashlib
import logging
import os
from dataclasses import dataclass, field
from typing import Any, Callable, Iterable, Optional, Sequence

from ui.config import EMBEDDING_BATCH_SIZE, MAX_STATUS_SCAN, QUESTION_PREVIEW_CHARS
from ui.formatting import display_filename, format_sources, preview
from ui.validation import validate_upload

logger = logging.getLogger(__name__)

MSG_MISSING_KEY = (
    "The OpenAI API key is missing. Add OPENAI_API_KEY to the .env file in the "
    "project folder and restart the app.")
MSG_INVALID_KEY = "OpenAI rejected the API key. Check OPENAI_API_KEY in the .env file."
MSG_RATE_LIMIT = (
    "OpenAI is rate-limiting requests, or the account has no credits left. "
    "Wait a moment and try again.")
MSG_TIMEOUT = "The request to OpenAI timed out. Please try again."
MSG_NETWORK = "Could not reach OpenAI. Check your internet connection and try again."
MSG_EMPTY_KB = (
    "No documentation is indexed yet. Add documents on the Manage Documents page.")
MSG_VECTOR_DB = "The document database (ChromaDB) could not be opened."
MSG_FEEDBACK_DB = "Could not reach the feedback database. Your answer is still shown."


class AssistantError(Exception):
    """An operation failed; str(error) is safe to show to users."""


# --- Error handling ------------------------------------------------------------

def _exception_chain(error: BaseException) -> list[BaseException]:
    chain: list[BaseException] = []
    current: Optional[BaseException] = error
    while current is not None and current not in chain:
        chain.append(current)
        current = current.__cause__ or current.__context__
    return chain


def friendly_error(error: BaseException, fallback: str) -> str:
    """Map an exception to a short message that reveals no internals.

    The rag modules wrap OpenAI errors in RuntimeError, so the whole cause
    chain is inspected. Anything unrecognised becomes ``fallback``.
    """
    import openai

    from services.feedback_service import FeedbackStorageError

    for exc in _exception_chain(error):
        if isinstance(exc, AssistantError):
            return str(exc)
        if isinstance(exc, openai.AuthenticationError):
            return MSG_INVALID_KEY
        if isinstance(exc, openai.RateLimitError):
            return MSG_RATE_LIMIT
        if isinstance(exc, openai.APITimeoutError):  # before APIConnectionError
            return MSG_TIMEOUT
        if isinstance(exc, openai.APIConnectionError):
            return MSG_NETWORK
        if isinstance(exc, FeedbackStorageError):
            return MSG_FEEDBACK_DB
        if type(exc).__module__.startswith("chromadb"):
            return MSG_VECTOR_DB
        text = str(exc)
        if "OPENAI_API_KEY is not set" in text:
            return MSG_MISSING_KEY
        if text.startswith("No documentation has been indexed"):
            return MSG_EMPTY_KB
    return fallback


def _log_failure(action: str, error: BaseException) -> None:
    names = " <- ".join(type(exc).__name__ for exc in _exception_chain(error))
    logger.warning("%s failed: %s", action, names)


def api_key_configured() -> bool:
    """True when OPENAI_API_KEY is set. The value itself is never returned."""
    import rag.embeddings  # noqa: F401  (loads .env on import)

    return bool(os.getenv("OPENAI_API_KEY", "").strip())


def model_info() -> dict[str, str]:
    from rag import embeddings, generator, vector_store

    return {
        "embedding_model": embeddings.MODEL_NAME,
        "generation_model": generator.MODEL_NAME,
        "collection": vector_store.COLLECTION_NAME,
    }


# --- Knowledge base status -----------------------------------------------------

@dataclass(frozen=True)
class DocumentSummary:
    name: str
    document_id: Optional[str]
    chunks: int


@dataclass
class KnowledgeBaseStatus:
    """state is "ready", "empty" or "error". documents is None when the
    collection was too large to scan, so no document count is claimed."""

    state: str
    chunk_count: int = 0
    documents: Optional[list[DocumentSummary]] = None
    document_ids: set[str] = field(default_factory=set)
    content_hashes: set[str] = field(default_factory=set)
    message: str = ""

    @property
    def document_count(self) -> Optional[int]:
        return None if self.documents is None else len(self.documents)


def _open_collection(client: Any) -> Any:
    """The existing collection, or None if it was never created.

    get_collection is used instead of get_or_create_collection so that
    looking at the status never creates anything.
    """
    from rag.vector_store import COLLECTION_NAME

    try:
        return client.get_collection(name=COLLECTION_NAME)
    except Exception as error:
        if "does not exist" in str(error) or type(error).__name__ == "NotFoundError":
            return None
        raise


def knowledge_base_status(client: Any) -> KnowledgeBaseStatus:
    """Read-only status of the ChromaDB collection.

    client is None when the ChromaDB folder does not exist yet.
    """
    if client is None:
        return KnowledgeBaseStatus("empty", message=MSG_EMPTY_KB)
    try:
        collection = _open_collection(client)
        if collection is None:
            return KnowledgeBaseStatus("empty", message=MSG_EMPTY_KB)
        count = collection.count()
        if count == 0:
            return KnowledgeBaseStatus("empty", message=MSG_EMPTY_KB)
        status = KnowledgeBaseStatus("ready", chunk_count=count)
        if count <= MAX_STATUS_SCAN:
            metadatas = collection.get(include=["metadatas"])["metadatas"]
            _summarize_documents(status, metadatas)
        return status
    except Exception as error:
        _log_failure("Reading the knowledge base status", error)
        return KnowledgeBaseStatus("error", message=MSG_VECTOR_DB)


def _summarize_documents(
    status: KnowledgeBaseStatus, metadatas: Iterable[Optional[dict]],
) -> None:
    counts: dict[tuple[Optional[str], str], int] = {}
    for metadata in metadatas:
        metadata = metadata or {}
        name = display_filename(metadata.get("source")) or "Unknown document"
        document_id = metadata.get("document_id")
        counts[(document_id, name)] = counts.get((document_id, name), 0) + 1
        if document_id:
            status.document_ids.add(document_id)
        if metadata.get("content_hash"):
            status.content_hashes.add(metadata["content_hash"])
    status.documents = [
        DocumentSummary(name, document_id, chunks)
        for (document_id, name), chunks in sorted(
            counts.items(), key=lambda item: item[0][1].lower())
    ]


# --- Asking questions ----------------------------------------------------------

def _feedback_payload(result: dict[str, Any]) -> dict[str, Any]:
    """What save_feedback needs later, without embeddings or full chunk text."""
    return {
        "sources": result.get("sources") or [],
        "retrieved_chunks": [
            {key: value for key, value in chunk.items() if key != "text"}
            for chunk in result.get("retrieved_chunks") or []
        ],
        "model_name": result.get("model_name"),
        "retrieval_top_k": result.get("retrieval_top_k"),
    }


def build_answer(result: dict[str, Any]) -> dict[str, Any]:
    """Turn a generate_answer result into the data stored in the chat history."""
    from evaluation.metrics import is_abstention
    from rag.generator import FALLBACK_MESSAGE

    answer = result.get("answer") or ""
    is_fallback = is_abstention(answer, (FALLBACK_MESSAGE,))
    warnings = []
    if not is_fallback and not result.get("citations"):
        warnings.append(
            "This answer has no citations. Check it against the sources below.")
    removed = result.get("unverified_citation_numbers") or []
    if removed:
        warnings.append(
            f"Removed {len(removed)} citation(s) to sources that were not retrieved.")
    return {
        "content": answer,
        "question": result.get("question"),
        "response_id": result.get("response_id"),
        "sources": format_sources(result),
        "is_fallback": is_fallback,
        "warnings": warnings,
        "feedback_payload": _feedback_payload(result),
    }


def ask_question(
    question: str,
    top_k: int,
    retrieve: Optional[Callable[..., list[dict]]] = None,
    generate: Optional[Callable[..., dict]] = None,
) -> dict[str, Any]:
    """Retrieve chunks and generate a grounded answer with the real pipeline.

    The question must already be validated. Raises AssistantError.
    """
    if retrieve is None:
        from rag.retriever import retrieve_chunks as retrieve
    if generate is None:
        from rag.generator import generate_answer as generate

    try:
        chunks = retrieve(question, top_k=top_k)
    except Exception as error:
        _log_failure("Retrieval", error)
        raise AssistantError(friendly_error(
            error, "Could not search the documentation. Please try again.")) from error
    try:
        result = generate(question, chunks, top_k=top_k)
    except Exception as error:
        _log_failure("Answer generation", error)
        raise AssistantError(friendly_error(
            error, "Could not generate an answer. Please try again.")) from error
    return build_answer(result)


# --- Feedback ------------------------------------------------------------------

def save_answer_feedback(
    message: dict[str, Any],
    rating: int,
    comment: Optional[str],
    session_id: Optional[str],
    save: Optional[Callable[..., Any]] = None,
) -> None:
    """Save a rating for a stored answer with the Stage 8 service.

    Nothing is regenerated: everything comes from the stored message.
    Raises AssistantError.
    """
    if save is None:
        from services.feedback_service import save_feedback as save

    payload = message.get("feedback_payload") or {}
    try:
        save(
            response_id=message["response_id"],
            question=message["question"],
            answer=message["content"],
            rating=rating,
            comment=comment or None,
            sources=payload.get("sources"),
            retrieved_chunks=payload.get("retrieved_chunks"),
            model_name=payload.get("model_name"),
            retrieval_top_k=payload.get("retrieval_top_k"),
            session_id=session_id,
        )
    except ValueError as error:
        # The feedback service's validation messages are written for users.
        raise AssistantError(str(error)) from error
    except Exception as error:
        _log_failure("Saving feedback", error)
        raise AssistantError(friendly_error(error, MSG_FEEDBACK_DB)) from error


def feedback_overview(
    limit: int,
    summary_fn: Optional[Callable[[], dict]] = None,
    list_fn: Optional[Callable[..., list[dict]]] = None,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Aggregate counts plus display rows for the most recent feedback."""
    from services import feedback_service

    summary_fn = summary_fn or feedback_service.get_feedback_summary
    list_fn = list_fn or feedback_service.list_recent_feedback
    try:
        summary = summary_fn()
        records = list_fn(limit=limit)
    except Exception as error:
        _log_failure("Reading feedback", error)
        raise AssistantError(friendly_error(
            error, "Could not read the feedback database.")) from error

    rows = [{
        "Date (UTC)": (record.get("updated_at") or record.get("created_at") or "")[:16]
                      .replace("T", " "),
        "Rating": "👍 Helpful" if record.get("rating") == 1 else "👎 Not helpful",
        "Question": preview(record.get("question"), QUESTION_PREVIEW_CHARS),
        "Comment": record.get("comment") or "",
        "Sources": ", ".join(record.get("source_filenames") or []),
        "Response ID": record.get("response_id") or "",
    } for record in records]
    return summary, rows


# --- Indexing documents --------------------------------------------------------

@dataclass
class FileOutcome:
    name: str
    status: str  # "indexed", "skipped" or "failed"
    chunks: int = 0
    message: str = ""


@dataclass
class IndexReport:
    outcomes: list[FileOutcome] = field(default_factory=list)

    def count(self, status: str) -> int:
        return sum(outcome.status == status for outcome in self.outcomes)

    @property
    def chunks_created(self) -> int:
        return sum(outcome.chunks for outcome in self.outcomes)


class _InMemoryUpload:
    """The minimal interface rag.loader.load_document reads (name, getvalue).

    Uploads are processed from memory, so no temporary file, and no path built
    from a browser-supplied filename, is ever created.
    """

    def __init__(self, name: str, data: bytes):
        self.name = name
        self._data = data

    def getvalue(self) -> bytes:
        return self._data


ProgressCallback = Callable[[str, int, int, str], None]


def _embed_in_batches(texts: list[str], embed: Callable[[list[str]], list]) -> list:
    embeddings: list = []
    for start in range(0, len(texts), EMBEDDING_BATCH_SIZE):
        embeddings.extend(embed(texts[start:start + EMBEDDING_BATCH_SIZE]))
    return embeddings


def index_documents(
    files: Sequence[tuple[str, bytes]],
    known_document_ids: Iterable[str] = (),
    known_content_hashes: Iterable[str] = (),
    progress: Optional[ProgressCallback] = None,
    embed: Optional[Callable[[list[str]], list]] = None,
    store: Optional[Callable[..., int]] = None,
) -> IndexReport:
    """Validate, extract, chunk, embed and store each file.

    Only called from an explicit button click, so Streamlit reruns never
    trigger paid embedding calls. Duplicates are skipped, never replaced:
    a file whose document id (derived from its name) or content hash is
    already indexed, or appears earlier in the same batch.
    """
    from rag.chunker import chunk_sections
    from rag.loader import document_id_from_name, load_document

    if embed is None:
        from rag.embeddings import create_embeddings as embed
    if store is None:
        from rag.vector_store import store_chunks as store

    document_ids = set(known_document_ids)
    content_hashes = set(known_content_hashes)
    report = IndexReport()

    def step(stage: str, index: int, name: str) -> None:
        if progress:
            progress(stage, index, len(files), name)

    for index, (raw_name, data) in enumerate(files):
        check = validate_upload(raw_name, len(data))
        name = check.display_name
        if not check.ok:
            report.outcomes.append(FileOutcome(name, "skipped", message=check.message))
            continue

        digest = hashlib.sha256(data).hexdigest()
        document_id = document_id_from_name(name) or "document"
        if digest in content_hashes:
            report.outcomes.append(FileOutcome(
                name, "skipped", message="The same content is already indexed."))
            continue
        if document_id in document_ids:
            report.outcomes.append(FileOutcome(
                name, "skipped",
                message=f"A document with id '{document_id}' is already indexed. "
                        "To re-index it, reset the collection first."))
            continue

        step("Reading document", index, name)
        try:
            sections = load_document(_InMemoryUpload(name, data))
        except UnicodeDecodeError:
            report.outcomes.append(FileOutcome(
                name, "failed", message="The text file is not UTF-8 encoded."))
            continue
        except Exception as error:
            _log_failure("Extracting text", error)
            report.outcomes.append(FileOutcome(
                name, "failed", message="Could not extract text from this file."))
            continue

        step("Creating chunks", index, name)
        # Whitespace-only chunks cannot be embedded, so they are dropped.
        chunks = [c for c in chunk_sections(sections) if c["text"].strip()]
        if not chunks:
            report.outcomes.append(FileOutcome(
                name, "skipped",
                message="No usable text found. Scanned PDFs need OCR first."))
            continue

        step("Generating embeddings", index, name)
        try:
            embeddings = _embed_in_batches([c["text"] for c in chunks], embed)
        except Exception as error:
            _log_failure("Creating embeddings", error)
            report.outcomes.append(FileOutcome(name, "failed", message=friendly_error(
                error, "Could not create embeddings for this file.")))
            continue

        step("Saving to ChromaDB", index, name)
        try:
            stored = store(chunks, embeddings, name,
                           document_id=document_id, content_hash=digest)
        except Exception as error:
            _log_failure("Saving to ChromaDB", error)
            report.outcomes.append(FileOutcome(name, "failed", message=friendly_error(
                error, "Could not save this file to ChromaDB.")))
            continue

        document_ids.add(document_id)
        content_hashes.add(digest)
        report.outcomes.append(FileOutcome(name, "indexed", chunks=stored))
        logger.info("Indexed a document into %d chunks.", stored)

    step("Completed", len(files), "")
    return report


def reset_knowledge_base() -> None:
    """Delete every indexed chunk (explicit, confirmed user action only)."""
    from rag.vector_store import reset_collection

    try:
        reset_collection()
    except Exception as error:
        _log_failure("Resetting the collection", error)
        raise AssistantError(friendly_error(
            error, "Could not reset the collection.")) from error
