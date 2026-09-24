"""Persistent thumbs-up/down feedback for generated answers, stored in SQLite.

This module owns all feedback database code so the UI never touches SQL.
"""

import json
import logging
import sqlite3
import unicodedata
from contextlib import closing
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional, Sequence

logger = logging.getLogger(__name__)

FEEDBACK_DB_PATH = Path(__file__).resolve().parent.parent / "data" / "feedback.db"

RATING_THUMBS_UP = 1
RATING_THUMBS_DOWN = -1
VALID_RATINGS = (RATING_THUMBS_UP, RATING_THUMBS_DOWN)

# Size limits keep the database small and avoid copying whole documents into it.
MAX_COMMENT_LENGTH = 1000
MAX_QUESTION_LENGTH = 2000
MAX_ANSWER_LENGTH = 8000
MAX_EXCERPT_LENGTH = 500
MAX_STORED_SOURCES = 20
MAX_RESPONSE_ID_LENGTH = 100
MAX_LIST_LIMIT = 100
DEFAULT_LIST_LIMIT = 10

SCHEMA = """
CREATE TABLE IF NOT EXISTS feedback (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    response_id TEXT NOT NULL UNIQUE,
    session_id TEXT,
    question TEXT NOT NULL,
    answer TEXT NOT NULL,
    rating INTEGER NOT NULL CHECK (rating IN (1, -1)),
    comment TEXT,
    sources_json TEXT,
    retrieved_chunks_json TEXT,
    model_name TEXT,
    retrieval_top_k INTEGER,
    created_at TEXT NOT NULL,
    updated_at TEXT
)
"""

# Only these fields are copied from a source or retrieved chunk. Anything else
# (for example an embedding vector) is dropped rather than stored.
_SOURCE_FIELDS = ("number", "filename", "page", "chunk_id", "document_id",
                  "section", "distance")
_CHUNK_FIELDS = ("source", "page", "chunk_id", "chunk_index", "document_id",
                 "section", "distance")

_UNSET: Any = object()


class FeedbackStorageError(RuntimeError):
    """The feedback database could not be read or written."""


class FeedbackNotFoundError(LookupError):
    """No feedback exists for the requested response id."""


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _connect(db_path: Optional[Path] = None) -> sqlite3.Connection:
    """Open the database, creating the directory and table when missing."""
    path = Path(db_path) if db_path is not None else FEEDBACK_DB_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(path, timeout=10)
    connection.row_factory = sqlite3.Row
    try:
        connection.execute(SCHEMA)
        connection.commit()
    except sqlite3.Error:
        connection.close()
        raise
    return connection


def _clean_text(value: str) -> str:
    """Normalize Unicode and drop control characters other than newline/tab."""
    normalized = unicodedata.normalize("NFC", value)
    return "".join(
        char for char in normalized
        if char in "\n\t" or unicodedata.category(char) != "Cc"
    )


def _truncate(value: str, limit: int) -> str:
    return value if len(value) <= limit else value[:limit].rstrip() + "…"


def _validate_response_id(response_id: Any) -> str:
    if not isinstance(response_id, str) or not response_id.strip():
        raise ValueError("response_id is required.")
    response_id = response_id.strip()
    if len(response_id) > MAX_RESPONSE_ID_LENGTH:
        raise ValueError("response_id is too long.")
    return response_id


def _validate_rating(rating: Any) -> int:
    # bool is a subclass of int, so True would otherwise pass as thumbs-up.
    if isinstance(rating, bool) or rating not in VALID_RATINGS:
        raise ValueError("rating must be 1 (thumbs up) or -1 (thumbs down).")
    return int(rating)


def _validate_comment(comment: Optional[str]) -> Optional[str]:
    """Trim a comment; return None when it is empty. Reject over-long comments."""
    if comment is None:
        return None
    if not isinstance(comment, str):
        raise ValueError("comment must be text.")
    comment = _clean_text(comment).strip()
    if len(comment) > MAX_COMMENT_LENGTH:
        raise ValueError(
            f"comment must be at most {MAX_COMMENT_LENGTH} characters.")
    return comment or None


def _required_text(value: Any, name: str, limit: int) -> str:
    if not isinstance(value, str) or not _clean_text(value).strip():
        raise ValueError(f"{name} is required.")
    return _truncate(_clean_text(value).strip(), limit)


def _pick(record: dict[str, Any], fields: Sequence[str]) -> dict[str, Any]:
    return {field: record[field] for field in fields
            if record.get(field) is not None}


def _serialize_sources(sources: Optional[Sequence[dict[str, Any]]]) -> str:
    """Keep source metadata and a truncated excerpt as JSON."""
    stored = []
    for source in list(sources or [])[:MAX_STORED_SOURCES]:
        item = _pick(source, _SOURCE_FIELDS)
        content = source.get("content")
        if isinstance(content, str) and content:
            item["excerpt"] = _truncate(
                _clean_text(content).strip(), MAX_EXCERPT_LENGTH)
        stored.append(item)
    return json.dumps(stored, ensure_ascii=False)


def _serialize_chunks(chunks: Optional[Sequence[dict[str, Any]]]) -> str:
    """Keep retrieved-chunk metadata only; the text is already in the sources."""
    stored = [_pick(chunk, _CHUNK_FIELDS)
              for chunk in list(chunks or [])[:MAX_STORED_SOURCES]]
    return json.dumps(stored, ensure_ascii=False)


def _row_to_dict(row: sqlite3.Row) -> dict[str, Any]:
    return dict(row)


def initialize_feedback_database(db_path: Optional[Path] = None) -> None:
    """Create the database file, its directory, and the feedback table."""
    try:
        _connect(db_path).close()
    except sqlite3.Error as error:
        logger.error("Could not initialize the feedback database: %s", error)
        raise FeedbackStorageError(
            "Could not initialize the feedback database.") from error


def save_feedback(
    response_id: str,
    question: str,
    answer: str,
    rating: int,
    comment: Optional[str] = None,
    sources: Optional[Sequence[dict[str, Any]]] = None,
    retrieved_chunks: Optional[Sequence[dict[str, Any]]] = None,
    model_name: Optional[str] = None,
    retrieval_top_k: Optional[int] = None,
    session_id: Optional[str] = None,
    db_path: Optional[Path] = None,
) -> dict[str, Any]:
    """Save feedback for a response, or replace the rating and comment.

    response_id is UNIQUE, so a second save for the same answer updates the
    existing row (rating, comment, updated_at) and never adds another one. The
    original question, answer, sources and created_at are left untouched.
    """
    response_id = _validate_response_id(response_id)
    rating = _validate_rating(rating)
    comment = _validate_comment(comment)
    question = _required_text(question, "question", MAX_QUESTION_LENGTH)
    answer = _required_text(answer, "answer", MAX_ANSWER_LENGTH)
    if retrieval_top_k is not None and (
        isinstance(retrieval_top_k, bool)
        or not isinstance(retrieval_top_k, int)
        or retrieval_top_k <= 0
    ):
        raise ValueError("retrieval_top_k must be a positive integer.")

    now = _utc_now()
    try:
        with closing(_connect(db_path)) as connection, connection:
            connection.execute(
                """
                INSERT INTO feedback (
                    response_id, session_id, question, answer, rating, comment,
                    sources_json, retrieved_chunks_json, model_name,
                    retrieval_top_k, created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, NULL)
                ON CONFLICT(response_id) DO UPDATE SET
                    rating = excluded.rating,
                    comment = excluded.comment,
                    updated_at = ?
                """,
                (
                    response_id, session_id, question, answer, rating, comment,
                    _serialize_sources(sources),
                    _serialize_chunks(retrieved_chunks),
                    model_name, retrieval_top_k, now, now,
                ),
            )
    except sqlite3.Error as error:
        logger.error("Could not save feedback: %s", error)
        raise FeedbackStorageError("Could not save your feedback.") from error

    saved = get_feedback_by_response_id(response_id, db_path=db_path)
    logger.info("Saved feedback for response_id=%s rating=%d.",
                response_id, rating)
    return saved  # type: ignore[return-value]  # the row was just written


def get_feedback_by_response_id(
    response_id: str, db_path: Optional[Path] = None
) -> Optional[dict[str, Any]]:
    """Return the feedback row for a response id, or None when there is none."""
    response_id = _validate_response_id(response_id)
    try:
        with closing(_connect(db_path)) as connection:
            row = connection.execute(
                "SELECT * FROM feedback WHERE response_id = ?", (response_id,)
            ).fetchone()
    except sqlite3.Error as error:
        logger.error("Could not read feedback: %s", error)
        raise FeedbackStorageError("Could not read feedback.") from error
    return _row_to_dict(row) if row else None


def update_feedback(
    response_id: str,
    rating: Optional[int] = None,
    comment: Optional[str] = _UNSET,
    db_path: Optional[Path] = None,
) -> dict[str, Any]:
    """Change the rating and/or comment of existing feedback.

    Leave an argument out to keep its current value; pass comment=None to
    clear the comment. Raises FeedbackNotFoundError when nothing was saved yet.
    """
    response_id = _validate_response_id(response_id)
    assignments: list[str] = []
    values: list[Any] = []
    if rating is not None:
        assignments.append("rating = ?")
        values.append(_validate_rating(rating))
    if comment is not _UNSET:
        assignments.append("comment = ?")
        values.append(_validate_comment(comment))
    if not assignments:
        raise ValueError("Provide a rating or a comment to update.")

    assignments.append("updated_at = ?")
    values.extend([_utc_now(), response_id])
    try:
        with closing(_connect(db_path)) as connection, connection:
            cursor = connection.execute(
                # The column names are fixed strings above, never user input.
                f"UPDATE feedback SET {', '.join(assignments)} "
                "WHERE response_id = ?",
                values,
            )
    except sqlite3.Error as error:
        logger.error("Could not update feedback: %s", error)
        raise FeedbackStorageError(
            "Could not update your feedback.") from error

    if cursor.rowcount == 0:
        raise FeedbackNotFoundError(
            f"No feedback exists for response_id={response_id}.")
    return get_feedback_by_response_id(  # type: ignore[return-value]
        response_id, db_path=db_path)


def get_feedback_summary(db_path: Optional[Path] = None) -> dict[str, Any]:
    """Return total, thumbs-up and thumbs-down counts plus the positive share."""
    try:
        with closing(_connect(db_path)) as connection:
            row = connection.execute(
                """
                SELECT COUNT(*) AS total,
                       COALESCE(SUM(rating = ?), 0) AS thumbs_up,
                       COALESCE(SUM(rating = ?), 0) AS thumbs_down
                FROM feedback
                """,
                (RATING_THUMBS_UP, RATING_THUMBS_DOWN),
            ).fetchone()
    except sqlite3.Error as error:
        logger.error("Could not summarize feedback: %s", error)
        raise FeedbackStorageError("Could not read the feedback summary.") from error

    total, thumbs_up, thumbs_down = row["total"], row["thumbs_up"], row["thumbs_down"]
    return {
        "total": total,
        "thumbs_up": thumbs_up,
        "thumbs_down": thumbs_down,
        "positive_percentage": round(100 * thumbs_up / total, 1) if total else 0.0,
    }


def list_recent_feedback(
    limit: int = DEFAULT_LIST_LIMIT, db_path: Optional[Path] = None
) -> list[dict[str, Any]]:
    """Return the newest feedback rows without answers, excerpts or chunk data.

    Meant for developers: each item has the id, response id, question, rating,
    comment, timestamps and the source filenames.
    """
    if isinstance(limit, bool) or not isinstance(limit, int) \
            or not 1 <= limit <= MAX_LIST_LIMIT:
        raise ValueError(f"limit must be between 1 and {MAX_LIST_LIMIT}.")

    try:
        with closing(_connect(db_path)) as connection:
            rows = connection.execute(
                """
                SELECT id, response_id, question, rating, comment,
                       sources_json, created_at, updated_at
                FROM feedback ORDER BY id DESC LIMIT ?
                """,
                (limit,),
            ).fetchall()
    except sqlite3.Error as error:
        logger.error("Could not list feedback: %s", error)
        raise FeedbackStorageError("Could not read feedback.") from error

    records = []
    for row in rows:
        record = _row_to_dict(row)
        try:
            sources = json.loads(record.pop("sources_json") or "[]")
        except json.JSONDecodeError:
            sources = []
        filenames = []
        for source in sources:
            name = source.get("filename")
            if name and name not in filenames:
                filenames.append(name)
        record["source_filenames"] = filenames
        records.append(record)
    return records
