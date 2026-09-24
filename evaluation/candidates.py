"""Turn thumbs-down feedback into *unreviewed* candidate evaluation cases.

Candidates are deliberately not valid golden cases: they have no expected
answer. A human must review each one and write a proper case before it joins
the golden dataset. The feedback database is opened read-only.
"""

import hashlib
import json
import re
import sqlite3
import unicodedata
from contextlib import closing
from pathlib import Path
from typing import Any, Iterable, Union

from services.feedback_service import RATING_THUMBS_DOWN

DEFAULT_CANDIDATES_PATH = (
    Path(__file__).resolve().parent.parent
    / "data" / "evaluation" / "feedback_candidates.jsonl"
)

# Columns the exporter reads. session_id, the answer text, excerpts and model
# details are never read, so they cannot leak into the candidate file.
REQUIRED_COLUMNS = frozenset({
    "response_id", "question", "rating", "comment", "sources_json", "created_at",
})

REVIEW_STATUS_UNREVIEWED = "unreviewed"


class FeedbackExportError(Exception):
    """The feedback database or an existing candidates file cannot be used."""


def normalize_question(question: str) -> str:
    """Casefold, collapse whitespace and drop trailing punctuation.

    Used to spot the same question asked slightly differently.
    """
    text = unicodedata.normalize("NFKC", question).casefold()
    text = re.sub(r"\s+", " ", text).strip()
    return text.rstrip(" ?!.")


def candidate_id(question: str) -> str:
    """Stable id derived from the normalized question."""
    digest = hashlib.sha256(normalize_question(question).encode("utf-8"))
    return f"cand_{digest.hexdigest()[:12]}"


def _read_negative_feedback(db_path: Path) -> list[sqlite3.Row]:
    if not db_path.is_file():
        raise FeedbackExportError(f"Feedback database not found: {db_path}")

    # mode=ro guarantees this tool can never modify or create feedback data.
    uri = f"{db_path.resolve().as_uri()}?mode=ro"
    try:
        with closing(sqlite3.connect(uri, uri=True)) as connection:
            connection.row_factory = sqlite3.Row
            columns = {row["name"] for row in
                       connection.execute("PRAGMA table_info(feedback)")}
            if not columns:
                raise FeedbackExportError(
                    "The feedback database has no 'feedback' table.")
            missing = REQUIRED_COLUMNS - columns
            if missing:
                raise FeedbackExportError(
                    "The feedback table is missing column(s): "
                    + ", ".join(sorted(missing)))
            return connection.execute(
                "SELECT response_id, question, comment, sources_json, created_at "
                "FROM feedback WHERE rating = ? ORDER BY created_at, id",
                (RATING_THUMBS_DOWN,),
            ).fetchall()
    except sqlite3.Error as error:
        raise FeedbackExportError(
            f"Could not read the feedback database: {error}") from error


def _retrieved_chunk_ids(sources_json: Any) -> list[str]:
    try:
        sources = json.loads(sources_json or "[]")
    except (TypeError, json.JSONDecodeError):
        return []
    ids: list[str] = []
    for source in sources if isinstance(sources, list) else []:
        chunk_id = source.get("chunk_id") if isinstance(source, dict) else None
        if isinstance(chunk_id, str) and chunk_id not in ids:
            ids.append(chunk_id)
    return ids


def build_candidates(
    db_path: Union[str, Path],
    exclude_questions: Iterable[str] = (),
    known_ids: Iterable[str] = (),
) -> list[dict[str, Any]]:
    """Return one candidate per distinct thumbs-down question.

    Questions equal (after normalization) to any in exclude_questions, or whose
    candidate id is in known_ids, are skipped. Repeated ratings of the same
    question are merged into one candidate.
    """
    excluded = {normalize_question(q) for q in exclude_questions}
    skipped_ids = set(known_ids)

    merged: dict[str, dict[str, Any]] = {}
    for row in _read_negative_feedback(Path(db_path)):
        question = (row["question"] or "").strip()
        if not question or normalize_question(question) in excluded:
            continue
        cid = candidate_id(question)
        if cid in skipped_ids:
            continue

        candidate = merged.setdefault(cid, {
            "candidate_id": cid,
            "review_status": REVIEW_STATUS_UNREVIEWED,
            "question": question,
            "feedback_count": 0,
            "feedback_comments": [],
            "retrieved_source_ids": [],
            "latest_feedback_at": row["created_at"],
            "review_notes": "",
        })
        candidate["feedback_count"] += 1
        candidate["latest_feedback_at"] = row["created_at"]
        comment = (row["comment"] or "").strip()
        if comment and comment not in candidate["feedback_comments"]:
            candidate["feedback_comments"].append(comment)
        for chunk_id in _retrieved_chunk_ids(row["sources_json"]):
            if chunk_id not in candidate["retrieved_source_ids"]:
                candidate["retrieved_source_ids"].append(chunk_id)

    return list(merged.values())


def read_existing_candidate_ids(path: Union[str, Path]) -> set[str]:
    """Return candidate ids already in an output file (empty if it is absent)."""
    path = Path(path)
    if not path.is_file():
        return set()
    ids: set[str] = set()
    try:
        with path.open("r", encoding="utf-8-sig") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                try:
                    ids.add(json.loads(line)["candidate_id"])
                except (json.JSONDecodeError, KeyError, TypeError) as error:
                    raise FeedbackExportError(
                        f"{path} line {line_number} is not a candidate record; "
                        "fix or remove it before exporting again.") from error
    except (OSError, UnicodeDecodeError) as error:
        raise FeedbackExportError(f"Could not read {path}: {error}") from error
    return ids


def append_candidates(
    path: Union[str, Path], candidates: Iterable[dict[str, Any]]
) -> int:
    """Append candidates to a JSONL file, keeping anything a reviewer edited."""
    candidates = list(candidates)
    if not candidates:
        return 0
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    # Start on a fresh line if the file was saved without a trailing newline.
    needs_newline = False
    if path.is_file() and path.stat().st_size:
        with path.open("rb") as handle:
            handle.seek(-1, 2)
            needs_newline = handle.read(1) not in (b"\n", b"\r")

    with path.open("a", encoding="utf-8", newline="\n") as handle:
        if needs_newline:
            handle.write("\n")
        for candidate in candidates:
            handle.write(json.dumps(candidate, ensure_ascii=False) + "\n")
    return len(candidates)
