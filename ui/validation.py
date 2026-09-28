"""Validation for questions and uploaded files. No Streamlit, no I/O."""

import re
import unicodedata
from dataclasses import dataclass
from pathlib import PurePath
from typing import Optional

from ui.config import (
    ALLOWED_EXTENSIONS,
    MAX_QUESTION_CHARS,
    MAX_UPLOAD_BYTES,
)

MAX_FILENAME_CHARS = 120
_UNSAFE_FILENAME_CHARS = re.compile(r"[^\w .()\-]")


class QuestionValidationError(ValueError):
    """The question cannot be sent to the assistant."""


def validate_question(question: Optional[str], max_chars: int = MAX_QUESTION_CHARS) -> str:
    """Return the trimmed question, or raise with a message fit for users."""
    cleaned = (question or "").strip()
    if not cleaned:
        raise QuestionValidationError("Please enter a question.")
    if len(cleaned) > max_chars:
        raise QuestionValidationError(
            f"Your question is {len(cleaned):,} characters long. "
            f"Please keep it under {max_chars:,} characters.")
    return cleaned


def sanitize_filename(name: Optional[str]) -> str:
    """Reduce an uploaded filename to a safe display name.

    Security: the browser controls this value, so it is never used as a
    filesystem path. Any directory part (``../../etc/passwd``,
    ``C:\\Users\\x\\a.pdf``) is dropped, control characters are removed, and
    only letters, digits, spaces and ``. _ - ( )`` are kept. Ordinary names
    such as ``Speed Schedule (1).pdf`` are unchanged, so their document id
    stays the same as before.
    """
    name = unicodedata.normalize("NFKC", name or "")
    # Take the last path component for both / and \ separators.
    name = re.split(r"[\\/]", name)[-1]
    name = "".join(c for c in name if unicodedata.category(c)[0] != "C")
    name = _UNSAFE_FILENAME_CHARS.sub("_", name)
    name = re.sub(r"\s+", " ", name).strip(" .")
    if len(name) > MAX_FILENAME_CHARS:
        suffix = PurePath(name).suffix[:10]
        name = name[:MAX_FILENAME_CHARS - len(suffix)].rstrip(" .") + suffix
    return name or "document"


@dataclass(frozen=True)
class UploadCheck:
    """Outcome of validating one uploaded file before any processing."""

    display_name: str
    ok: bool
    message: str = ""


def validate_upload(
    name: Optional[str],
    size: int,
    max_bytes: int = MAX_UPLOAD_BYTES,
) -> UploadCheck:
    """Check the extension, emptiness and size of an uploaded file."""
    display_name = sanitize_filename(name)
    extension = PurePath(display_name).suffix.lower()
    if extension not in ALLOWED_EXTENSIONS:
        allowed = ", ".join(e.lstrip(".").upper() for e in ALLOWED_EXTENSIONS)
        return UploadCheck(display_name, False,
                           f"Unsupported file type. Upload {allowed} files.")
    if size <= 0:
        return UploadCheck(display_name, False, "The file is empty.")
    if size > max_bytes:
        limit_mb = max_bytes / (1024 * 1024)
        return UploadCheck(display_name, False,
                           f"The file is larger than the {limit_mb:g} MB limit.")
    return UploadCheck(display_name, True)
