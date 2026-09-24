import logging
from io import BytesIO
from pathlib import Path
from typing import Any, Optional

from pypdf import PdfReader

logger = logging.getLogger(__name__)

# A section is one piece of a document together with the page it came from:
# {"text": str, "page": int | None}. TXT and Markdown files have no pages,
# so their page is None rather than an invented number.
Section = dict[str, Any]


def load_document(uploaded_file: Any) -> list[Section]:
    """Extract text sections from an uploaded TXT, Markdown, or PDF file."""
    file_extension = Path(uploaded_file.name).suffix.lower()
    file_bytes = uploaded_file.getvalue()

    if file_extension in {".txt", ".md"}:
        return [{"text": file_bytes.decode("utf-8"), "page": None}]

    if file_extension == ".pdf":
        reader = PdfReader(BytesIO(file_bytes))
        sections: list[Section] = []
        for page_number, page in enumerate(reader.pages, start=1):
            text = page.extract_text() or ""
            if text.strip():
                sections.append({"text": text, "page": page_number})
        logger.info(
            "Extracted %d non-empty pages from a %d page PDF.",
            len(sections),
            len(reader.pages),
        )
        return sections

    raise ValueError(
        f"Unsupported file type: {file_extension or 'unknown'}. "
        "Please upload a PDF, TXT, or Markdown file."
    )


def section_text(sections: list[Section]) -> str:
    """Join section texts back into the full document text."""
    return "\n".join(section["text"] for section in sections)


def document_id_from_name(filename: str) -> Optional[str]:
    """Derive a stable, filesystem-safe document id from a filename."""
    stem = Path(filename).stem.strip().lower()
    slug = "".join(
        character if character.isalnum() else "_" for character in stem
    ).strip("_")
    while "__" in slug:
        slug = slug.replace("__", "_")
    return slug or None
