import re
from typing import Any, Iterator, Optional

# Markdown-style heading, used to label a chunk with the section it sits under.
_HEADING_PATTERN = re.compile(r"^#{1,6}\s+(.*\S)\s*$", re.MULTILINE)

Chunk = dict[str, Any]


def _iter_chunk_spans(
    text: str, chunk_size: int, overlap: int
) -> Iterator[tuple[int, str]]:
    """Yield (start offset, chunk text) pairs for overlapping chunks."""
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive.")

    if overlap < 0 or overlap >= chunk_size:
        raise ValueError(
            "overlap must be non-negative and smaller than chunk_size.")

    step = chunk_size - overlap
    for start in range(0, len(text), step):
        chunk = text[start:start + chunk_size]
        if chunk:
            yield start, chunk


def chunk_text(text: str, chunk_size: int = 1000, overlap: int = 200) -> list[str]:
    """Split text into overlapping character-based chunks."""
    return [chunk for _, chunk in _iter_chunk_spans(text, chunk_size, overlap)]


def _find_heading(text: str, start: int) -> Optional[str]:
    """Return the last Markdown heading the given offset sits under.

    The whole text is scanned so a heading split across a chunk boundary is
    never matched half-way through.
    """
    heading = None
    for match in _HEADING_PATTERN.finditer(text):
        if match.start() > start:
            break
        heading = match.group(1)
    return heading


def chunk_sections(
    sections: list[dict[str, Any]],
    chunk_size: int = 1000,
    overlap: int = 200,
) -> list[Chunk]:
    """Chunk document sections, keeping page numbers and section headings.

    Returns chunks shaped {"text", "page", "section", "chunk_index"} where
    chunk_index counts across the whole document. Missing page or section
    metadata stays None instead of being invented.
    """
    chunks: list[Chunk] = []

    for section in sections:
        text = section.get("text", "")
        page = section.get("page")

        for start, chunk in _iter_chunk_spans(text, chunk_size, overlap):
            chunks.append({
                "text": chunk,
                "page": page,
                "section": _find_heading(text, start),
                "chunk_index": len(chunks),
            })

    return chunks
