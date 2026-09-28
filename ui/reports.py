"""Read-only access to Stage 10 evaluation reports for the Evaluation page.

Security: the page never accepts a path from the browser. It lists report
folders found under two fixed roots, and a selection is resolved only by
looking its id up in that list. Resolved paths must also stay inside a root,
so names like ``../../.env`` or symlinks pointing elsewhere are rejected.
"""

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

from ui.config import MAX_DOWNLOAD_BYTES, MAX_FAILURE_ROWS

PROJECT_ROOT = Path(__file__).resolve().parent.parent
BASELINE_DIR = PROJECT_ROOT / "reports" / "baseline"
RUNS_DIR = PROJECT_ROOT / "reports" / "evaluation"

# The only files offered for download.
DOWNLOADABLE_FILES: tuple[str, ...] = (
    "summary.json", "results.csv", "failures.jsonl")

_SAFE_NAME = re.compile(r"^[A-Za-z0-9_\-]+$")


class ReportError(Exception):
    """A report is missing or malformed; str(error) is safe to show."""


@dataclass(frozen=True)
class ReportRef:
    report_id: str  # "baseline" or "runs/<folder name>"
    label: str
    path: Path


def _inside(path: Path, root: Path) -> bool:
    try:
        return path.resolve().is_relative_to(root.resolve())
    except OSError:
        return False


def list_reports(
    baseline_dir: Path = BASELINE_DIR, runs_dir: Path = RUNS_DIR,
) -> list[ReportRef]:
    """Report folders containing summary.json, newest first.

    Timestamped runs sort by folder name (newest first); the baseline is last.
    """
    refs: list[ReportRef] = []
    if runs_dir.is_dir():
        for folder in sorted(runs_dir.iterdir(), key=lambda p: p.name, reverse=True):
            if (folder.is_dir() and _SAFE_NAME.match(folder.name)
                    and _inside(folder, runs_dir)
                    and (folder / "summary.json").is_file()):
                refs.append(ReportRef(f"runs/{folder.name}", folder.name, folder))
    if (baseline_dir / "summary.json").is_file():
        refs.append(ReportRef("baseline", "Baseline", baseline_dir))
    return refs


def resolve_report(report_id: str, refs: list[ReportRef]) -> ReportRef:
    """Return the listed report with this id; anything else is rejected."""
    for ref in refs:
        if ref.report_id == report_id:
            return ref
    raise ReportError("That report does not exist.")


def load_summary(ref: ReportRef) -> dict[str, Any]:
    """Parse summary.json and check the fields the page relies on."""
    path = ref.path / "summary.json"
    try:
        summary = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as error:
        raise ReportError("This report has no summary.json.") from error
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ReportError("summary.json could not be read or is not valid JSON.") from error
    if not isinstance(summary, dict) or not isinstance(summary.get("metrics"), dict):
        raise ReportError("summary.json does not contain a metrics section.")
    return summary


def load_failures(ref: ReportRef, limit: int = MAX_FAILURE_ROWS) -> tuple[list[dict], int]:
    """Up to ``limit`` failure records plus the number of unreadable lines.

    Only the first ``limit`` lines are read, never the whole results file.
    """
    path = ref.path / "failures.jsonl"
    if not path.is_file():
        return [], 0
    records, bad_lines = [], 0
    try:
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                if len(records) >= limit:
                    break
                if not line.strip():
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    bad_lines += 1
                    continue
                if isinstance(record, dict):
                    records.append(record)
                else:
                    bad_lines += 1
    except (OSError, UnicodeDecodeError) as error:
        raise ReportError("failures.jsonl could not be read.") from error
    return records, bad_lines


def failure_rows(records: list[dict]) -> list[dict[str, Any]]:
    """Flat rows for display. Golden expected answers are not included."""
    rows = []
    for record in records:
        rank = record.get("first_relevant_rank")
        coverage = record.get("keyword_coverage")
        rows.append({
            "Case": record.get("case_id", ""),
            "Category": record.get("category", ""),
            "Reasons": ", ".join(record.get("failure_reasons") or []),
            "Question": record.get("question", ""),
            "First relevant rank": "" if rank is None else rank,
            "Keyword coverage": "" if coverage is None else round(coverage, 2),
            "Error": record.get("error") or "",
        })
    return rows


def read_download(ref: ReportRef, filename: str) -> Optional[bytes]:
    """Bytes of an allow-listed report file, or None if missing or too large."""
    if filename not in DOWNLOADABLE_FILES:
        raise ReportError("That file cannot be downloaded.")
    path = ref.path / filename
    if not path.is_file() or not _inside(path, ref.path):
        return None
    if path.stat().st_size > MAX_DOWNLOAD_BYTES:
        return None
    return path.read_bytes()
