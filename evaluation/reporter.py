"""Write Stage 10 evaluation reports. Existing runs are never overwritten.

Each run gets its own directory containing:
    summary.json    run metadata, configuration and aggregate metrics
    results.jsonl   one complete CaseResult per line
    results.csv     important flat fields for a quick look in a spreadsheet
    failures.jsonl  cases with errors or weak metrics, with failure_reasons
"""

import csv
import json
import platform
import subprocess
from datetime import datetime
from functools import lru_cache
from importlib import metadata
from pathlib import Path
from typing import Any, Optional, Sequence

from evaluation.config import EvaluationConfig
from evaluation.results import CaseResult

PROJECT_ROOT = Path(__file__).resolve().parent.parent

CSV_FIELDS: tuple[str, ...] = (
    "case_id", "category", "difficulty", "answerable",
    "source_hit", "hit_at_1", "hit_at_3", "hit_at_5",
    "first_relevant_rank", "reciprocal_rank",
    "keyword_coverage", "missing_keywords",
    "citation_count", "citation_precision", "expected_source_citation_hit",
    "grounded", "abstained", "correct_abstention", "incorrect_abstention",
    "judge_correctness", "judge_grounding", "judge_relevance",
    "judge_unsupported_claims", "judge_error",
    "retrieval_latency_ms", "total_latency_ms",
    "retrieved_chunk_ids", "error", "failure_reasons",
)

_TRACKED_PACKAGES = ("openai", "chromadb", "pydantic", "pypdf", "streamlit")


def create_run_directory(config: EvaluationConfig, now: datetime) -> Path:
    """Create a new, empty directory for this run.

    With config.run_directory the exact path is used and must not already
    contain files. Otherwise a timestamped folder is created inside
    config.output_directory, adding _2, _3... if that name is taken.
    """
    if config.run_directory:
        run_dir = Path(config.run_directory)
        if run_dir.exists() and (not run_dir.is_dir() or any(run_dir.iterdir())):
            raise FileExistsError(
                f"Report directory already exists and is not empty: {run_dir}")
        run_dir.mkdir(parents=True, exist_ok=True)
        return run_dir

    base = Path(config.output_directory)
    name = now.strftime("%Y-%m-%d_%H%M%S")
    for attempt in range(1, 1000):
        run_dir = base / (name if attempt == 1 else f"{name}_{attempt}")
        try:
            run_dir.mkdir(parents=True, exist_ok=False)
            return run_dir
        except FileExistsError:
            continue
    raise FileExistsError(f"Could not create a new report directory in {base}")


def _csv_value(value: Any) -> Any:
    if value is None:
        return ""
    if isinstance(value, list):
        return "; ".join(str(item) for item in value)
    return value


def _csv_row(result: CaseResult) -> dict[str, Any]:
    data = result.model_dump()
    judge = result.judge
    data.update({
        "grounded": result.grounding.grounded if result.grounding else None,
        "judge_correctness": judge.correctness if judge else None,
        "judge_grounding": judge.grounding if judge else None,
        "judge_relevance": judge.relevance if judge else None,
        "judge_unsupported_claims": judge.unsupported_claims if judge else None,
        "retrieved_chunk_ids": [s.chunk_id for s in result.retrieved_sources],
    })
    return {name: _csv_value(data.get(name)) for name in CSV_FIELDS}


def _write_jsonl(path: Path, results: Sequence[CaseResult]) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for result in results:
            handle.write(result.model_dump_json() + "\n")


def write_reports(
    run_dir: Path, summary: dict[str, Any], results: Sequence[CaseResult],
) -> dict[str, Path]:
    """Write the four report files into run_dir and return their paths."""
    paths = {
        "summary": run_dir / "summary.json",
        "results": run_dir / "results.jsonl",
        "csv": run_dir / "results.csv",
        "failures": run_dir / "failures.jsonl",
    }
    paths["summary"].write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8")
    _write_jsonl(paths["results"], results)
    _write_jsonl(paths["failures"], [r for r in results if r.failure_reasons])
    # utf-8-sig so Excel on Windows shows non-ASCII text correctly.
    with paths["csv"].open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS)
        writer.writeheader()
        for result in results:
            writer.writerow(_csv_row(result))
    return paths


def _git_commit() -> Optional[str]:
    try:
        completed = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"], cwd=PROJECT_ROOT,
            capture_output=True, text=True, timeout=5, check=True)
    except (OSError, subprocess.SubprocessError):
        return None
    return completed.stdout.strip() or None


@lru_cache(maxsize=1)
def environment_info() -> dict[str, Any]:
    """Safe environment details: versions only, never environment variables.

    Cached because package metadata lookups are slow and cannot change
    during one process.
    """
    packages = {}
    for name in _TRACKED_PACKAGES:
        try:
            packages[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            packages[name] = None
    return {
        "python": platform.python_version(),
        "platform": platform.system(),
        "packages": packages,
        "git_commit": _git_commit(),
    }
