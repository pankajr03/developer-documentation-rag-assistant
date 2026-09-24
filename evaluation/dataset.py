"""Load and validate the JSONL golden evaluation dataset.

Nothing here calls an LLM, the embedding API, or the vector store.
"""

import json
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Union

from pydantic import ValidationError

from evaluation.schemas import DIFFICULTIES, EvaluationCase

DEFAULT_DATASET_PATH = (
    Path(__file__).resolve().parent.parent
    / "data" / "evaluation" / "golden_dataset.jsonl"
)


@dataclass(frozen=True)
class DatasetIssue:
    """One problem found in the dataset file."""

    line: int
    message: str

    def __str__(self) -> str:
        return f"line {self.line}: {self.message}"


class EvaluationDatasetError(Exception):
    """The dataset could not be read, or it contains invalid records."""

    def __init__(self, message: str, issues: Iterable[DatasetIssue] = ()):
        self.issues = list(issues)
        details = "\n".join(f"  {issue}" for issue in self.issues)
        super().__init__(f"{message}\n{details}" if details else message)


class DatasetNotFoundError(EvaluationDatasetError):
    """The dataset file does not exist."""


@dataclass
class DatasetReadResult:
    """Valid cases plus every problem found, so nothing is skipped silently."""

    cases: list[EvaluationCase] = field(default_factory=list)
    issues: list[DatasetIssue] = field(default_factory=list)


def _format_validation_error(error: ValidationError) -> str:
    parts = []
    for item in error.errors(include_url=False):
        location = ".".join(str(part) for part in item["loc"])
        message = item["msg"].removeprefix("Value error, ")
        parts.append(f"{location}: {message}" if location else message)
    return "; ".join(parts)


def read_evaluation_dataset(path: Union[str, Path]) -> DatasetReadResult:
    """Read a JSONL file and report every invalid line instead of stopping.

    Blank lines are ignored but still counted, so reported line numbers match
    the ones shown in an editor. The first record with a given id is kept and
    later duplicates are reported as issues.
    """
    path = Path(path)
    if not path.is_file():
        raise DatasetNotFoundError(f"Dataset file not found: {path}")

    result = DatasetReadResult()
    first_line_for_id: dict[str, int] = {}

    try:
        # utf-8-sig also accepts a BOM, which some Windows editors add.
        with path.open("r", encoding="utf-8-sig") as handle:
            for line_number, raw_line in enumerate(handle, start=1):
                line = raw_line.strip()
                if not line:
                    continue

                try:
                    record = json.loads(line)
                except json.JSONDecodeError as error:
                    result.issues.append(DatasetIssue(
                        line_number, f"malformed JSON ({error.msg}, column {error.colno})"))
                    continue
                if not isinstance(record, dict):
                    result.issues.append(DatasetIssue(
                        line_number, "each line must be a JSON object"))
                    continue

                try:
                    case = EvaluationCase.model_validate(record)
                except ValidationError as error:
                    label = record.get("id")
                    prefix = f"[{label}] " if isinstance(label, str) and label else ""
                    result.issues.append(DatasetIssue(
                        line_number, prefix + _format_validation_error(error)))
                    continue

                if case.id in first_line_for_id:
                    result.issues.append(DatasetIssue(
                        line_number,
                        f"duplicate id {case.id!r} (first used on line "
                        f"{first_line_for_id[case.id]})"))
                    continue
                first_line_for_id[case.id] = line_number
                result.cases.append(case)
    except UnicodeDecodeError as error:
        raise EvaluationDatasetError(
            f"Dataset file is not valid UTF-8: {path}") from error
    except OSError as error:
        raise EvaluationDatasetError(
            f"Could not read dataset file {path}: {error.strerror or error}"
        ) from error

    return result


def load_evaluation_dataset(
    path: Union[str, Path] = DEFAULT_DATASET_PATH,
) -> list[EvaluationCase]:
    """Return every case, or raise EvaluationDatasetError listing all problems."""
    result = read_evaluation_dataset(path)
    if result.issues:
        raise EvaluationDatasetError(
            f"{len(result.issues)} invalid record(s) in {path}", result.issues)
    if not result.cases:
        raise EvaluationDatasetError(f"Dataset contains no cases: {path}")
    return result.cases


def summarize_dataset(cases: Iterable[EvaluationCase]) -> dict[str, Any]:
    """Count cases by answerability, category and difficulty."""
    cases = list(cases)
    by_difficulty = Counter(case.difficulty for case in cases)
    return {
        "total": len(cases),
        "answerable": sum(case.answerable for case in cases),
        "unanswerable": sum(not case.answerable for case in cases),
        "by_category": dict(sorted(Counter(c.category for c in cases).items())),
        "by_difficulty": {level: by_difficulty[level] for level in DIFFICULTIES},
    }


def find_missing_source_ids(
    cases: Iterable[EvaluationCase], available_ids: Iterable[str]
) -> dict[str, list[str]]:
    """Map case id -> expected source ids that are not in available_ids."""
    available = set(available_ids)
    missing: dict[str, list[str]] = {}
    for case in cases:
        absent = [sid for sid in case.expected_source_ids if sid not in available]
        if absent:
            missing[case.id] = absent
    return missing
