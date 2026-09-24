"""Validate the golden evaluation dataset.

Usage:
    python scripts/validate_evaluation_dataset.py
    python scripts/validate_evaluation_dataset.py --dataset path/to/file.jsonl
    python scripts/validate_evaluation_dataset.py --check-index

Exits with 0 on PASS and 1 on FAIL, so it can run in CI. --check-index also
confirms that every expected source id exists in the local ChromaDB collection
(it reads chunk ids only and makes no API calls).
"""

import argparse
import sys
from pathlib import Path
from typing import Optional, Sequence

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from evaluation.dataset import (  # noqa: E402
    DEFAULT_DATASET_PATH,
    EvaluationDatasetError,
    find_missing_source_ids,
    read_evaluation_dataset,
    summarize_dataset,
)


def _indexed_chunk_ids() -> set[str]:
    # Imported here so plain validation never needs ChromaDB.
    from rag.vector_store import _get_collection

    return set(_get_collection().get(include=[])["ids"])


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Validate a JSONL evaluation dataset.")
    parser.add_argument(
        "--dataset", type=Path, default=DEFAULT_DATASET_PATH,
        help="path to the JSONL dataset (default: %(default)s)")
    parser.add_argument(
        "--check-index", action="store_true",
        help="also check that expected source ids exist in ChromaDB")
    args = parser.parse_args(argv)

    # Windows consoles often use a legacy code page; never crash on Unicode.
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(errors="replace")

    print(f"Dataset: {args.dataset}")
    try:
        result = read_evaluation_dataset(args.dataset)
    except EvaluationDatasetError as error:
        print(f"Error: {error}")
        print("\nResult: FAIL")
        return 1

    summary = summarize_dataset(result.cases)
    print(f"Total valid cases: {summary['total']}")
    print(f"Answerable: {summary['answerable']}")
    print(f"Unanswerable: {summary['unanswerable']}")
    print("By category:")
    for category, count in summary["by_category"].items():
        print(f"  {category}: {count}")
    print("By difficulty:")
    for level, count in summary["by_difficulty"].items():
        print(f"  {level}: {count}")

    problems = [str(issue) for issue in result.issues]
    if not result.cases and not result.issues:
        problems.append("the dataset contains no cases")

    if args.check_index:
        try:
            missing = find_missing_source_ids(result.cases, _indexed_chunk_ids())
        except Exception as error:  # ChromaDB can fail in several ways
            problems.append(f"could not read the ChromaDB index: {error}")
        else:
            for case_id, ids in missing.items():
                problems.append(
                    f"[{case_id}] expected source ids not in the index: "
                    + ", ".join(ids))
            if not missing:
                print("Index check: every expected source id exists.")

    if problems:
        print(f"\nProblems ({len(problems)}):")
        for problem in problems:
            print(f"  {problem}")
        print("\nResult: FAIL")
        return 1

    print("\nResult: PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
