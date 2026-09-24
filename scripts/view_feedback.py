"""Developer tool: print the feedback summary and the most recent ratings.

Usage:
    python scripts/view_feedback.py
    python scripts/view_feedback.py --limit 20

Answers, source excerpts, embeddings and secrets are never printed.
"""

import argparse
import sys
from pathlib import Path
from typing import Any, Optional, Sequence

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from services.feedback_service import (  # noqa: E402
    DEFAULT_LIST_LIMIT,
    MAX_LIST_LIMIT,
    FeedbackStorageError,
    get_feedback_summary,
    list_recent_feedback,
)

RATING_LABELS = {1: "thumbs up (1)", -1: "thumbs down (-1)"}


def parse_limit(value: str) -> int:
    """Argparse type: an integer from 1 to MAX_LIST_LIMIT."""
    try:
        limit = int(value)
    except ValueError:
        raise argparse.ArgumentTypeError(f"{value!r} is not a whole number.")
    if not 1 <= limit <= MAX_LIST_LIMIT:
        raise argparse.ArgumentTypeError(
            f"limit must be between 1 and {MAX_LIST_LIMIT}.")
    return limit


def format_record(record: dict[str, Any]) -> str:
    """Render one feedback row as readable text."""
    filenames = ", ".join(record["source_filenames"]) or "none"
    return "\n".join([
        f"ID:           {record['id']}",
        f"Response ID:  {record['response_id']}",
        f"Question:     {record['question']}",
        f"Rating:       {RATING_LABELS.get(record['rating'], record['rating'])}",
        f"Comment:      {record['comment'] or '-'}",
        f"Created at:   {record['created_at']}",
        f"Updated at:   {record['updated_at'] or '-'}",
        f"Sources:      {filenames}",
    ])


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Show feedback saved for generated answers.")
    parser.add_argument(
        "--limit", type=parse_limit, default=DEFAULT_LIST_LIMIT,
        help=f"number of recent records to show (1-{MAX_LIST_LIMIT}, "
             f"default {DEFAULT_LIST_LIMIT})")
    args = parser.parse_args(argv)

    # Windows consoles often use a legacy code page; never crash on Unicode.
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(errors="replace")

    try:
        summary = get_feedback_summary()
        records = list_recent_feedback(args.limit)
    except FeedbackStorageError as error:
        print(f"Error: {error}", file=sys.stderr)
        return 1

    print(
        f"Total ratings: {summary['total']}  |  "
        f"Thumbs up: {summary['thumbs_up']}  |  "
        f"Thumbs down: {summary['thumbs_down']}  |  "
        f"Positive: {summary['positive_percentage']}%"
    )
    if not records:
        print("\nNo feedback has been saved yet.")
        return 0

    print(f"\nMost recent {len(records)} record(s):")
    for record in records:
        print("\n" + format_record(record))
    return 0


if __name__ == "__main__":
    sys.exit(main())
