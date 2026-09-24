"""Export thumbs-down questions as unreviewed candidate evaluation cases.

Usage:
    python scripts/export_feedback_candidates.py
    python scripts/export_feedback_candidates.py --output data/evaluation/feedback_candidates.jsonl

The feedback database is opened read-only. Candidates go to a separate file
and are never added to the golden dataset automatically; a human must review
them first. Re-running skips questions already exported or already in the
golden dataset.
"""

import argparse
import sys
from pathlib import Path
from typing import Optional, Sequence

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from evaluation.candidates import (  # noqa: E402
    DEFAULT_CANDIDATES_PATH,
    FeedbackExportError,
    append_candidates,
    build_candidates,
    read_existing_candidate_ids,
)
from evaluation.dataset import (  # noqa: E402
    DEFAULT_DATASET_PATH,
    EvaluationDatasetError,
    read_evaluation_dataset,
)
from services.feedback_service import FEEDBACK_DB_PATH  # noqa: E402


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Export negatively rated questions as unreviewed candidates.")
    parser.add_argument("--output", type=Path, default=DEFAULT_CANDIDATES_PATH,
                        help="candidates JSONL file (default: %(default)s)")
    parser.add_argument("--db", type=Path, default=FEEDBACK_DB_PATH,
                        help="feedback database to read (default: %(default)s)")
    parser.add_argument("--golden", type=Path, default=DEFAULT_DATASET_PATH,
                        help="golden dataset whose questions are skipped "
                             "(default: %(default)s)")
    args = parser.parse_args(argv)

    try:
        golden_questions: list[str] = []
        if args.golden.is_file():
            golden_questions = [
                case.question for case in read_evaluation_dataset(args.golden).cases]
        known_ids = read_existing_candidate_ids(args.output)
        candidates = build_candidates(
            args.db, exclude_questions=golden_questions, known_ids=known_ids)
        written = append_candidates(args.output, candidates)
    except (FeedbackExportError, EvaluationDatasetError) as error:
        print(f"Error: {error}", file=sys.stderr)
        return 1

    print(f"Feedback database (read-only): {args.db}")
    print(f"Candidates file:               {args.output}")
    print(f"New candidates written:        {written}")
    print(f"Already exported (skipped):    {len(known_ids)}")
    print("All candidates are marked 'unreviewed'. Review them by hand before "
          "adding anything to the golden dataset.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
