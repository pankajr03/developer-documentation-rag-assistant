"""Run the golden evaluation dataset through the real RAG pipeline.

Usage:
    python scripts/run_rag_evaluation.py
    python scripts/run_rag_evaluation.py --top-k 5 --category race_distances
    python scripts/run_rag_evaluation.py --skip-generation
    python scripts/run_rag_evaluation.py --recommended-thresholds

Exit codes:
    0  PASS: every configured threshold met, and no more case errors than
       --max-case-errors (default 0).
    1  FAIL: a threshold was missed or too many cases raised errors. The
       reports are still written.
    2  The run could not start: invalid configuration, dataset not loadable,
       or the vector database is unavailable. No reports are written.
"""

import argparse
import logging
import sys
from pathlib import Path
from typing import Any, Optional, Sequence

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pydantic import ValidationError  # noqa: E402

from evaluation.config import (  # noqa: E402
    DEFAULT_OUTPUT_DIRECTORY,
    RECOMMENDED_THRESHOLDS,
    EvaluationConfig,
)
from evaluation.dataset import (  # noqa: E402
    DEFAULT_DATASET_PATH,
    EvaluationDatasetError,
)
from evaluation.runner import (  # noqa: E402
    EvaluationRun,
    EvaluationSetupError,
    run_evaluation,
)

EXIT_PASS, EXIT_FAIL, EXIT_SETUP_ERROR = 0, 1, 2

_THRESHOLD_ARGS = {
    "fail_below_source_hit": "min_source_hit_rate",
    "fail_below_mrr": "min_mrr",
    "fail_below_keyword_coverage": "min_keyword_coverage",
    "fail_below_abstention": "min_correct_abstention_rate",
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Evaluate the RAG pipeline against the golden dataset.")
    parser.add_argument(
        "--dataset", type=Path, default=DEFAULT_DATASET_PATH,
        help="JSONL golden dataset (default: %(default)s)")
    parser.add_argument(
        "--top-k", type=int, default=5, help="chunks to retrieve (default: 5)")
    parser.add_argument(
        "--case-id", action="append", default=[], dest="case_ids",
        help="evaluate only this case id (repeatable)")
    parser.add_argument("--category", help="evaluate only this category")
    parser.add_argument(
        "--difficulty", help="beginner, intermediate or advanced")
    parser.add_argument("--max-cases", type=int, help="stop after N cases")
    parser.add_argument(
        "--skip-generation", action="store_true",
        help="retrieval metrics only; no answer-generation calls")
    parser.add_argument(
        "--use-llm-judge", action="store_true",
        help="also score answers with an LLM judge (extra cost, off by default)")
    parser.add_argument("--judge-model", help="model for the LLM judge")
    parser.add_argument(
        "--output-dir", default=DEFAULT_OUTPUT_DIRECTORY,
        help="parent folder for timestamped run folders (default: %(default)s)")
    parser.add_argument(
        "--run-dir",
        help="exact report folder for this run; must be new or empty")
    parser.add_argument(
        "--max-case-errors", type=int, default=0,
        help="case errors allowed before the run FAILs (default: 0)")
    parser.add_argument(
        "--recommended-thresholds", action="store_true",
        help="apply the suggested thresholds "
             + ", ".join(f"{k}={v}" for k, v in RECOMMENDED_THRESHOLDS.items()))
    parser.add_argument(
        "--fail-below-source-hit", type=float, metavar="RATE")
    parser.add_argument("--fail-below-mrr", type=float, metavar="SCORE")
    parser.add_argument(
        "--fail-below-keyword-coverage", type=float, metavar="RATE")
    parser.add_argument(
        "--fail-below-abstention", type=float, metavar="RATE",
        help="minimum correct-abstention rate")
    parser.add_argument(
        "-v", "--verbose", action="store_true", help="show progress logs")
    return parser


def config_from_args(args: argparse.Namespace) -> EvaluationConfig:
    """Build and validate the configuration (raises ValidationError)."""
    thresholds: dict[str, Any] = (
        dict(RECOMMENDED_THRESHOLDS) if args.recommended_thresholds else {})
    for arg_name, field_name in _THRESHOLD_ARGS.items():
        value = getattr(args, arg_name)
        if value is not None:
            thresholds[field_name] = value
    return EvaluationConfig(
        top_k=args.top_k,
        max_cases=args.max_cases,
        category=args.category,
        difficulty=args.difficulty,
        case_ids=args.case_ids,
        generate_answers=not args.skip_generation,
        use_llm_judge=args.use_llm_judge,
        judge_model=args.judge_model,
        output_directory=args.output_dir,
        run_directory=args.run_dir,
        max_case_errors=args.max_case_errors,
        thresholds=thresholds,
    )


def _fmt(value: Optional[float]) -> str:
    return "n/a" if value is None else f"{value:.3f}"


def print_summary(run: EvaluationRun) -> None:
    summary = run.summary
    metrics = summary["metrics"]
    print(f"Cases evaluated: {metrics['total_cases']} "
          f"({metrics['answerable_cases']} answerable, "
          f"{metrics['unanswerable_cases']} unanswerable)")
    print(f"Case errors: {metrics['failed_cases']}")
    print(f"Source-hit rate: {_fmt(metrics['source_hit_rate'])}")
    print(f"Hit@1 / Hit@3 / Hit@5: {_fmt(metrics['hit_at_1'])} / "
          f"{_fmt(metrics['hit_at_3'])} / {_fmt(metrics['hit_at_5'])}")
    print(f"MRR: {_fmt(metrics['mrr'])}")
    print(f"Keyword coverage: {_fmt(metrics['average_keyword_coverage'])}")
    print(f"Citation precision: {_fmt(metrics['citation_precision'])}")
    print(f"Correct abstention rate: {_fmt(metrics['correct_abstention_rate'])}")
    print(f"Incorrect abstentions: {metrics['incorrect_abstention_count'] if metrics['incorrect_abstention_count'] is not None else 'n/a'}")
    if metrics["judge_cases"] or metrics["judge_failures"]:
        print(f"LLM judge (estimates) correctness/grounding/relevance: "
              f"{_fmt(metrics['judge_average_correctness'])} / "
              f"{_fmt(metrics['judge_average_grounding'])} / "
              f"{_fmt(metrics['judge_average_relevance'])} "
              f"({metrics['judge_failures']} judge failures)")
    print(f"Cases in failures.jsonl: {summary['failure_case_count']}")
    for check in summary["thresholds"]:
        status = "ok" if check["passed"] else "BELOW"
        print(f"  threshold {check['metric']} >= {check['minimum']}: "
              f"{_fmt(check['actual'])} {status}")
    if summary["errors"]["exceeded"]:
        print(f"  case errors {summary['errors']['count']} exceed "
              f"--max-case-errors {summary['errors']['max_case_errors']}")
    print(f"Report directory: {run.report_dir}")
    print(f"\nResult: {'PASS' if run.passed else 'FAIL'}")


def main(argv: Optional[Sequence[str]] = None, **runner_kwargs: Any) -> int:
    """CLI entry point. runner_kwargs lets tests inject fake dependencies."""
    args = build_parser().parse_args(argv)
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(errors="replace")
    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING,
        format="%(levelname)s %(name)s: %(message)s")

    try:
        config = config_from_args(args)
    except ValidationError as error:
        print(f"Invalid configuration:\n{error}")
        return EXIT_SETUP_ERROR

    print(f"Dataset: {args.dataset}")
    try:
        run = run_evaluation(args.dataset, config, **runner_kwargs)
    except (EvaluationDatasetError, EvaluationSetupError, FileExistsError) as error:
        print(f"Error: {error}")
        print("\nResult: FAIL (the evaluation could not run)")
        return EXIT_SETUP_ERROR

    print_summary(run)
    return EXIT_PASS if run.passed else EXIT_FAIL


if __name__ == "__main__":
    sys.exit(main())
