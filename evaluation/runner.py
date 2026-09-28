"""Run golden evaluation cases through the real RAG pipeline and score them.

Only the question text (and top_k) is sent to the retriever and generator.
Golden labels such as expected_answer, expected_keywords, expected_source_ids,
answerable and notes are used for scoring afterwards, never as model input.
The one exception is the optional LLM judge, which runs after the answer is
final and needs the reference answer to grade it.

The run reads ChromaDB but never writes to it, never re-embeds documents,
never edits the golden dataset, and never touches the feedback database.
"""

import logging
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Optional, Sequence, Union

from evaluation.config import EvaluationConfig
from evaluation.dataset import find_missing_source_ids, load_evaluation_dataset
from evaluation.metrics import (
    abstention_outcome,
    check_thresholds,
    failure_reasons,
    grounding_check,
    group_summaries,
    is_abstention,
    keyword_coverage,
    match_source,
    missing_metadata,
    retrieval_metrics,
    score_citations,
    summarize_results,
)
from evaluation.reporter import (
    create_run_directory,
    environment_info,
    write_reports,
)
from evaluation.results import CaseResult, RetrievedSource
from evaluation.schemas import EvaluationCase

logger = logging.getLogger(__name__)

Retriever = Callable[..., list[dict[str, Any]]]
Generator = Callable[..., dict[str, Any]]
Judge = Callable[..., Any]

JUDGE_DISCLAIMER = (
    "LLM judge scores are model-based estimates, not objective truth. "
    "They never change the deterministic metrics.")
MAX_ERROR_LENGTH = 500


class EvaluationSetupError(Exception):
    """The run cannot start: bad configuration, no cases, or no vector store."""


@dataclass
class EvaluationRun:
    """Everything a run produced."""

    summary: dict[str, Any]
    results: list[CaseResult]
    report_dir: Optional[Path] = None
    report_files: dict[str, Path] = field(default_factory=dict)

    @property
    def passed(self) -> bool:
        return bool(self.summary["passed"])


def select_cases(
    cases: Sequence[EvaluationCase], config: EvaluationConfig,
) -> list[EvaluationCase]:
    """Apply case_ids, category, difficulty and max_cases filters, in order."""
    if not cases:
        raise EvaluationSetupError("The evaluation dataset contains no cases.")

    selected = list(cases)
    if config.case_ids:
        by_id = {case.id: case for case in selected}
        unknown = [case_id for case_id in config.case_ids if case_id not in by_id]
        if unknown:
            raise EvaluationSetupError(
                "Unknown case id(s): " + ", ".join(unknown))
        wanted = set(config.case_ids)
        selected = [case for case in selected if case.id in wanted]
    if config.category:
        selected = [c for c in selected if c.category == config.category]
    if config.difficulty:
        selected = [c for c in selected if c.difficulty == config.difficulty]
    if not selected:
        filters = {
            "case_ids": config.case_ids or None,
            "category": config.category,
            "difficulty": config.difficulty,
        }
        applied = ", ".join(f"{k}={v}" for k, v in filters.items() if v)
        raise EvaluationSetupError(f"No cases match the filters ({applied}).")
    if config.max_cases is not None:
        selected = selected[:config.max_cases]
    return selected


def check_vector_store() -> dict[str, Any]:
    """Confirm the ChromaDB collection exists and has records, read-only.

    Unlike the app, this never creates a missing collection or directory.
    """
    import chromadb

    from rag.vector_store import CHROMA_PATH, COLLECTION_NAME

    if not CHROMA_PATH.is_dir():
        raise EvaluationSetupError(
            f"The vector database was not found at {CHROMA_PATH}. "
            "Index documents in the app first.")
    try:
        client = chromadb.PersistentClient(path=str(CHROMA_PATH))
        collection = client.get_collection(name=COLLECTION_NAME)
        record_count = collection.count()
        ids = collection.get(include=[])["ids"]
    except Exception as error:
        raise EvaluationSetupError(
            f"The {COLLECTION_NAME} ChromaDB collection is unavailable: "
            f"{type(error).__name__}") from error
    if record_count == 0:
        raise EvaluationSetupError(
            f"The {COLLECTION_NAME} collection is empty. "
            "Index documents in the app first.")
    return {"collection": COLLECTION_NAME, "record_count": record_count,
            "ids": ids}


def _error_text(error: Exception) -> str:
    return f"{type(error).__name__}: {error}"[:MAX_ERROR_LENGTH]


def _elapsed_ms(start: float, clock: Callable[[], float]) -> float:
    return round((clock() - start) * 1000, 1)


def _citations_from(output: dict[str, Any]) -> tuple[list[dict], list[int]]:
    """Structured citations from a generator result.

    Falls back to reading [Source N] markers against the returned sources for
    results that predate the structured "citations" field.
    """
    if "citations" in output:
        return (list(output["citations"]),
                list(output.get("unverified_citation_numbers", [])))
    from rag.generator import build_citations

    return build_citations(output.get("answer", ""), output.get("sources", [])), []


def evaluate_case(
    case: EvaluationCase,
    config: EvaluationConfig,
    retrieve: Retriever,
    generate: Optional[Generator] = None,
    judge: Optional[Judge] = None,
    clock: Callable[[], float] = time.perf_counter,
) -> CaseResult:
    """Run one case and score it. Errors are recorded, never raised."""
    result = CaseResult(
        case_id=case.id,
        question=case.question,
        category=case.category,
        difficulty=case.difficulty,
        answerable=case.answerable,
        tags=case.tags,
        expected_source_ids=case.expected_source_ids,
    )
    start = clock()

    try:
        chunks = list(retrieve(case.question, top_k=config.top_k))
    except Exception as error:
        result.error, result.error_stage = _error_text(error), "retrieval"
        result.retrieval_latency_ms = _elapsed_ms(start, clock)
        result.total_latency_ms = result.retrieval_latency_ms
        result.failure_reasons = failure_reasons(
            result, config.low_keyword_coverage)
        return result
    result.retrieval_latency_ms = _elapsed_ms(start, clock)

    relevance = []
    for rank, chunk in enumerate(chunks, start=1):
        matched, matched_on = match_source(
            chunk, case.expected_source_ids, case.expected_source_titles)
        relevance.append(matched is not None)
        result.retrieved_sources.append(RetrievedSource(
            rank=rank,
            chunk_id=chunk.get("chunk_id"),
            document_id=chunk.get("document_id"),
            source=chunk.get("source"),
            page=chunk.get("page"),
            section=chunk.get("section"),
            distance=chunk.get("distance"),
            matched_expected=matched,
            matched_on=matched_on,
            missing_metadata=missing_metadata(chunk),
        ))
    # Unanswerable cases have no expected sources, so a "miss" is not a
    # retrieval failure; their retrieved chunks are kept for inspection only.
    if case.answerable:
        for name, value in retrieval_metrics(relevance, config.top_k).items():
            setattr(result, name, value)

    output: dict[str, Any] = {}
    if generate is not None:
        generation_start = clock()
        try:
            output = generate(case.question, chunks, top_k=config.top_k)
        except Exception as error:
            result.error, result.error_stage = _error_text(error), "generation"
        else:
            _score_answer(result, case, config, chunks, output)
        result.generation_latency_ms = _elapsed_ms(generation_start, clock)
    result.total_latency_ms = _elapsed_ms(start, clock)

    if judge is not None and result.generated_answer is not None:
        try:
            result.judge = judge(
                case.question, case.expected_answer,
                result.generated_answer, output.get("sources", []))
        except Exception as error:
            result.judge_error = _error_text(error)

    result.failure_reasons = failure_reasons(result, config.low_keyword_coverage)
    return result


def _score_answer(
    result: CaseResult,
    case: EvaluationCase,
    config: EvaluationConfig,
    chunks: list[dict[str, Any]],
    output: dict[str, Any],
) -> None:
    """Fill in the answer, citation, grounding and abstention metrics."""
    answer = output.get("answer") or ""
    result.generated_answer = answer
    result.response_id = output.get("response_id")

    coverage, matched, missing = keyword_coverage(answer, case.expected_keywords)
    result.keyword_coverage = coverage
    result.matched_keywords, result.missing_keywords = matched, missing

    citations, unverified = _citations_from(output)
    scored = score_citations(
        citations, unverified, chunks, case.expected_source_ids,
        case.expected_source_titles, case.answerable)
    for name, value in scored.items():
        setattr(result, name, value)

    abstained = is_abstention(answer, config.fallback_phrases)
    result.abstained = abstained
    result.correct_abstention, result.incorrect_abstention = abstention_outcome(
        case.answerable, abstained, result.citation_count)
    result.grounding = grounding_check(
        len(chunks), result.returned_citations, abstained,
        case.answerable, result.source_hit)


def _default_models(config: EvaluationConfig, judge: Optional[Judge]) -> dict:
    from rag import embeddings, generator
    from rag.vector_store import COLLECTION_NAME

    return {
        "embedding_model": embeddings.MODEL_NAME,
        "generation_model": generator.MODEL_NAME if config.generate_answers else None,
        "judge_model": getattr(judge, "model", None) if judge else None,
        "judge_temperature": getattr(judge, "temperature", None) if judge else None,
        "collection": COLLECTION_NAME,
    }


def run_evaluation(
    dataset_path: Union[str, Path],
    config: Optional[EvaluationConfig] = None,
    *,
    retrieve: Optional[Retriever] = None,
    generate: Optional[Generator] = None,
    judge: Optional[Judge] = None,
    vector_store_check: Optional[Callable[[], dict[str, Any]]] = None,
    clock: Callable[[], float] = time.perf_counter,
    now: Optional[datetime] = None,
) -> EvaluationRun:
    """Evaluate the selected golden cases and write the report files.

    With no retrieve/generate arguments the application's real
    rag.retriever.retrieve_chunks and rag.generator.generate_answer are used,
    after a read-only check that the ChromaDB collection is available. Tests
    pass fakes instead, so no API calls are made.

    Raises EvaluationSetupError (or EvaluationDatasetError) when the run
    cannot start. Errors inside a single case are recorded on that case.
    """
    config = config or EvaluationConfig()
    cases = load_evaluation_dataset(dataset_path)
    selected = select_cases(cases, config)

    if retrieve is None:
        from rag.retriever import retrieve_chunks

        retrieve = retrieve_chunks
        vector_store_check = vector_store_check or check_vector_store
    if config.generate_answers and generate is None:
        from rag.generator import generate_answer

        generate = generate_answer
    if not config.generate_answers:
        generate = None
    if config.use_llm_judge and judge is None:
        from evaluation.judge import DEFAULT_JUDGE_MODEL, LLMJudge

        judge = LLMJudge(model=config.judge_model or DEFAULT_JUDGE_MODEL)
    if not config.use_llm_judge:
        judge = None

    store_info: dict[str, Any] = {}
    if vector_store_check is not None:
        store_info = vector_store_check()

    now = now or datetime.now().astimezone()
    report_dir = create_run_directory(config, now)

    results = []
    for index, case in enumerate(selected, start=1):
        logger.info("Evaluating %s (%d/%d)", case.id, index, len(selected))
        results.append(evaluate_case(case, config, retrieve, generate, judge, clock))

    summary = build_summary(
        dataset_path, len(cases), selected, results, config, judge,
        store_info, now, report_dir)
    report_files = write_reports(report_dir, summary, results)
    return EvaluationRun(summary, results, report_dir, report_files)


def build_summary(
    dataset_path: Union[str, Path],
    dataset_case_count: int,
    selected: Sequence[EvaluationCase],
    results: Sequence[CaseResult],
    config: EvaluationConfig,
    judge: Optional[Judge],
    store_info: dict[str, Any],
    now: datetime,
    report_dir: Optional[Path],
) -> dict[str, Any]:
    """Assemble the summary.json document for a run."""
    metrics = summarize_results(results)
    checks = check_thresholds(metrics, config.thresholds)
    failed_case_ids = [r.case_id for r in results if r.error]
    too_many_errors = len(failed_case_ids) > config.max_case_errors
    models = _default_models(config, judge)

    vector_store: dict[str, Any] = {"collection": models.pop("collection")}
    if store_info:
        vector_store["record_count"] = store_info.get("record_count")
        vector_store["expected_source_ids_missing_from_index"] = (
            find_missing_source_ids(selected, store_info.get("ids", [])))

    return {
        "timestamp": now.isoformat(timespec="seconds"),
        "report_directory": str(report_dir) if report_dir else None,
        "dataset": {
            "path": str(dataset_path),
            "case_count": dataset_case_count,
            "selected_case_count": len(selected),
            "selected_case_ids": [case.id for case in selected],
        },
        "config": config.model_dump(mode="json"),
        "models": models,
        "vector_store": vector_store,
        "metrics": metrics.model_dump(mode="json"),
        "by_category": _dump(group_summaries(results, "category")),
        "by_difficulty": _dump(group_summaries(results, "difficulty")),
        "by_answerability": _dump({
            ("answerable" if key == "True" else "unanswerable"): value
            for key, value in group_summaries(results, "answerable").items()}),
        "errors": {
            "count": len(failed_case_ids),
            "case_ids": failed_case_ids,
            "max_case_errors": config.max_case_errors,
            "exceeded": too_many_errors,
        },
        "failure_case_count": sum(bool(r.failure_reasons) for r in results),
        "thresholds": [check.model_dump(mode="json") for check in checks],
        "passed": not too_many_errors and all(c.passed for c in checks),
        "notes": {"llm_judge": JUDGE_DISCLAIMER} if judge else {},
        "environment": environment_info(),
    }


def _dump(groups: dict[str, Any]) -> dict[str, Any]:
    return {name: summary.model_dump(mode="json")
            for name, summary in groups.items()}
