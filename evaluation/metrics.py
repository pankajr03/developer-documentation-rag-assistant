"""Deterministic Stage 10 metrics. Nothing here calls an LLM or the vector store.

These metrics are cheap and repeatable, but limited: keyword coverage and
citation checks show whether an answer mentions the right things and points at
retrieved chunks. They do not prove the answer is factually correct.
"""

import re
from statistics import mean
from typing import Any, Iterable, Mapping, Optional, Sequence

from evaluation.config import EvaluationThresholds
from evaluation.results import (
    CaseResult,
    GroundingCheck,
    MetricSummary,
    ReturnedCitation,
    ThresholdCheck,
)

HIT_AT_K: tuple[int, ...] = (1, 3, 5)

# Metadata every retrieved chunk should carry. page is legitimately absent for
# TXT and Markdown files, but it is still reported so gaps are visible.
EXPECTED_METADATA_FIELDS: tuple[str, ...] = (
    "chunk_id", "document_id", "source", "page")


# --- Normalisation -----------------------------------------------------------

def normalize_identifier(value: Any) -> str:
    """Casefold and trim an id or title so 'Doc_A ' matches 'doc_a'."""
    return str(value).strip().casefold() if value is not None else ""


def normalize_text(text: str) -> str:
    """Casefold and collapse all whitespace to single spaces."""
    return " ".join(text.casefold().split())


def normalize_phrase(text: str) -> str:
    """Like normalize_text, but also drops punctuation (for fallback phrases)."""
    text = text.replace("’", "'").casefold()
    text = re.sub(r"[^\w\s]", "", text)
    return " ".join(text.split())


# --- Source matching ---------------------------------------------------------

def match_source(
    retrieved: Mapping[str, Any],
    expected_source_ids: Iterable[str],
    expected_source_titles: Iterable[str] = (),
) -> tuple[Optional[str], Optional[str]]:
    """Return (expected value matched, field it matched on), or (None, None).

    Matching is exact after trimming and casefolding, never by substring:
    1. the chunk's ``chunk_id`` against expected_source_ids (this project's
       golden dataset uses chunk ids);
    2. the chunk's ``document_id`` against expected_source_ids (for datasets
       that list whole documents);
    3. only when the chunk has neither stable id, its filename (``source`` or
       ``filename``) against expected_source_titles.
    """
    ids = {normalize_identifier(value): value for value in expected_source_ids}
    for field in ("chunk_id", "document_id"):
        value = normalize_identifier(retrieved.get(field))
        if value and value in ids:
            return ids[value], field

    has_stable_id = any(
        normalize_identifier(retrieved.get(field))
        for field in ("chunk_id", "document_id"))
    if not has_stable_id:
        titles = {normalize_identifier(t): t for t in expected_source_titles}
        title = normalize_identifier(
            retrieved.get("source") or retrieved.get("filename"))
        if title and title in titles:
            return titles[title], "title"

    return None, None


def missing_metadata(retrieved: Mapping[str, Any]) -> list[str]:
    """Names of expected metadata fields that are absent from a chunk."""
    return [
        field for field in EXPECTED_METADATA_FIELDS
        if retrieved.get(field) in (None, "")
    ]


# --- Retrieval ---------------------------------------------------------------

def retrieval_metrics(relevance: Sequence[bool], top_k: int) -> dict[str, Any]:
    """Source hit, Hit@k and reciprocal rank from per-rank relevance flags.

    relevance[i] is True when the chunk at rank i + 1 matched an expected
    source. Hit@k is None when top_k < k, because fewer than k results were
    requested and the metric cannot be measured. If the collection simply held
    fewer than k chunks, every chunk was searched, so Hit@k is still measured.
    """
    first_rank = next(
        (rank for rank, relevant in enumerate(relevance, start=1) if relevant),
        None)
    metrics: dict[str, Any] = {
        "source_hit": first_rank is not None,
        "first_relevant_rank": first_rank,
        "reciprocal_rank": 1 / first_rank if first_rank else 0.0,
    }
    for k in HIT_AT_K:
        metrics[f"hit_at_{k}"] = (
            None if top_k < k else first_rank is not None and first_rank <= k)
    return metrics


# --- Answer ------------------------------------------------------------------

def keyword_coverage(
    answer: str, keywords: Iterable[str],
) -> tuple[Optional[float], list[str], list[str]]:
    """Return (coverage, matched, missing) for expected keywords in an answer.

    Matching is case-insensitive, ignores extra whitespace, and requires whole
    words or phrases: "with" does not match inside "without". Coverage is None
    when there are no expected keywords.
    """
    unique = list(dict.fromkeys(
        keyword.strip() for keyword in keywords if keyword.strip()))
    if not unique:
        return None, [], []

    text = normalize_text(answer)
    matched, missing = [], []
    for keyword in unique:
        pattern = rf"(?<!\w){re.escape(normalize_text(keyword))}(?!\w)"
        (matched if re.search(pattern, text) else missing).append(keyword)
    return len(matched) / len(unique), matched, missing


def is_abstention(answer: str, fallback_phrases: Iterable[str]) -> bool:
    """True when the answer contains an accepted fallback phrase.

    Case, punctuation and spacing are ignored.
    """
    text = normalize_phrase(answer)
    return any(
        phrase and phrase in text
        for phrase in (normalize_phrase(p) for p in fallback_phrases))


def score_citations(
    citations: Sequence[Mapping[str, Any]],
    unverified_numbers: Sequence[int],
    retrieved: Sequence[Mapping[str, Any]],
    expected_source_ids: Sequence[str],
    expected_source_titles: Sequence[str],
    answerable: bool,
) -> dict[str, Any]:
    """Check the answer's structured citations against the retrieved chunks.

    A citation counts as correct only when its chunk id is one of the chunks
    retrieved for this question. Citations to source numbers that were never
    retrieved, or without a chunk id, are unverifiable. The answer text itself
    is never searched for source names, so mentioning a document earns nothing.
    """
    retrieved_ids = {
        normalize_identifier(chunk.get("chunk_id")) for chunk in retrieved
    } - {""}

    returned: list[ReturnedCitation] = []
    unverifiable: list[str] = []
    for citation in citations:
        chunk_id = citation.get("chunk_id")
        in_retrieved = normalize_identifier(chunk_id) in retrieved_ids
        if not chunk_id:
            unverifiable.append(
                f"Source {citation.get('number')}: no chunk_id metadata")
        elif not in_retrieved:
            unverifiable.append(
                f"Source {citation.get('number')}: not a retrieved chunk")
        matched, _ = match_source(
            citation, expected_source_ids, expected_source_titles)
        returned.append(ReturnedCitation(
            number=citation.get("number"),
            chunk_id=chunk_id,
            document_id=citation.get("document_id"),
            filename=citation.get("filename"),
            page=citation.get("page"),
            in_retrieved=in_retrieved,
            matched_expected=matched,
        ))
    for number in unverified_numbers:
        unverifiable.append(f"Source {number}: not a retrieved source number")
        returned.append(ReturnedCitation(number=number, in_retrieved=False))

    count = len(returned)
    matching = sum(citation.in_retrieved for citation in returned)
    expected_hit = None
    if answerable:
        expected_hit = any(
            c.in_retrieved and c.matched_expected for c in returned)

    return {
        "returned_citations": returned,
        "citation_count": count,
        "citations_matching_retrieved": matching,
        "citation_precision": matching / count if count else None,
        "expected_source_citation_hit": expected_hit,
        "unverifiable_citations": unverifiable,
    }


def grounding_check(
    retrieved_count: int,
    citations: Sequence[ReturnedCitation],
    abstained: bool,
    answerable: bool,
    source_hit: Optional[bool],
) -> GroundingCheck:
    """Conservative citation-level grounding signals for one answer.

    "Useful evidence" means an expected source was retrieved, which is only
    possible for answerable cases. ``grounded`` is None for abstentions, and
    otherwise True only when the answer cites retrieved chunks and nothing else.
    """
    has_context = retrieved_count > 0
    has_citations = bool(citations)
    refer_to_retrieved = (
        all(c.in_retrieved for c in citations) if has_citations else None)
    useful_evidence = answerable and bool(source_hit)
    return GroundingCheck(
        has_retrieved_context=has_context,
        has_citations=has_citations,
        citations_refer_to_retrieved=refer_to_retrieved,
        used_fallback=abstained,
        answered_without_useful_evidence=not abstained and not useful_evidence,
        grounded=None if abstained else (
            has_context and has_citations and bool(refer_to_retrieved)),
    )


def abstention_outcome(
    answerable: bool, abstained: bool, citation_count: int,
) -> tuple[Optional[bool], Optional[bool]]:
    """Return (correct_abstention, incorrect_abstention).

    Unanswerable: correct when the assistant used the fallback and cited
    nothing, i.e. it did not present retrieved chunks as evidence.
    Answerable: abstaining is an incorrect abstention.
    """
    if answerable:
        return None, abstained
    return abstained and citation_count == 0, None


def failure_reasons(result: CaseResult, low_keyword_coverage: float) -> list[str]:
    """Why a case belongs in failures.jsonl (empty list when it looks fine)."""
    reasons = []
    if result.error:
        reasons.append("error")
    if result.answerable and result.source_hit is False:
        reasons.append("missed_expected_sources")
    if (result.keyword_coverage is not None
            and result.keyword_coverage < low_keyword_coverage):
        reasons.append("low_keyword_coverage")
    if result.unverifiable_citations:
        reasons.append("incorrect_citations")
    if result.correct_abstention is False:
        reasons.append("failed_abstention")
    if result.incorrect_abstention:
        reasons.append("incorrect_abstention")
    return reasons


# --- Aggregation -------------------------------------------------------------

def _mean(values: Iterable[Optional[float]]) -> Optional[float]:
    present = [float(value) for value in values if value is not None]
    return round(mean(present), 4) if present else None


def summarize_results(results: Sequence[CaseResult]) -> MetricSummary:
    """Aggregate per-case results. Each metric averages only the cases where it
    was measured, so a metric with no measurements is None, not zero.

    Citation precision is micro-averaged: correct citations / all citations.
    """
    answerable = [r for r in results if r.answerable]
    unanswerable = [r for r in results if not r.answerable]

    total_citations = sum(r.citation_count or 0 for r in results)
    matching_citations = sum(r.citations_matching_retrieved or 0 for r in results)
    abstention_checked = [
        r.incorrect_abstention for r in answerable
        if r.incorrect_abstention is not None]
    judged = [r.judge for r in results if r.judge is not None]

    return MetricSummary(
        total_cases=len(results),
        completed_cases=sum(r.error is None for r in results),
        failed_cases=sum(r.error is not None for r in results),
        answerable_cases=len(answerable),
        unanswerable_cases=len(unanswerable),
        source_hit_rate=_mean(r.source_hit for r in answerable),
        hit_at_1=_mean(r.hit_at_1 for r in answerable),
        hit_at_3=_mean(r.hit_at_3 for r in answerable),
        hit_at_5=_mean(r.hit_at_5 for r in answerable),
        mrr=_mean(r.reciprocal_rank for r in answerable),
        average_keyword_coverage=_mean(r.keyword_coverage for r in results),
        citation_precision=(
            round(matching_citations / total_citations, 4)
            if total_citations else None),
        expected_source_citation_hit_rate=_mean(
            r.expected_source_citation_hit for r in answerable),
        grounded_rate=_mean(
            r.grounding.grounded for r in results if r.grounding),
        correct_abstention_rate=_mean(
            r.correct_abstention for r in unanswerable),
        incorrect_abstention_count=(
            sum(abstention_checked) if abstention_checked else None),
        average_retrieval_latency_ms=_mean(
            r.retrieval_latency_ms for r in results),
        average_total_latency_ms=_mean(
            r.total_latency_ms for r in results if r.error is None),
        judge_cases=len(judged),
        judge_failures=sum(r.judge_error is not None for r in results),
        judge_average_correctness=_mean(j.correctness for j in judged),
        judge_average_grounding=_mean(j.grounding for j in judged),
        judge_average_relevance=_mean(j.relevance for j in judged),
        judge_unsupported_claims_rate=_mean(
            j.unsupported_claims for j in judged),
    )


def group_summaries(
    results: Sequence[CaseResult], key: str,
) -> dict[str, MetricSummary]:
    """Summaries per value of a case field, e.g. 'category' or 'difficulty'."""
    groups: dict[str, list[CaseResult]] = {}
    for result in results:
        groups.setdefault(str(getattr(result, key)), []).append(result)
    return {name: summarize_results(group)
            for name, group in sorted(groups.items())}


THRESHOLD_METRICS: dict[str, str] = {
    "min_source_hit_rate": "source_hit_rate",
    "min_mrr": "mrr",
    "min_keyword_coverage": "average_keyword_coverage",
    "min_correct_abstention_rate": "correct_abstention_rate",
}


def check_thresholds(
    summary: MetricSummary, thresholds: EvaluationThresholds,
) -> list[ThresholdCheck]:
    """Compare configured minimums with the summary.

    A metric that could not be measured (None) fails its threshold, because
    the gate cannot be confirmed.
    """
    checks = []
    for threshold_name, metric in THRESHOLD_METRICS.items():
        minimum = getattr(thresholds, threshold_name)
        if minimum is None:
            continue
        actual = getattr(summary, metric)
        checks.append(ThresholdCheck(
            metric=metric,
            minimum=minimum,
            actual=actual,
            passed=actual is not None and actual >= minimum,
        ))
    return checks
