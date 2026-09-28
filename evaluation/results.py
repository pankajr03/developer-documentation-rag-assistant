"""Typed per-case and summary results for a Stage 10 evaluation run.

Metrics that cannot validly be measured for a case are None (null in JSON),
never a misleading zero. For example, keyword coverage is None when a case has
no expected keywords, and retrieval hit metrics are None for unanswerable cases.
"""

from typing import Optional

from pydantic import BaseModel, ConfigDict, Field


class RetrievedSource(BaseModel):
    """One retrieved chunk, as scored against the expected sources."""

    rank: int
    chunk_id: Optional[str] = None
    document_id: Optional[str] = None
    source: Optional[str] = None
    page: Optional[int] = None
    section: Optional[str] = None
    distance: Optional[float] = None
    # The expected source id (or title) this chunk matched, and on which field.
    matched_expected: Optional[str] = None
    matched_on: Optional[str] = None
    missing_metadata: list[str] = Field(default_factory=list)


class ReturnedCitation(BaseModel):
    """One [Source N] citation in the generated answer."""

    number: int
    chunk_id: Optional[str] = None
    document_id: Optional[str] = None
    filename: Optional[str] = None
    page: Optional[int] = None
    # True when the citation points at a chunk that was actually retrieved.
    in_retrieved: bool
    matched_expected: Optional[str] = None


class GroundingCheck(BaseModel):
    """Conservative, citation-level grounding signals.

    These show whether the answer is tied to retrieved evidence. They do not
    prove the answer is factually correct.
    """

    has_retrieved_context: bool
    has_citations: bool
    citations_refer_to_retrieved: Optional[bool] = None
    used_fallback: bool
    answered_without_useful_evidence: bool
    grounded: Optional[bool] = None


class JudgeScores(BaseModel):
    """Model-based estimates from the optional LLM judge (0-4 scale)."""

    correctness: int
    grounding: int
    relevance: int
    unsupported_claims: bool
    reason: str


class CaseResult(BaseModel):
    """Everything measured for one evaluation case."""

    model_config = ConfigDict(extra="forbid")

    case_id: str
    question: str
    category: str
    difficulty: str
    answerable: bool
    tags: list[str] = Field(default_factory=list)
    expected_source_ids: list[str] = Field(default_factory=list)

    retrieved_sources: list[RetrievedSource] = Field(default_factory=list)
    generated_answer: Optional[str] = None
    response_id: Optional[str] = None
    returned_citations: list[ReturnedCitation] = Field(default_factory=list)

    # Retrieval (answerable cases only).
    source_hit: Optional[bool] = None
    hit_at_1: Optional[bool] = None
    hit_at_3: Optional[bool] = None
    hit_at_5: Optional[bool] = None
    first_relevant_rank: Optional[int] = None
    reciprocal_rank: Optional[float] = None

    # Answer (only when an answer was generated).
    keyword_coverage: Optional[float] = None
    matched_keywords: list[str] = Field(default_factory=list)
    missing_keywords: list[str] = Field(default_factory=list)
    citation_count: Optional[int] = None
    citations_matching_retrieved: Optional[int] = None
    citation_precision: Optional[float] = None
    expected_source_citation_hit: Optional[bool] = None
    unverifiable_citations: list[str] = Field(default_factory=list)
    grounding: Optional[GroundingCheck] = None
    abstained: Optional[bool] = None
    correct_abstention: Optional[bool] = None
    incorrect_abstention: Optional[bool] = None

    # Optional LLM judge. Never changes any deterministic metric above.
    judge: Optional[JudgeScores] = None
    judge_error: Optional[str] = None

    retrieval_latency_ms: Optional[float] = None
    generation_latency_ms: Optional[float] = None
    total_latency_ms: Optional[float] = None

    error: Optional[str] = None
    error_stage: Optional[str] = None
    failure_reasons: list[str] = Field(default_factory=list)


class MetricSummary(BaseModel):
    """Aggregate metrics over a group of case results."""

    total_cases: int
    completed_cases: int
    failed_cases: int
    answerable_cases: int
    unanswerable_cases: int

    source_hit_rate: Optional[float] = None
    hit_at_1: Optional[float] = None
    hit_at_3: Optional[float] = None
    hit_at_5: Optional[float] = None
    mrr: Optional[float] = None

    average_keyword_coverage: Optional[float] = None
    citation_precision: Optional[float] = None
    expected_source_citation_hit_rate: Optional[float] = None
    grounded_rate: Optional[float] = None
    correct_abstention_rate: Optional[float] = None
    incorrect_abstention_count: Optional[int] = None

    average_retrieval_latency_ms: Optional[float] = None
    average_total_latency_ms: Optional[float] = None

    judge_cases: int = 0
    judge_failures: int = 0
    judge_average_correctness: Optional[float] = None
    judge_average_grounding: Optional[float] = None
    judge_average_relevance: Optional[float] = None
    judge_unsupported_claims_rate: Optional[float] = None


class ThresholdCheck(BaseModel):
    """One configured quality gate and whether the run met it."""

    metric: str
    minimum: float
    actual: Optional[float] = None
    passed: bool
