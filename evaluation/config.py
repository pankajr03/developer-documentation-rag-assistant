"""Typed configuration for a Stage 10 evaluation run."""

from pathlib import Path
from typing import Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from evaluation.schemas import DIFFICULTIES

DEFAULT_OUTPUT_DIRECTORY = str(
    Path(__file__).resolve().parent.parent / "reports" / "evaluation")

# Answers that count as a safe abstention. They are compared after
# normalisation (case, punctuation and spacing are ignored), so small wording
# changes in punctuation or capitalisation do not break abstention scoring.
# The first phrase is the app's own rag.generator.FALLBACK_MESSAGE.
DEFAULT_FALLBACK_PHRASES: tuple[str, ...] = (
    "I could not find this information in the uploaded documents.",
    "I couldn't find that information in the indexed documentation.",
)

# Suggested starting points only, applied with --recommended-thresholds.
# They are not universal truths: tune them for your own documents.
RECOMMENDED_THRESHOLDS: dict[str, float] = {
    "min_source_hit_rate": 0.80,
    "min_mrr": 0.60,
    "min_keyword_coverage": 0.60,
    "min_correct_abstention_rate": 0.80,
}

_Rate = Optional[float]


class EvaluationThresholds(BaseModel):
    """Optional quality gates. A threshold left as None is not checked."""

    model_config = ConfigDict(extra="forbid")

    min_source_hit_rate: _Rate = Field(default=None, ge=0, le=1)
    min_mrr: _Rate = Field(default=None, ge=0, le=1)
    min_keyword_coverage: _Rate = Field(default=None, ge=0, le=1)
    min_correct_abstention_rate: _Rate = Field(default=None, ge=0, le=1)


class EvaluationConfig(BaseModel):
    """What to evaluate and how.

    Filters are applied in this order: case_ids, category, difficulty, then
    max_cases. max_case_errors is the case-error policy: the run is marked
    FAIL when more than this many cases raise an error.
    """

    model_config = ConfigDict(extra="forbid")

    top_k: int = Field(default=5, gt=0)
    max_cases: Optional[int] = Field(default=None, gt=0)
    category: Optional[str] = None
    difficulty: Optional[str] = None
    case_ids: list[str] = Field(default_factory=list)
    generate_answers: bool = True
    use_llm_judge: bool = False
    judge_model: Optional[str] = None
    output_directory: str = DEFAULT_OUTPUT_DIRECTORY
    run_directory: Optional[str] = None
    thresholds: EvaluationThresholds = Field(
        default_factory=EvaluationThresholds)
    max_case_errors: int = Field(default=0, ge=0)
    low_keyword_coverage: float = Field(default=0.5, ge=0, le=1)
    fallback_phrases: tuple[str, ...] = DEFAULT_FALLBACK_PHRASES

    @field_validator("category", "difficulty", mode="before")
    @classmethod
    def _blank_becomes_none(cls, value: Optional[str]) -> Optional[str]:
        if isinstance(value, str):
            value = value.strip()
        return value or None

    @field_validator("difficulty")
    @classmethod
    def _known_difficulty(cls, value: Optional[str]) -> Optional[str]:
        if value is not None and value not in DIFFICULTIES:
            raise ValueError(
                f"difficulty must be one of {', '.join(DIFFICULTIES)}")
        return value

    @field_validator("case_ids")
    @classmethod
    def _clean_case_ids(cls, values: list[str]) -> list[str]:
        cleaned = [value.strip() for value in values]
        if any(not value for value in cleaned):
            raise ValueError("case ids must not be empty")
        return list(dict.fromkeys(cleaned))

    @field_validator("fallback_phrases")
    @classmethod
    def _needs_a_fallback_phrase(cls, values: tuple[str, ...]) -> tuple[str, ...]:
        if not any(value.strip() for value in values):
            raise ValueError("at least one fallback phrase is required")
        return values

    @model_validator(mode="after")
    def _judge_needs_answers(self) -> "EvaluationConfig":
        if self.use_llm_judge and not self.generate_answers:
            raise ValueError(
                "use_llm_judge needs generated answers; "
                "do not combine it with skipping generation")
        return self
