"""Typed schema for one case in the golden evaluation dataset."""

from typing import Any, Literal, Optional

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    StrictBool,
    field_validator,
    model_validator,
)

Difficulty = Literal["beginner", "intermediate", "advanced"]
DIFFICULTIES: tuple[str, ...] = ("beginner", "intermediate", "advanced")

_LIST_FIELDS = (
    "expected_keywords",
    "expected_source_ids",
    "expected_source_titles",
    "tags",
)


class EvaluationCase(BaseModel):
    """One question with the answer and sources a good response should have.

    expected_source_ids holds retrieval identifiers (this project uses chunk
    ids such as ``speed_schedule_1_p1_c2``). Any one of them supporting the
    answer is acceptable. Unanswerable cases use an empty list.
    """

    # Unknown keys are rejected so a typo like "expected_keyword" is caught.
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    id: str
    question: str
    expected_answer: str
    expected_keywords: list[str]
    expected_source_ids: list[str]
    category: str
    difficulty: Difficulty
    answerable: StrictBool
    expected_source_titles: list[str] = Field(default_factory=list)
    notes: str = ""
    tags: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator(*_LIST_FIELDS, mode="before")
    @classmethod
    def _none_becomes_empty_list(cls, value: Any) -> Any:
        return [] if value is None else value

    @field_validator("notes", mode="before")
    @classmethod
    def _none_becomes_empty_notes(cls, value: Any) -> Any:
        return "" if value is None else value

    @field_validator("metadata", mode="before")
    @classmethod
    def _none_becomes_empty_metadata(cls, value: Any) -> Any:
        return {} if value is None else value

    @field_validator("id", "question", "expected_answer", "category")
    @classmethod
    def _not_empty(cls, value: str) -> str:
        if not value:
            raise ValueError("must not be empty")
        return value

    @field_validator(*_LIST_FIELDS)
    @classmethod
    def _items_not_empty(cls, values: list[str]) -> list[str]:
        cleaned = [item.strip() for item in values]
        if any(not item for item in cleaned):
            raise ValueError("list items must not be empty")
        return cleaned

    @model_validator(mode="after")
    def _answerable_needs_a_source(self) -> "EvaluationCase":
        if self.answerable and not self.expected_source_ids:
            raise ValueError(
                "answerable cases need at least one expected_source_ids entry"
            )
        return self
