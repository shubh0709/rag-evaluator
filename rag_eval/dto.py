"""Strict input contracts for data entering the evaluator."""

from __future__ import annotations

import re
from typing import Any, Literal

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    StrictBool,
    StrictInt,
    StrictStr,
    ValidationError,
    field_validator,
)

from .core import RagResponse, RetrievedItem, TestCase


class InputModel(BaseModel):
    """Represent input model data."""

    model_config = ConfigDict(extra="ignore")


def validation_issue(exc: ValidationError) -> str:
    """Describe a bad field without echoing potentially sensitive input values."""
    error = exc.errors()[0]
    location = ".".join(str(part) for part in error["loc"]) or "input"
    return f"{location}: {error['msg']}"


class RetrievedItemInput(InputModel):
    """Represent retrieved item input data."""

    id: StrictStr = Field(min_length=1)
    title: StrictStr = ""
    text: StrictStr = ""
    source_url: StrictStr = ""
    document_id: StrictStr = ""
    chunk_id: StrictStr = ""

    def to_domain(self) -> RetrievedItem:
        """Convert validated input into a domain object."""
        return RetrievedItem(**self.model_dump())


class CaseInput(InputModel):
    """Represent case input data."""

    id: StrictStr
    name: StrictStr
    question: StrictStr
    relevant_ids: tuple[StrictStr, ...]
    expected_answer: StrictStr
    category: StrictStr
    user_id: StrictStr = "employee"
    notes: StrictStr = ""
    supporting_evidence: StrictStr = ""
    relevance_grades: dict[StrictStr, StrictInt] = Field(default_factory=dict)
    context_relevance: dict[StrictStr, StrictBool] = Field(default_factory=dict)
    reference_points: tuple[StrictStr, ...] = ()
    point_support: dict[StrictStr, tuple[StrictStr, ...]] = Field(default_factory=dict)
    exact_match_references: tuple[StrictStr, ...] = ()
    answerable: StrictBool | None = None
    trap_prior_answer: StrictStr = ""
    trap_context_answer: StrictStr = ""

    def to_domain(self) -> TestCase:
        """Convert validated input into a domain object."""
        return TestCase(**self.model_dump())


class ResponseInput(InputModel):
    """Represent response input data."""

    answer: StrictStr
    retrieved: tuple[RetrievedItemInput, ...]
    citations: tuple[StrictStr, ...] = ()

    def to_domain(self, *, response_keys: list[str] | None = None) -> RagResponse:
        """Convert validated input into a domain object."""
        response = RagResponse(
            self.answer,
            tuple(item.to_domain() for item in self.retrieved),
            self.citations,
            raw={"response_keys": response_keys} if response_keys is not None else {},
        )
        response.validate()
        return response


class BotRetrievedItemInput(InputModel):
    """Represent bot retrieved item input data."""

    id: StrictStr = Field(min_length=1)
    title: StrictStr
    text: StrictStr
    source_url: StrictStr
    document_id: StrictStr = ""
    chunk_id: StrictStr = ""

    def to_domain(self) -> RetrievedItem:
        """Convert validated input into a domain object."""
        return RetrievedItem(**self.model_dump())


class BotResponseInput(InputModel):
    """Represent bot response input data."""

    answer: StrictStr
    retrieved: tuple[BotRetrievedItemInput, ...]
    citations: tuple[StrictStr, ...]

    def to_domain(self, *, response_keys: list[str]) -> RagResponse:
        """Convert validated input into a domain object."""
        response = RagResponse(
            self.answer,
            tuple(item.to_domain() for item in self.retrieved),
            self.citations,
            raw={"response_keys": response_keys},
        )
        response.validate()
        return response


class CasePackInput(InputModel):
    """Represent case pack input data."""

    schema_version: Literal[1]
    cases: list[CaseInput]
    responses: dict[StrictStr, ResponseInput]


class JudgeItemInput(InputModel):
    """Represent judge item input data."""

    text: StrictStr = Field(min_length=1)
    verdict: StrictStr
    explanation: StrictStr = Field(min_length=1)
    evidence_ids: list[StrictStr]
    supporting_evidence: StrictStr
    citation_id: StrictStr | None = None

    @field_validator("text", "explanation")
    @classmethod
    def nonblank(cls, value: str) -> str:
        """Reject blank text values."""
        if not value.strip():
            raise ValueError("must not be blank")
        return value


class JudgeResponseInput(InputModel):
    """Represent judge response input data."""

    model: StrictStr
    model_version: StrictStr
    prompt_version: StrictStr
    rubric_version: StrictStr
    items: list[JudgeItemInput] = Field(min_length=1)
    generated_questions: list[StrictStr] | None = None
    question_embedding: list[float] | None = None
    generated_question_embeddings: list[list[float]] | None = None
    resolved_model_version: StrictStr | None = None
    response_id: StrictStr | None = None


class SavedRunInput(InputModel):
    """Represent saved run input data."""

    schema_version: Literal[2]
    id: StrictStr
    results: list[dict[str, Any]]
    summary: dict[str, Any]

    @field_validator("id")
    @classmethod
    def safe_run_id(cls, value: str) -> str:
        """Validate a run identifier for storage."""
        if re.fullmatch(r"run-[A-Za-z0-9-]+", value) is None:
            raise ValueError("Invalid run ID")
        return value
