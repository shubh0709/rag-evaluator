"""Typed records for evaluator calculations, traces, and saved runs."""

from __future__ import annotations

from typing import NotRequired, TypedDict


class Calculation(TypedDict):
    """Represent calculation data."""

    status: str
    score: float | None
    reason: str
    k: int
    numerator: int
    denominator: int
    hits: list[str]
    misses: list[str]
    top_k: list[str]


class PrecisionCalculation(TypedDict):
    """Represent precision calculation data."""

    status: str
    score: float | None
    reason: str
    k: int
    numerator: int
    denominator: int
    relevant_ranks: list[int]
    duplicate_ranks: list[int]
    top_k: list[str]


class ReciprocalRankCalculation(TypedDict):
    """Represent reciprocal rank calculation data."""

    status: str
    score: float | None
    reason: str
    first_relevant_rank: int | None
    first_relevant_id: str | None
    ranked_ids: list[str]


class MrrSummary(TypedDict):
    """Represent mrr summary data."""

    score: float | None
    reciprocal_rank_sum: float
    scored_cases: int
    excluded_cases: int


class NdcgCalculation(TypedDict):
    """Represent ndcg calculation data."""

    status: str
    score: float | None
    reason: str
    k: int
    dcg: float
    ideal_dcg: float
    ranked_grades: list[int]
    ideal_grades: list[int]
    discounted_gains: list[float]
    ideal_discounted_gains: list[float]
    top_k: list[str]
    duplicate_ranks: list[int]


class AveragePrecisionCalculation(TypedDict):
    """Represent average precision calculation data."""

    status: str
    score: float | None
    reason: str
    relevant_ranks: list[int]
    precision_at_relevant_ranks: list[float]
    relevant_total: int
    ranked_ids: list[str]
    duplicate_ranks: list[int]


class ContextPrecisionCalculation(TypedDict):
    """Represent context precision calculation data."""

    status: str
    score: float | None
    reason: str
    relevant_ranks: list[int]
    precision_at_relevant_ranks: list[float]
    relevant_retrieved: int
    ranked_ids: list[str]
    duplicate_ranks: list[int]
    unjudged_ids: list[str]


class ContextRecallCalculation(TypedDict):
    """Represent context recall calculation data."""

    status: str
    score: float | None
    reason: str
    covered_points: list[str]
    missed_points: list[str]
    support_by_point: dict[str, list[str]]
    reference_total: int
    ranked_ids: list[str]


class MeanMetricSummary(TypedDict):
    """Represent mean metric summary data."""

    score: float | None
    score_sum: float
    scored_cases: int
    excluded_cases: int


class ExactMatchCalculation(TypedDict):
    """Represent exact match calculation data."""

    status: str
    score: float | None
    reason: str
    normalization_version: str
    normalized_answer: str
    normalized_references: list[str]
    matched_reference_index: int | None


class TokenF1Calculation(TypedDict):
    """Represent token f1 calculation data."""

    status: str
    score: float | None
    reason: str
    normalization_version: str
    answer_tokens: list[str]
    reference_tokens: list[list[str]]
    matched_reference_index: int | None
    overlap: int
    precision: float
    recall: float


class TraceEvent(TypedDict):
    """Represent trace event data."""

    at: str
    stage: str
    status: str
    detail: str


class ExecutionTrace(TypedDict):
    """Represent execution trace data."""

    request: dict[str, str]
    started_at: str | None
    finished_at: str | None
    recorded_at: str
    events: list[TraceEvent]
    ranked_retrieval_status: str
    generation_context_status: str


class CaseRecord(TypedDict):
    """Represent case record data."""

    id: str
    name: str
    question: str
    relevant_ids: tuple[str, ...]
    expected_answer: str
    category: str
    user_id: str
    notes: str
    supporting_evidence: str
    relevance_grades: NotRequired[dict[str, int]]
    context_relevance: NotRequired[dict[str, bool]]
    reference_points: NotRequired[tuple[str, ...]]
    point_support: NotRequired[dict[str, tuple[str, ...]]]
    exact_match_references: NotRequired[tuple[str, ...]]
    answerable: NotRequired[bool | None]
    trap_prior_answer: NotRequired[str]
    trap_context_answer: NotRequired[str]


class ResponseRecord(TypedDict):
    """Represent response record data."""

    answer: str
    retrieved: list[dict[str, str]] | tuple[dict[str, str], ...]
    citations: list[str] | tuple[str, ...]
    raw: dict[str, object]


class Result(TypedDict):
    """Represent result data."""

    case: CaseRecord
    metric: str
    metric_version: str
    status: str
    error: str | None
    response: ResponseRecord | None
    calculation: Calculation | None
    precision_calculation: NotRequired[PrecisionCalculation | None]
    precision_status: NotRequired[str]
    reciprocal_rank_calculation: NotRequired[ReciprocalRankCalculation | None]
    reciprocal_rank_status: NotRequired[str]
    ndcg_calculation: NotRequired[NdcgCalculation | None]
    ndcg_status: NotRequired[str]
    average_precision_calculation: NotRequired[AveragePrecisionCalculation | None]
    average_precision_status: NotRequired[str]
    context_precision_calculation: NotRequired[ContextPrecisionCalculation | None]
    context_precision_status: NotRequired[str]
    context_recall_calculation: NotRequired[ContextRecallCalculation | None]
    context_recall_status: NotRequired[str]
    exact_match_calculation: NotRequired[ExactMatchCalculation | None]
    exact_match_status: NotRequired[str]
    token_f1_calculation: NotRequired[TokenF1Calculation | None]
    token_f1_status: NotRequired[str]
    generation_judgments: NotRequired[dict[str, dict[str, object]]]
    trace: NotRequired[ExecutionTrace]


class Run(TypedDict):
    """Represent run data."""

    schema_version: int
    id: str
    created_at: str
    dataset_version: str
    mode: str
    config: dict[str, object]
    threshold: float
    k: int
    results: list[Result]
    summary: dict[str, int]
    mrr_summary: NotRequired[MrrSummary]
    mrr_recomputed: NotRequired[bool]
    map_summary: NotRequired[MeanMetricSummary]
    generation_summary: NotRequired[dict[str, float | int | None]]
    content_redacted: NotRequired[bool]
