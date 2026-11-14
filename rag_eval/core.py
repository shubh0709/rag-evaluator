"""Provide core utilities and tests."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from copy import deepcopy
from collections import Counter
from collections.abc import Mapping
from datetime import datetime, timezone
from hashlib import sha256
import json
from math import log2
import re
import string
from typing import Unpack, cast
import unicodedata
from uuid import uuid4

from .options import RunOptions

from .records import (
    Calculation,
    PrecisionCalculation,
    ReciprocalRankCalculation,
    NdcgCalculation,
    AveragePrecisionCalculation,
    ContextPrecisionCalculation,
    ContextRecallCalculation,
    ExactMatchCalculation,
    TokenF1Calculation,
    TraceEvent,
    ExecutionTrace,
    CaseRecord,
    ResponseRecord,
    Result,
    Run,
)


def _nonempty_string(value: object) -> bool:
    return isinstance(value, str) and bool(value.strip())


def _string(value: object) -> bool:
    return isinstance(value, str)


def _valid_reference_tuple(value: object) -> bool:
    return isinstance(value, tuple) and all(
        _nonempty_string(item) for item in cast(tuple[object, ...], value)
    )


def _valid_relevance_grades(value: object) -> bool:
    return isinstance(value, dict) and all(
        _nonempty_string(key)
        and isinstance(grade, int)
        and not isinstance(grade, bool)
        and grade >= 0
        for key, grade in cast(dict[object, object], value).items()
    )


def _positive_integer(value: object) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value >= 1


@dataclass(frozen=True)
class TestCase:
    """Represent test case data."""

    id: str
    name: str
    question: str
    relevant_ids: tuple[str, ...]
    expected_answer: str
    category: str
    user_id: str = "employee"
    notes: str = ""
    supporting_evidence: str = ""
    relevance_grades: dict[str, int] = field(default_factory=dict[str, int])
    context_relevance: dict[str, bool] = field(default_factory=dict[str, bool])
    reference_points: tuple[str, ...] = ()
    point_support: dict[str, tuple[str, ...]] = field(default_factory=dict[str, tuple[str, ...]])
    exact_match_references: tuple[str, ...] = ()
    answerable: bool | None = None
    trap_prior_answer: str = ""
    trap_context_answer: str = ""

    def validate(self) -> None:
        """Validate the case and its identity, retrieval, and generation labels."""
        _validate_case_identity(self)
        _validate_case_context(self)
        _validate_case_generation(self)


def _validate_case_identity(case: TestCase) -> None:
    """Validate the identity labels of a case."""
    runtime: Mapping[str, object] = vars(case)
    for label, value in (
        ("Case ID", runtime["id"]),
        ("name", runtime["name"]),
        ("question", runtime["question"]),
        ("category", runtime["category"]),
        ("user ID", runtime["user_id"]),
    ):
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"{label} is required")
    if any(
        not isinstance(value, str)
        for value in (runtime["expected_answer"], runtime["notes"], runtime["supporting_evidence"])
    ):
        raise ValueError("Expected answer, supporting evidence, and notes must be strings")
    if not isinstance(runtime["relevant_ids"], tuple) or any(
        not isinstance(item, str) or not item.strip()
        for item in cast(tuple[object, ...], runtime["relevant_ids"])
    ):
        raise ValueError("Relevant IDs must be nonempty strings")
    if len(set(case.relevant_ids)) != len(case.relevant_ids):
        raise ValueError(f"Duplicate relevant IDs in {case.id}")


def _validate_case_context(case: TestCase) -> None:
    """Validate the context labels of a case."""
    runtime: Mapping[str, object] = vars(case)
    if not _valid_relevance_grades(runtime["relevance_grades"]):
        raise ValueError("Relevance grades must map nonempty IDs to nonnegative integers")
    if not isinstance(runtime["context_relevance"], dict) or any(
        not isinstance(key, str) or not key.strip() or not isinstance(value, bool)
        for key, value in cast(dict[object, object], runtime["context_relevance"]).items()
    ):
        raise ValueError("Context relevance must map nonempty IDs to boolean judgments")
    if (
        not isinstance(runtime["reference_points"], tuple)
        or any(
            not isinstance(point, str) or not point.strip()
            for point in cast(tuple[object, ...], runtime["reference_points"])
        )
        or len(set(case.reference_points)) != len(case.reference_points)
    ):
        raise ValueError("Reference points must be distinct nonempty strings")
    if (
        not isinstance(runtime["point_support"], dict)
        or set(cast(dict[object, object], runtime["point_support"])) != set(case.reference_points)
        or any(
            not isinstance(ids, tuple)
            or any(
                not isinstance(item, str) or not item.strip()
                for item in cast(tuple[object, ...], ids)
            )
            for ids in cast(dict[object, object], runtime["point_support"]).values()
        )
    ):
        raise ValueError("Point support must map every reference point to a tuple of context IDs")


def _validate_case_generation(case: TestCase) -> None:
    """Validate the generation labels of a case."""
    runtime: Mapping[str, object] = vars(case)
    if not isinstance(runtime["exact_match_references"], tuple) or any(
        not isinstance(reference, str) or not reference.strip()
        for reference in cast(tuple[object, ...], runtime["exact_match_references"])
    ):
        raise ValueError("Exact Match references must be nonempty strings")
    if any(not normalize_exact_match(reference) for reference in case.exact_match_references):
        raise ValueError("Exact Match reference is empty after normalization")
    if runtime["answerable"] is not None and not isinstance(runtime["answerable"], bool):
        raise ValueError("Answerable must be true, false, or null")
    if not isinstance(runtime["trap_prior_answer"], str) or not isinstance(
        runtime["trap_context_answer"], str
    ):
        raise ValueError("Trap answers must be strings")
    if bool(runtime["trap_prior_answer"]) != bool(runtime["trap_context_answer"]):
        raise ValueError("Trap prior and context answers must be provided together")


@dataclass(frozen=True)
class RetrievedItem:
    """Represent retrieved item data."""

    id: str
    title: str = ""
    text: str = ""
    source_url: str = ""
    document_id: str = ""
    chunk_id: str = ""

    def validate(self) -> None:
        """Validate the object and its nested fields."""
        runtime: Mapping[str, object] = vars(self)
        id_value: object = runtime["id"]
        title_value: object = runtime["title"]
        text_value: object = runtime["text"]
        source_url_value: object = runtime["source_url"]
        document_id_value: object = runtime["document_id"]
        chunk_id_value: object = runtime["chunk_id"]
        if not isinstance(id_value, str) or not id_value.strip():
            raise ValueError("Retrieved ID is required")
        if any(
            not isinstance(value, str)
            for value in (
                title_value,
                text_value,
                source_url_value,
                document_id_value,
                chunk_id_value,
            )
        ):
            raise ValueError("Retrieved item fields must be strings")


@dataclass(frozen=True)
class RagResponse:
    """Represent rag response data."""

    answer: str
    retrieved: tuple[RetrievedItem, ...]
    citations: tuple[str, ...] = ()
    raw: dict[str, object] = field(default_factory=lambda: {})

    def validate(self) -> None:
        """Validate the object and its nested fields."""
        runtime: Mapping[str, object] = vars(self)
        answer_value: object = runtime["answer"]
        retrieved_value: object = runtime["retrieved"]
        citations_value: object = runtime["citations"]
        raw_value: object = runtime["raw"]
        if not isinstance(answer_value, str):
            raise ValueError("Answer must be a string")
        if not isinstance(retrieved_value, tuple):
            raise ValueError("Retrieved results must be a ranked tuple")
        for item in cast(tuple[object, ...], retrieved_value):
            if not isinstance(item, RetrievedItem):
                raise ValueError("Retrieved results must contain retrieved items")
            item.validate()
        if not isinstance(citations_value, tuple) or any(
            not isinstance(item, str) or not item.strip()
            for item in cast(tuple[object, ...], citations_value)
        ):
            raise ValueError("Citations must be nonempty ID strings")
        if not isinstance(raw_value, dict):
            raise ValueError("Raw metadata must be a dictionary")


def case_record(case: TestCase) -> CaseRecord:
    """Case record."""
    return {
        "id": case.id,
        "name": case.name,
        "question": case.question,
        "relevant_ids": case.relevant_ids,
        "expected_answer": case.expected_answer,
        "category": case.category,
        "user_id": case.user_id,
        "notes": case.notes,
        "supporting_evidence": case.supporting_evidence,
        "relevance_grades": case.relevance_grades,
        "context_relevance": case.context_relevance,
        "reference_points": case.reference_points,
        "point_support": case.point_support,
        "exact_match_references": case.exact_match_references,
        "answerable": case.answerable,
        "trap_prior_answer": case.trap_prior_answer,
        "trap_context_answer": case.trap_context_answer,
    }


def response_record(response: RagResponse) -> ResponseRecord:
    """Response record."""
    return {
        "answer": response.answer,
        "retrieved": [
            {
                "id": item.id,
                "title": item.title,
                "text": item.text,
                "source_url": item.source_url,
                "document_id": item.document_id,
                "chunk_id": item.chunk_id,
            }
            for item in response.retrieved
        ],
        "citations": list(response.citations),
        "raw": response.raw,
    }


def normalize_exact_match(value: str) -> str:
    """Casefold, remove punctuation and English articles, then collapse whitespace (v1)."""
    if not _string(value):
        raise ValueError("Exact Match input must be a string")
    folded = value.casefold()
    without_punctuation = "".join(
        character
        for character in folded
        if character not in string.punctuation
        and not unicodedata.category(character).startswith("P")
    )
    without_articles = re.sub(r"\b(a|an|the)\b", " ", without_punctuation)
    return " ".join(without_articles.split())


def exact_match(answer: str, references: tuple[str, ...]) -> ExactMatchCalculation:
    """Binary normalized equality against any explicitly labeled reference."""
    if not _valid_reference_tuple(references):
        raise ValueError("Exact Match references must be nonempty strings")
    normalized_answer = normalize_exact_match(answer)
    normalized_references = [normalize_exact_match(reference) for reference in references]
    if not references:
        return {
            "status": "not_applicable",
            "score": None,
            "reason": "No Exact Match references",
            "normalization_version": "1.0.0",
            "normalized_answer": normalized_answer,
            "normalized_references": [],
            "matched_reference_index": None,
        }
    if any(not reference for reference in normalized_references):
        raise ValueError("Exact Match reference is empty after normalization")
    match = next(
        (
            index
            for index, reference in enumerate(normalized_references)
            if normalized_answer == reference
        ),
        None,
    )
    return {
        "status": "scored",
        "score": 1.0 if match is not None else 0.0,
        "reason": "" if match is not None else "Normalized answer differs from every reference",
        "normalization_version": "1.0.0",
        "normalized_answer": normalized_answer,
        "normalized_references": normalized_references,
        "matched_reference_index": match,
    }


def token_f1(answer: str, references: tuple[str, ...]) -> TokenF1Calculation:
    """Maximum bag-of-tokens F1 over explicitly labeled answers."""
    if not _valid_reference_tuple(references):
        raise ValueError("Token F1 references must be nonempty strings")
    answer_tokens = normalize_exact_match(answer).split()
    reference_tokens = [normalize_exact_match(reference).split() for reference in references]
    if not references:
        return {
            "status": "not_applicable",
            "score": None,
            "reason": "No Token F1 references",
            "normalization_version": "1.0.0",
            "answer_tokens": answer_tokens,
            "reference_tokens": [],
            "matched_reference_index": None,
            "overlap": 0,
            "precision": 0.0,
            "recall": 0.0,
        }
    if any(not tokens for tokens in reference_tokens):
        raise ValueError("Token F1 reference is empty after normalization")
    candidates: list[tuple[float, int, float, float]] = []
    for tokens in reference_tokens:
        overlap = sum((Counter(answer_tokens) & Counter(tokens)).values())
        precision = overlap / len(answer_tokens) if answer_tokens else 0.0
        recall = overlap / len(tokens)
        score = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        candidates.append((score, overlap, precision, recall))
    best_index = max(range(len(candidates)), key=lambda index: candidates[index][0])
    score, overlap, precision, recall = candidates[best_index]
    return {
        "status": "scored",
        "score": score,
        "reason": "",
        "normalization_version": "1.0.0",
        "answer_tokens": answer_tokens,
        "reference_tokens": reference_tokens,
        "matched_reference_index": best_index,
        "overlap": overlap,
        "precision": precision,
        "recall": recall,
    }


def recall_at_k(retrieved_ids: list[str], relevant_ids: list[str], k: int) -> Calculation:
    """Set recall of relevant IDs among the first k ranked hits.

    Duplicate retrieved IDs do not count twice; ranks are not shifted by deduplication.
    """
    if k < 1:
        raise ValueError("K must be at least 1")
    if not relevant_ids:
        return {
            "status": "not_applicable",
            "score": None,
            "reason": "No labeled relevant IDs",
            "k": k,
            "numerator": 0,
            "denominator": 0,
            "hits": [],
            "misses": [],
            "top_k": retrieved_ids[:k],
        }
    relevant = set(relevant_ids)
    top_k = retrieved_ids[:k]
    found = set(top_k) & relevant
    hits = [item for item in relevant_ids if item in found]
    misses = [item for item in relevant_ids if item not in found]
    return {
        "status": "scored",
        "score": len(found) / len(relevant),
        "reason": "",
        "k": k,
        "numerator": len(found),
        "denominator": len(relevant),
        "hits": hits,
        "misses": misses,
        "top_k": top_k,
    }


def precision_at_k(
    retrieved_ids: list[str], relevant_ids: list[str], k: int
) -> PrecisionCalculation:
    """Relevant unique IDs in the first K ranks, divided by K.

    An unfilled rank and a duplicate both use a position without earning a hit.
    """
    if k < 1:
        raise ValueError("K must be at least 1")
    top_k = retrieved_ids[:k]
    if not relevant_ids:
        return {
            "status": "not_applicable",
            "score": None,
            "reason": "No labeled relevant IDs",
            "k": k,
            "numerator": 0,
            "denominator": k,
            "relevant_ranks": [],
            "duplicate_ranks": [],
            "top_k": top_k,
        }
    relevant = set(relevant_ids)
    seen: set[str] = set()
    relevant_ranks: list[int] = []
    duplicate_ranks: list[int] = []
    for rank, item in enumerate(top_k, 1):
        if item in seen:
            duplicate_ranks.append(rank)
        elif item in relevant:
            relevant_ranks.append(rank)
        seen.add(item)
    return {
        "status": "scored",
        "score": len(relevant_ranks) / k,
        "reason": "",
        "k": k,
        "numerator": len(relevant_ranks),
        "denominator": k,
        "relevant_ranks": relevant_ranks,
        "duplicate_ranks": duplicate_ranks,
        "top_k": top_k,
    }


def reciprocal_rank(retrieved_ids: list[str], relevant_ids: list[str]) -> ReciprocalRankCalculation:
    """Score the first labeled relevant hit over the complete ranked list."""
    if not relevant_ids:
        return {
            "status": "not_applicable",
            "score": None,
            "reason": "No labeled relevant IDs",
            "first_relevant_rank": None,
            "first_relevant_id": None,
            "ranked_ids": list(retrieved_ids),
        }
    relevant = set(relevant_ids)
    for rank, item in enumerate(retrieved_ids, 1):
        if item in relevant:
            return {
                "status": "scored",
                "score": 1 / rank,
                "reason": "",
                "first_relevant_rank": rank,
                "first_relevant_id": item,
                "ranked_ids": list(retrieved_ids),
            }
    return {
        "status": "scored",
        "score": 0.0,
        "reason": "No relevant ID retrieved",
        "first_relevant_rank": None,
        "first_relevant_id": None,
        "ranked_ids": list(retrieved_ids),
    }


def ndcg_at_k(
    retrieved_ids: list[str], relevance_grades: dict[str, int], k: int
) -> NdcgCalculation:
    """Normalized DCG with exponential gain and log2(rank + 1) discount."""
    if k < 1:
        raise ValueError("K must be at least 1")
    if any(not _nonempty_string(item) for item in retrieved_ids):
        raise ValueError("Retrieved IDs must be nonempty strings")
    if not _valid_relevance_grades(relevance_grades):
        raise ValueError("Relevance grades must map nonempty IDs to nonnegative integers")
    top_k = retrieved_ids[:k]
    seen: set[str] = set()
    grades: list[int] = []
    duplicates: list[int] = []
    for rank, item in enumerate(top_k, 1):
        if item in seen:
            grades.append(0)
            duplicates.append(rank)
        else:
            grades.append(relevance_grades.get(item, 0))
            seen.add(item)
    ideal_grades = sorted(relevance_grades.values(), reverse=True)[:k]
    gains = [(2**grade - 1) / log2(rank + 1) for rank, grade in enumerate(grades, 1)]
    ideal_gains = [(2**grade - 1) / log2(rank + 1) for rank, grade in enumerate(ideal_grades, 1)]
    dcg, ideal_dcg = sum(gains), sum(ideal_gains)
    return {
        "status": "scored" if ideal_dcg > 0 else "not_applicable",
        "score": dcg / ideal_dcg if ideal_dcg > 0 else None,
        "reason": "" if ideal_dcg > 0 else "No positive graded relevance labels",
        "k": k,
        "dcg": dcg,
        "ideal_dcg": ideal_dcg,
        "ranked_grades": grades,
        "ideal_grades": ideal_grades,
        "discounted_gains": gains,
        "ideal_discounted_gains": ideal_gains,
        "top_k": top_k,
        "duplicate_ranks": duplicates,
    }


def average_precision(
    retrieved_ids: list[str], relevant_ids: list[str]
) -> AveragePrecisionCalculation:
    """Average prefix precision at first occurrences of labeled relevant IDs."""
    if not relevant_ids:
        return {
            "status": "not_applicable",
            "score": None,
            "reason": "No labeled relevant IDs",
            "relevant_ranks": [],
            "precision_at_relevant_ranks": [],
            "relevant_total": 0,
            "ranked_ids": list(retrieved_ids),
            "duplicate_ranks": [],
        }
    relevant, seen = set(relevant_ids), set[str]()
    ranks: list[int] = []
    precisions: list[float] = []
    duplicates: list[int] = []
    for rank, item in enumerate(retrieved_ids, 1):
        if item in seen:
            duplicates.append(rank)
        elif item in relevant:
            ranks.append(rank)
            precisions.append(len(ranks) / rank)
        seen.add(item)
    return {
        "status": "scored",
        "score": sum(precisions) / len(relevant),
        "reason": "",
        "relevant_ranks": ranks,
        "precision_at_relevant_ranks": precisions,
        "relevant_total": len(relevant),
        "ranked_ids": list(retrieved_ids),
        "duplicate_ranks": duplicates,
    }


def context_precision(
    retrieved_ids: list[str], judgments: dict[str, bool]
) -> ContextPrecisionCalculation:
    """RAGAS-style rank-sensitive precision from explicit context judgments."""
    unjudged = list(dict.fromkeys(item for item in retrieved_ids if item not in judgments))
    if not judgments or unjudged:
        return {
            "status": "not_applicable",
            "score": None,
            "reason": (
                "Missing context relevance judgments"
                if not judgments
                else "Unjudged retrieved context IDs"
            ),
            "relevant_ranks": [],
            "precision_at_relevant_ranks": [],
            "relevant_retrieved": 0,
            "ranked_ids": list(retrieved_ids),
            "duplicate_ranks": [],
            "unjudged_ids": unjudged,
        }
    ranks: list[int] = []
    precisions: list[float] = []
    duplicates: list[int] = []
    seen: set[str] = set()
    for rank, item in enumerate(retrieved_ids, 1):
        if item in seen:
            duplicates.append(rank)
        elif judgments[item]:
            ranks.append(rank)
            precisions.append(len(ranks) / rank)
        seen.add(item)
    return {
        "status": "scored",
        "score": sum(precisions) / len(ranks) if ranks else 0.0,
        "reason": "",
        "relevant_ranks": ranks,
        "precision_at_relevant_ranks": precisions,
        "relevant_retrieved": len(ranks),
        "ranked_ids": list(retrieved_ids),
        "duplicate_ranks": duplicates,
        "unjudged_ids": [],
    }


def context_recall(
    retrieved_ids: list[str], reference_points: list[str], point_support: dict[str, tuple[str, ...]]
) -> ContextRecallCalculation:
    """Coverage of independently labeled reference claims by retrieved context."""
    if not reference_points:
        return {
            "status": "not_applicable",
            "score": None,
            "reason": "No reference points",
            "covered_points": [],
            "missed_points": [],
            "support_by_point": {},
            "reference_total": 0,
            "ranked_ids": list(retrieved_ids),
        }
    if set(point_support) != set(reference_points):
        raise ValueError("Point support must cover every reference point")
    retrieved = set(retrieved_ids)
    support = {
        point: [item for item in point_support[point] if item in retrieved]
        for point in reference_points
    }
    covered = [point for point in reference_points if support[point]]
    missed = [point for point in reference_points if not support[point]]
    return {
        "status": "scored",
        "score": len(covered) / len(reference_points),
        "reason": "",
        "covered_points": covered,
        "missed_points": missed,
        "support_by_point": support,
        "reference_total": len(reference_points),
        "ranked_ids": list(retrieved_ids),
    }


def with_recomputed_mrr(run: Run) -> Run:
    """Show MRR for older saved runs using their stored labels and ranked IDs.

    Return a copy so historical artifacts retain their original metric provenance.
    """
    if "mrr_summary" in run:
        return run
    displayed = deepcopy(run)
    scores: list[float] = []
    for result in displayed["results"]:
        if "reciprocal_rank_calculation" not in result:
            response = result["response"]
            if response is None:
                result["reciprocal_rank_calculation"] = None
                result["reciprocal_rank_status"] = "error"
                continue
            ranked = response["retrieved"]
            rr = reciprocal_rank(
                [item["id"] for item in ranked], list(result["case"]["relevant_ids"])
            )
            result["reciprocal_rank_calculation"] = rr
            result["reciprocal_rank_status"] = _metric_status(rr["score"], displayed["threshold"])
        calculation = result["reciprocal_rank_calculation"]
        if calculation is not None and calculation["score"] is not None:
            scores.append(calculation["score"])
    total = sum(scores)
    displayed["mrr_summary"] = {
        "score": total / len(scores) if scores else None,
        "reciprocal_rank_sum": total,
        "scored_cases": len(scores),
        "excluded_cases": len(displayed["results"]) - len(scores),
    }
    displayed["mrr_recomputed"] = True
    return displayed


def _metric_status(score: float | None, threshold: float) -> str:
    return "not_applicable" if score is None else "passed" if score >= threshold else "failed"


@dataclass(frozen=True)
class TraceClock:
    """Capture the evaluator timestamp and optional live execution times."""

    at: str
    timing: tuple[str, str] | None


def _trace(
    case: TestCase,
    *,
    mode: str,
    response: RagResponse | None,
    error: str | None,
    clock: TraceClock,
) -> ExecutionTrace:
    request = {"question": case.question, "user_id": case.user_id}
    events: list[TraceEvent] = [
        {
            "at": clock.timing[0] if clock.timing else clock.at,
            "stage": "request",
            "status": "available",
            "detail": f"{mode} request recorded",
        },
        {
            "at": clock.timing[1] if clock.timing else clock.at,
            "stage": "response",
            "status": "error" if error else "available",
            "detail": error or "Response captured",
        },
        {
            "at": clock.at,
            "stage": "metrics",
            "status": "skipped" if error else "available",
            "detail": (
                "Metrics unavailable after execution error"
                if error
                else "Retrieval, context, and Exact Match metrics calculated where labeled"
            ),
        },
    ]
    return {
        "request": request,
        "started_at": clock.timing[0] if clock.timing else None,
        "finished_at": clock.timing[1] if clock.timing else None,
        "recorded_at": clock.at,
        "events": events,
        "ranked_retrieval_status": (
            "missing" if error else "available" if response is not None else "missing"
        ),
        "generation_context_status": "missing",
    }


def dataset_version(cases: list[TestCase]) -> str:
    """Dataset version."""

    canonical = json.dumps([asdict(c) for c in cases], sort_keys=True)
    return sha256(canonical.encode()).hexdigest()[:12]


def _error_result(case: TestCase, error: str, settings: RunOptions, clock: TraceClock) -> Result:
    """Build a result when execution or response validation fails."""
    return {
        "case": case_record(case),
        "metric": "recall_at_k",
        "metric_version": "1.0.0",
        "status": "error",
        "error": error,
        "response": None,
        "calculation": None,
        "precision_calculation": None,
        "precision_status": "error",
        "reciprocal_rank_calculation": None,
        "reciprocal_rank_status": "error",
        "ndcg_calculation": None,
        "ndcg_status": "error",
        "average_precision_calculation": None,
        "average_precision_status": "error",
        "context_precision_calculation": None,
        "context_precision_status": "error",
        "context_recall_calculation": None,
        "context_recall_status": "error",
        "exact_match_calculation": None,
        "exact_match_status": "error",
        "token_f1_calculation": None,
        "token_f1_status": "error",
        "trace": _trace(case, mode=settings["mode"], response=None, error=error, clock=clock),
    }


def _score_result(
    case: TestCase, response: RagResponse, settings: RunOptions, clock: TraceClock
) -> Result:
    """Calculate every labeled metric for one valid response."""
    calculation = recall_at_k(
        [r.id for r in response.retrieved], list(case.relevant_ids), settings["k"]
    )
    precision = precision_at_k(
        [r.id for r in response.retrieved], list(case.relevant_ids), settings["k"]
    )
    rr = reciprocal_rank([r.id for r in response.retrieved], list(case.relevant_ids))
    ndcg = ndcg_at_k([r.id for r in response.retrieved], case.relevance_grades, settings["k"])
    ap = average_precision([r.id for r in response.retrieved], list(case.relevant_ids))
    cp = context_precision([r.id for r in response.retrieved], case.context_relevance)
    cr = context_recall(
        [r.id for r in response.retrieved], list(case.reference_points), case.point_support
    )
    em = exact_match(response.answer, case.exact_match_references)
    f1 = token_f1(response.answer, case.exact_match_references)
    return {
        "case": case_record(case),
        "metric": "recall_at_k",
        "metric_version": "1.0.0",
        "status": _metric_status(calculation["score"], settings["threshold"]),
        "error": None,
        "response": response_record(response),
        "calculation": calculation,
        "precision_calculation": precision,
        "precision_status": _metric_status(precision["score"], settings["threshold"]),
        "reciprocal_rank_calculation": rr,
        "reciprocal_rank_status": _metric_status(rr["score"], settings["threshold"]),
        "ndcg_calculation": ndcg,
        "ndcg_status": _metric_status(ndcg["score"], settings["threshold"]),
        "average_precision_calculation": ap,
        "average_precision_status": _metric_status(ap["score"], settings["threshold"]),
        "context_precision_calculation": cp,
        "context_precision_status": _metric_status(cp["score"], settings["threshold"]),
        "context_recall_calculation": cr,
        "context_recall_status": _metric_status(cr["score"], settings["threshold"]),
        "exact_match_calculation": em,
        "exact_match_status": _metric_status(em["score"], settings["threshold"]),
        "token_f1_calculation": f1,
        "token_f1_status": _metric_status(f1["score"], settings["threshold"]),
        "trace": _trace(case, mode=settings["mode"], response=response, error=None, clock=clock),
    }


def _run_record(
    cases: list[TestCase], results: list[Result], settings: RunOptions, run_id: str
) -> Run:
    """Summarize scored cases into a serializable run."""
    rr_scores = [
        rr["score"]
        for result in results
        if (rr := result.get("reciprocal_rank_calculation")) is not None and rr["score"] is not None
    ]
    rr_sum = sum(rr_scores)
    ap_scores = [
        ap["score"]
        for result in results
        if (ap := result.get("average_precision_calculation")) is not None
        and ap["score"] is not None
    ]
    ap_sum = sum(ap_scores)
    return {
        "schema_version": 2,
        "id": run_id,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "dataset_version": dataset_version(cases),
        "mode": settings["mode"],
        "config": settings["config"],
        "threshold": settings["threshold"],
        "k": settings["k"],
        "results": results,
        "summary": {
            status: sum((r["status"] == status for r in results))
            for status in ("passed", "failed", "not_applicable", "error")
        },
        "mrr_summary": {
            "score": rr_sum / len(rr_scores) if rr_scores else None,
            "reciprocal_rank_sum": rr_sum,
            "scored_cases": len(rr_scores),
            "excluded_cases": len(results) - len(rr_scores),
        },
        "map_summary": {
            "score": ap_sum / len(ap_scores) if ap_scores else None,
            "score_sum": ap_sum,
            "scored_cases": len(ap_scores),
            "excluded_cases": len(results) - len(ap_scores),
        },
    }


def make_run(
    cases: list[TestCase],
    responses: Mapping[str, RagResponse | Exception],
    **settings: Unpack[RunOptions],
) -> Run:
    """Validate inputs, score cases, and assemble an evaluation run."""
    if not _positive_integer(settings["k"]):
        raise ValueError("K must be a positive integer")
    if not 0 <= settings["threshold"] <= 1:
        raise ValueError("Threshold must be between 0 and 1")
    run_id = f"run-{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}-{uuid4().hex[:6]}"
    results: list[Result] = []
    for case in cases:
        case.validate()
        supplied = responses.get(case.id)
        clock = TraceClock(
            datetime.now(timezone.utc).isoformat(),
            (settings.get("execution_times") or {}).get(case.id),
        )
        if isinstance(supplied, Exception) or supplied is None:
            result = _error_result(case, str(supplied or "Missing response"), settings, clock)
        else:
            try:
                supplied.validate()
            except ValueError as exc:
                result = _error_result(case, f"Invalid RAG response: {exc}", settings, clock)
            else:
                result = _score_result(case, supplied, settings, clock)
        results.append(result)
    return _run_record(cases, results, settings, run_id)
