from __future__ import annotations

from dataclasses import asdict, dataclass, field
from copy import deepcopy
from collections.abc import Mapping
from datetime import datetime, timezone
from hashlib import sha256
from math import log2
from typing import NotRequired, TypedDict
from uuid import uuid4


@dataclass(frozen=True)
class TestCase:
    id: str
    name: str
    question: str
    relevant_ids: tuple[str, ...]
    expected_answer: str
    category: str
    user_id: str = "employee"
    notes: str = ""
    supporting_evidence: str = ""
    relevance_grades: dict[str, int] = field(default_factory=dict)
    context_relevance: dict[str, bool] = field(default_factory=dict)
    reference_points: tuple[str, ...] = ()
    point_support: dict[str, tuple[str, ...]] = field(default_factory=dict)

    def validate(self) -> None:
        for label, value in (("Case ID", self.id), ("name", self.name),
                             ("question", self.question), ("category", self.category),
                             ("user ID", self.user_id)):
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"{label} is required")
        if any(not isinstance(value, str) for value in (
            self.expected_answer, self.notes, self.supporting_evidence
        )):
            raise ValueError("Expected answer, supporting evidence, and notes must be strings")
        if not isinstance(self.relevant_ids, tuple) or any(
            not isinstance(item, str) or not item.strip() for item in self.relevant_ids
        ):
            raise ValueError("Relevant IDs must be nonempty strings")
        if len(set(self.relevant_ids)) != len(self.relevant_ids):
            raise ValueError(f"Duplicate relevant IDs in {self.id}")
        if not isinstance(self.relevance_grades, dict) or any(
            not isinstance(key, str) or not key.strip() or
            not isinstance(grade, int) or isinstance(grade, bool) or grade < 0
            for key, grade in self.relevance_grades.items()
        ):
            raise ValueError("Relevance grades must map nonempty IDs to nonnegative integers")
        if not isinstance(self.context_relevance, dict) or any(
            not isinstance(key, str) or not key.strip() or not isinstance(value, bool)
            for key, value in self.context_relevance.items()
        ):
            raise ValueError("Context relevance must map nonempty IDs to boolean judgments")
        if not isinstance(self.reference_points, tuple) or any(
            not isinstance(point, str) or not point.strip() for point in self.reference_points
        ) or len(set(self.reference_points)) != len(self.reference_points):
            raise ValueError("Reference points must be distinct nonempty strings")
        if not isinstance(self.point_support, dict) or set(self.point_support) != set(self.reference_points) or any(
            not isinstance(ids, tuple) or any(not isinstance(item, str) or not item.strip() for item in ids)
            for ids in self.point_support.values()
        ):
            raise ValueError("Point support must map every reference point to a tuple of context IDs")


@dataclass(frozen=True)
class RetrievedItem:
    id: str
    title: str = ""
    text: str = ""
    source_url: str = ""
    document_id: str = ""
    chunk_id: str = ""

    def validate(self) -> None:
        if not isinstance(self.id, str) or not self.id.strip():
            raise ValueError("Retrieved ID is required")
        if any(not isinstance(value, str) for value in (
            self.title, self.text, self.source_url, self.document_id, self.chunk_id
        )):
            raise ValueError("Retrieved item fields must be strings")


@dataclass(frozen=True)
class RagResponse:
    answer: str
    retrieved: tuple[RetrievedItem, ...]
    citations: tuple[str, ...] = ()
    raw: dict[str, object] = field(default_factory=lambda: {})

    def validate(self) -> None:
        if not isinstance(self.answer, str):
            raise ValueError("Answer must be a string")
        if not isinstance(self.retrieved, tuple):
            raise ValueError("Retrieved results must be a ranked tuple")
        for item in self.retrieved:
            if not isinstance(item, RetrievedItem):
                raise ValueError("Retrieved results must contain retrieved items")
            item.validate()
        if not isinstance(self.citations, tuple) or any(
            not isinstance(item, str) or not item.strip() for item in self.citations
        ):
            raise ValueError("Citations must be nonempty ID strings")
        if not isinstance(self.raw, dict):
            raise ValueError("Raw metadata must be a dictionary")


class Calculation(TypedDict):
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
    status: str
    score: float | None
    reason: str
    first_relevant_rank: int | None
    first_relevant_id: str | None
    ranked_ids: list[str]


class MrrSummary(TypedDict):
    score: float | None
    reciprocal_rank_sum: float
    scored_cases: int
    excluded_cases: int


class NdcgCalculation(TypedDict):
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
    status: str
    score: float | None
    reason: str
    relevant_ranks: list[int]
    precision_at_relevant_ranks: list[float]
    relevant_total: int
    ranked_ids: list[str]
    duplicate_ranks: list[int]


class ContextPrecisionCalculation(TypedDict):
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
    status: str
    score: float | None
    reason: str
    covered_points: list[str]
    missed_points: list[str]
    support_by_point: dict[str, list[str]]
    reference_total: int
    ranked_ids: list[str]


class MeanMetricSummary(TypedDict):
    score: float | None
    score_sum: float
    scored_cases: int
    excluded_cases: int


class TraceEvent(TypedDict):
    at: str
    stage: str
    status: str
    detail: str


class ExecutionTrace(TypedDict):
    request: dict[str, str]
    started_at: str | None
    finished_at: str | None
    recorded_at: str
    events: list[TraceEvent]
    ranked_retrieval_status: str
    generation_context_status: str


class CaseRecord(TypedDict):
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


class ResponseRecord(TypedDict):
    answer: str
    retrieved: list[dict[str, str]] | tuple[dict[str, str], ...]
    citations: list[str] | tuple[str, ...]
    raw: dict[str, object]


class Result(TypedDict):
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
    trace: NotRequired[ExecutionTrace]


class Run(TypedDict):
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
    content_redacted: NotRequired[bool]


def case_record(case: TestCase) -> CaseRecord:
    return {"id": case.id, "name": case.name, "question": case.question,
            "relevant_ids": case.relevant_ids, "expected_answer": case.expected_answer,
            "category": case.category, "user_id": case.user_id, "notes": case.notes,
            "supporting_evidence": case.supporting_evidence,
            "relevance_grades": case.relevance_grades,
            "context_relevance": case.context_relevance,
            "reference_points": case.reference_points,
            "point_support": case.point_support}


def response_record(response: RagResponse) -> ResponseRecord:
    return {"answer": response.answer,
            "retrieved": [{"id": item.id, "title": item.title, "text": item.text,
                           "source_url": item.source_url, "document_id": item.document_id,
                           "chunk_id": item.chunk_id} for item in response.retrieved],
            "citations": list(response.citations), "raw": response.raw}


def recall_at_k(retrieved_ids: list[str], relevant_ids: list[str], k: int) -> Calculation:
    """Set recall of relevant IDs among the first k ranked hits.

    Duplicate retrieved IDs do not count twice; ranks are not shifted by deduplication.
    """
    if k < 1:
        raise ValueError("K must be at least 1")
    if not relevant_ids:
        return {"status": "not_applicable", "score": None, "reason": "No labeled relevant IDs", "k": k,
                "numerator": 0, "denominator": 0, "hits": [], "misses": [], "top_k": retrieved_ids[:k]}
    relevant = set(relevant_ids)
    top_k = retrieved_ids[:k]
    found = set(top_k) & relevant
    hits = [item for item in relevant_ids if item in found]
    misses = [item for item in relevant_ids if item not in found]
    return {"status": "scored", "score": len(found) / len(relevant), "reason": "",
            "k": k, "numerator": len(found), "denominator": len(relevant), "hits": hits,
            "misses": misses, "top_k": top_k}


def precision_at_k(retrieved_ids: list[str], relevant_ids: list[str], k: int) -> PrecisionCalculation:
    """Relevant unique IDs in the first K ranks, divided by K.

    An unfilled rank and a duplicate both use a position without earning a hit.
    """
    if k < 1:
        raise ValueError("K must be at least 1")
    top_k = retrieved_ids[:k]
    if not relevant_ids:
        return {"status": "not_applicable", "score": None, "reason": "No labeled relevant IDs",
                "k": k, "numerator": 0, "denominator": k, "relevant_ranks": [],
                "duplicate_ranks": [], "top_k": top_k}
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
    return {"status": "scored", "score": len(relevant_ranks) / k, "reason": "",
            "k": k, "numerator": len(relevant_ranks), "denominator": k,
            "relevant_ranks": relevant_ranks, "duplicate_ranks": duplicate_ranks, "top_k": top_k}


def reciprocal_rank(retrieved_ids: list[str], relevant_ids: list[str]) -> ReciprocalRankCalculation:
    """Score the first labeled relevant hit over the complete ranked list."""
    if not relevant_ids:
        return {"status": "not_applicable", "score": None, "reason": "No labeled relevant IDs",
                "first_relevant_rank": None, "first_relevant_id": None, "ranked_ids": list(retrieved_ids)}
    relevant = set(relevant_ids)
    for rank, item in enumerate(retrieved_ids, 1):
        if item in relevant:
            return {"status": "scored", "score": 1 / rank, "reason": "",
                    "first_relevant_rank": rank, "first_relevant_id": item,
                    "ranked_ids": list(retrieved_ids)}
    return {"status": "scored", "score": 0.0, "reason": "No relevant ID retrieved",
            "first_relevant_rank": None, "first_relevant_id": None,
            "ranked_ids": list(retrieved_ids)}


def ndcg_at_k(retrieved_ids: list[str], relevance_grades: dict[str, int], k: int) -> NdcgCalculation:
    """Normalized DCG with exponential gain and log2(rank + 1) discount."""
    if k < 1:
        raise ValueError("K must be at least 1")
    if any(not isinstance(item, str) or not item.strip() for item in retrieved_ids):
        raise ValueError("Retrieved IDs must be nonempty strings")
    if not isinstance(relevance_grades, dict) or any(
        not isinstance(key, str) or not key.strip() or
        not isinstance(grade, int) or isinstance(grade, bool) or grade < 0
        for key, grade in relevance_grades.items()
    ):
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
    ideal_gains = [(2**grade - 1) / log2(rank + 1)
                   for rank, grade in enumerate(ideal_grades, 1)]
    dcg, ideal_dcg = sum(gains), sum(ideal_gains)
    return {"status": "scored" if ideal_dcg > 0 else "not_applicable",
            "score": dcg / ideal_dcg if ideal_dcg > 0 else None,
            "reason": "" if ideal_dcg > 0 else "No positive graded relevance labels",
            "k": k, "dcg": dcg, "ideal_dcg": ideal_dcg,
            "ranked_grades": grades, "ideal_grades": ideal_grades,
            "discounted_gains": gains, "ideal_discounted_gains": ideal_gains,
            "top_k": top_k, "duplicate_ranks": duplicates}


def average_precision(retrieved_ids: list[str], relevant_ids: list[str]) -> AveragePrecisionCalculation:
    """Average prefix precision at first occurrences of labeled relevant IDs."""
    if not relevant_ids:
        return {"status": "not_applicable", "score": None, "reason": "No labeled relevant IDs",
                "relevant_ranks": [], "precision_at_relevant_ranks": [],
                "relevant_total": 0, "ranked_ids": list(retrieved_ids), "duplicate_ranks": []}
    relevant, seen = set(relevant_ids), set()
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
    return {"status": "scored", "score": sum(precisions) / len(relevant), "reason": "",
            "relevant_ranks": ranks, "precision_at_relevant_ranks": precisions,
            "relevant_total": len(relevant), "ranked_ids": list(retrieved_ids),
            "duplicate_ranks": duplicates}


def context_precision(retrieved_ids: list[str], judgments: dict[str, bool]) -> ContextPrecisionCalculation:
    """RAGAS-style rank-sensitive precision from explicit context judgments."""
    unjudged = list(dict.fromkeys(item for item in retrieved_ids if item not in judgments))
    if not judgments or unjudged:
        return {"status": "not_applicable", "score": None,
                "reason": "Missing context relevance judgments" if not judgments else "Unjudged retrieved context IDs",
                "relevant_ranks": [], "precision_at_relevant_ranks": [],
                "relevant_retrieved": 0, "ranked_ids": list(retrieved_ids),
                "duplicate_ranks": [], "unjudged_ids": unjudged}
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
    return {"status": "scored", "score": sum(precisions) / len(ranks) if ranks else 0.0,
            "reason": "", "relevant_ranks": ranks,
            "precision_at_relevant_ranks": precisions,
            "relevant_retrieved": len(ranks), "ranked_ids": list(retrieved_ids),
            "duplicate_ranks": duplicates, "unjudged_ids": []}


def context_recall(retrieved_ids: list[str], reference_points: list[str],
                   point_support: dict[str, tuple[str, ...]]) -> ContextRecallCalculation:
    """Coverage of independently labeled reference claims by retrieved context."""
    if not reference_points:
        return {"status": "not_applicable", "score": None, "reason": "No reference points",
                "covered_points": [], "missed_points": [], "support_by_point": {},
                "reference_total": 0, "ranked_ids": list(retrieved_ids)}
    if set(point_support) != set(reference_points):
        raise ValueError("Point support must cover every reference point")
    retrieved = set(retrieved_ids)
    support = {point: [item for item in point_support[point] if item in retrieved]
               for point in reference_points}
    covered = [point for point in reference_points if support[point]]
    missed = [point for point in reference_points if not support[point]]
    return {"status": "scored", "score": len(covered) / len(reference_points),
            "reason": "", "covered_points": covered, "missed_points": missed,
            "support_by_point": support, "reference_total": len(reference_points),
            "ranked_ids": list(retrieved_ids)}


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
            rr = reciprocal_rank([item["id"] for item in ranked], list(result["case"]["relevant_ids"]))
            result["reciprocal_rank_calculation"] = rr
            result["reciprocal_rank_status"] = _metric_status(rr["score"], displayed["threshold"])
        calculation = result["reciprocal_rank_calculation"]
        if calculation is not None and calculation["score"] is not None:
            scores.append(calculation["score"])
    total = sum(scores)
    displayed["mrr_summary"] = {"score": total / len(scores) if scores else None,
                                "reciprocal_rank_sum": total, "scored_cases": len(scores),
                                "excluded_cases": len(displayed["results"]) - len(scores)}
    displayed["mrr_recomputed"] = True
    return displayed


def _metric_status(score: float | None, threshold: float) -> str:
    return "not_applicable" if score is None else "passed" if score >= threshold else "failed"


def _trace(case: TestCase, *, mode: str, response: RagResponse | None,
           error: str | None, at: str, timing: tuple[str, str] | None) -> ExecutionTrace:
    request = {"question": case.question, "user_id": case.user_id}
    events: list[TraceEvent] = [
        {"at": timing[0] if timing else at, "stage": "request", "status": "available",
         "detail": f"{mode} request recorded"},
        {"at": timing[1] if timing else at, "stage": "response", "status": "error" if error else "available",
         "detail": error or "Response captured"},
        {"at": at, "stage": "metrics", "status": "skipped" if error else "available",
         "detail": "Metrics unavailable after execution error" if error else "Retrieval and context metrics calculated where labeled"},
    ]
    return {"request": request, "started_at": timing[0] if timing else None,
            "finished_at": timing[1] if timing else None, "recorded_at": at, "events": events,
            "ranked_retrieval_status": "missing" if error else "available" if response is not None else "missing",
            "generation_context_status": "missing"}


def dataset_version(cases: list[TestCase]) -> str:
    import json
    canonical = json.dumps([asdict(c) for c in cases], sort_keys=True)
    return sha256(canonical.encode()).hexdigest()[:12]


def make_run(cases: list[TestCase], responses: Mapping[str, RagResponse | Exception], *, mode: str,
             k: int, threshold: float, config: dict[str, object],
             execution_times: Mapping[str, tuple[str, str]] | None = None) -> Run:
    if not 0 <= threshold <= 1:
        raise ValueError("Threshold must be between 0 and 1")
    run_id = f"run-{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}-{uuid4().hex[:6]}"
    results: list[Result] = []
    for case in cases:
        case.validate()
        supplied = responses.get(case.id)
        recorded_case = case_record(case)
        captured_at = datetime.now(timezone.utc).isoformat()
        if isinstance(supplied, Exception) or supplied is None:
            error = str(supplied or "Missing response")
            results.append({"case": recorded_case, "metric": "recall_at_k",
                            "metric_version": "1.0.0", "status": "error",
                            "error": error, "response": None, "calculation": None,
                            "precision_calculation": None, "precision_status": "error",
                            "reciprocal_rank_calculation": None, "reciprocal_rank_status": "error",
                            "ndcg_calculation": None, "ndcg_status": "error",
                            "average_precision_calculation": None, "average_precision_status": "error",
                            "context_precision_calculation": None, "context_precision_status": "error",
                            "context_recall_calculation": None, "context_recall_status": "error",
                            "trace": _trace(case, mode=mode, response=None, error=error, at=captured_at,
                                            timing=(execution_times or {}).get(case.id))})
            continue
        response = supplied
        try:
            response.validate()
        except ValueError as exc:
            error = f"Invalid RAG response: {exc}"
            results.append({"case": recorded_case, "metric": "recall_at_k",
                            "metric_version": "1.0.0", "status": "error",
                            "error": error, "response": None, "calculation": None,
                            "precision_calculation": None, "precision_status": "error",
                            "reciprocal_rank_calculation": None, "reciprocal_rank_status": "error",
                            "ndcg_calculation": None, "ndcg_status": "error",
                            "average_precision_calculation": None, "average_precision_status": "error",
                            "context_precision_calculation": None, "context_precision_status": "error",
                            "context_recall_calculation": None, "context_recall_status": "error",
                            "trace": _trace(case, mode=mode, response=None, error=error, at=captured_at,
                                            timing=(execution_times or {}).get(case.id))})
            continue
        calculation = recall_at_k([r.id for r in response.retrieved], list(case.relevant_ids), k)
        precision = precision_at_k([r.id for r in response.retrieved], list(case.relevant_ids), k)
        rr = reciprocal_rank([r.id for r in response.retrieved], list(case.relevant_ids))
        ndcg = ndcg_at_k([r.id for r in response.retrieved], case.relevance_grades, k)
        ap = average_precision([r.id for r in response.retrieved], list(case.relevant_ids))
        cp = context_precision([r.id for r in response.retrieved], case.context_relevance)
        cr = context_recall([r.id for r in response.retrieved], list(case.reference_points), case.point_support)
        score = calculation["score"]
        status = _metric_status(score, threshold)
        results.append({"case": recorded_case, "metric": "recall_at_k",
                        "metric_version": "1.0.0", "status": status,
                        "error": None, "response": response_record(response),
                        "calculation": calculation, "precision_calculation": precision,
                        "precision_status": _metric_status(precision["score"], threshold),
                        "reciprocal_rank_calculation": rr,
                        "reciprocal_rank_status": _metric_status(rr["score"], threshold),
                        "ndcg_calculation": ndcg,
                        "ndcg_status": _metric_status(ndcg["score"], threshold),
                        "average_precision_calculation": ap,
                        "average_precision_status": _metric_status(ap["score"], threshold),
                        "context_precision_calculation": cp,
                        "context_precision_status": _metric_status(cp["score"], threshold),
                        "context_recall_calculation": cr,
                        "context_recall_status": _metric_status(cr["score"], threshold),
                        "trace": _trace(case, mode=mode, response=response, error=None, at=captured_at,
                                        timing=(execution_times or {}).get(case.id))})
    rr_scores = [rr["score"] for result in results
                 if (rr := result.get("reciprocal_rank_calculation")) is not None
                 and rr["score"] is not None]
    rr_sum = sum(rr_scores)
    ap_scores = [ap["score"] for result in results
                 if (ap := result.get("average_precision_calculation")) is not None
                 and ap["score"] is not None]
    ap_sum = sum(ap_scores)
    return {"schema_version": 2, "id": run_id, "created_at": datetime.now(timezone.utc).isoformat(),
            "dataset_version": dataset_version(cases), "mode": mode, "config": config,
            "threshold": threshold, "k": k, "results": results,
            "summary": {status: sum(r["status"] == status for r in results)
                        for status in ("passed", "failed", "not_applicable", "error")},
            "mrr_summary": {"score": rr_sum / len(rr_scores) if rr_scores else None,
                            "reciprocal_rank_sum": rr_sum, "scored_cases": len(rr_scores),
                            "excluded_cases": len(results) - len(rr_scores)},
            "map_summary": {"score": ap_sum / len(ap_scores) if ap_scores else None,
                            "score_sum": ap_sum, "scored_cases": len(ap_scores),
                            "excluded_cases": len(results) - len(ap_scores)}}
