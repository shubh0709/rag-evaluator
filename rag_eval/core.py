from __future__ import annotations

from dataclasses import asdict, dataclass, field
from collections.abc import Mapping
from datetime import datetime, timezone
from hashlib import sha256
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

    def validate(self) -> None:
        if not self.id.strip() or not self.question.strip():
            raise ValueError("Case ID and question are required")
        if len(set(self.relevant_ids)) != len(self.relevant_ids):
            raise ValueError(f"Duplicate relevant IDs in {self.id}")


@dataclass(frozen=True)
class RetrievedItem:
    id: str
    title: str = ""
    text: str = ""
    source_url: str = ""


@dataclass(frozen=True)
class RagResponse:
    answer: str
    retrieved: tuple[RetrievedItem, ...]
    citations: tuple[str, ...] = ()
    raw: dict[str, object] = field(default_factory=lambda: {})


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


class TraceEvent(TypedDict):
    at: str
    stage: str
    status: str
    detail: str


class ExecutionTrace(TypedDict):
    request: dict[str, str]
    started_at: str
    finished_at: str
    events: list[TraceEvent]
    retrieved_context_status: str


class CaseRecord(TypedDict):
    id: str
    name: str
    question: str
    relevant_ids: tuple[str, ...]
    expected_answer: str
    category: str
    user_id: str
    notes: str


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
    trace: NotRequired[ExecutionTrace]


class Run(TypedDict):
    id: str
    created_at: str
    dataset_version: str
    mode: str
    config: dict[str, object]
    threshold: float
    k: int
    results: list[Result]
    summary: dict[str, int]
    content_redacted: NotRequired[bool]


def case_record(case: TestCase) -> CaseRecord:
    return {"id": case.id, "name": case.name, "question": case.question,
            "relevant_ids": case.relevant_ids, "expected_answer": case.expected_answer,
            "category": case.category, "user_id": case.user_id, "notes": case.notes}


def response_record(response: RagResponse) -> ResponseRecord:
    return {"answer": response.answer,
            "retrieved": [{"id": item.id, "title": item.title, "text": item.text,
                           "source_url": item.source_url} for item in response.retrieved],
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


def _metric_status(score: float | None, threshold: float) -> str:
    return "not_applicable" if score is None else "passed" if score >= threshold else "failed"


def _trace(case: TestCase, *, mode: str, response: RagResponse | None,
           error: str | None, at: str) -> ExecutionTrace:
    request = {"question": case.question, "user_id": case.user_id}
    events: list[TraceEvent] = [
        {"at": at, "stage": "request", "status": "available", "detail": f"{mode} case submitted"},
        {"at": at, "stage": "response", "status": "error" if error else "available",
         "detail": error or "Response captured"},
        {"at": at, "stage": "metrics", "status": "skipped" if error else "available",
         "detail": "Metrics unavailable after execution error" if error else "Recall@K and Precision@K calculated"},
    ]
    return {"request": request, "started_at": at, "finished_at": at, "events": events,
            "retrieved_context_status": "missing" if error else "available" if response is not None else "missing"}


def dataset_version(cases: list[TestCase]) -> str:
    import json
    canonical = json.dumps([asdict(c) for c in cases], sort_keys=True)
    return sha256(canonical.encode()).hexdigest()[:12]


def make_run(cases: list[TestCase], responses: Mapping[str, RagResponse | Exception], *, mode: str,
             k: int, threshold: float, config: dict[str, object]) -> Run:
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
                            "trace": _trace(case, mode=mode, response=None, error=error, at=captured_at)})
            continue
        response = supplied
        calculation = recall_at_k([r.id for r in response.retrieved], list(case.relevant_ids), k)
        precision = precision_at_k([r.id for r in response.retrieved], list(case.relevant_ids), k)
        score = calculation["score"]
        status = _metric_status(score, threshold)
        results.append({"case": recorded_case, "metric": "recall_at_k",
                        "metric_version": "1.0.0", "status": status,
                        "error": None, "response": response_record(response),
                        "calculation": calculation, "precision_calculation": precision,
                        "precision_status": _metric_status(precision["score"], threshold),
                        "trace": _trace(case, mode=mode, response=response, error=None, at=captured_at)})
    return {"id": run_id, "created_at": datetime.now(timezone.utc).isoformat(),
            "dataset_version": dataset_version(cases), "mode": mode, "config": config,
            "threshold": threshold, "k": k, "results": results,
            "summary": {status: sum(r["status"] == status for r in results)
                        for status in ("passed", "failed", "not_applicable", "error")}}
