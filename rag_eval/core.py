from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from hashlib import sha256
from typing import Any
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
    raw: dict[str, Any] = field(default_factory=dict)


def recall_at_k(retrieved_ids: list[str], relevant_ids: list[str], k: int) -> dict[str, Any]:
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


def dataset_version(cases: list[TestCase]) -> str:
    import json
    canonical = json.dumps([asdict(c) for c in cases], sort_keys=True)
    return sha256(canonical.encode()).hexdigest()[:12]


def make_run(cases: list[TestCase], responses: dict[str, RagResponse | Exception], *, mode: str,
             k: int, threshold: float, config: dict[str, Any]) -> dict[str, Any]:
    if not 0 <= threshold <= 1:
        raise ValueError("Threshold must be between 0 and 1")
    run_id = f"run-{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}-{uuid4().hex[:6]}"
    results: list[dict[str, Any]] = []
    for case in cases:
        case.validate()
        supplied = responses.get(case.id)
        base = {"case": asdict(case), "metric": "recall_at_k", "metric_version": "1.0.0"}
        if isinstance(supplied, Exception) or supplied is None:
            results.append({**base, "status": "error", "error": str(supplied or "Missing response"),
                            "response": None, "calculation": None})
            continue
        response = supplied
        calculation = recall_at_k([r.id for r in response.retrieved], list(case.relevant_ids), k)
        status = ("not_applicable" if calculation["status"] == "not_applicable" else
                  "passed" if calculation["score"] >= threshold else "failed")
        results.append({**base, "status": status, "error": None, "response": asdict(response),
                        "calculation": calculation})
    return {"id": run_id, "created_at": datetime.now(timezone.utc).isoformat(),
            "dataset_version": dataset_version(cases), "mode": mode, "config": config,
            "threshold": threshold, "k": k, "results": results,
            "summary": {status: sum(r["status"] == status for r in results)
                        for status in ("passed", "failed", "not_applicable", "error")}}
