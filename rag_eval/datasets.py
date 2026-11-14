"""Portable synthetic case packs without connection credentials."""

from __future__ import annotations

from dataclasses import asdict
from pydantic import ValidationError

from .core import RagResponse, TestCase
from .dto import CasePackInput, validation_issue


def export_case_pack(cases: list[TestCase], responses: dict[str, RagResponse]) -> dict[str, object]:
    """Export case pack."""
    return {
        "schema_version": 1,
        "cases": [asdict(c) for c in cases],
        "responses": {
            case.id: asdict(responses[case.id]) for case in cases if case.id in responses
        },
    }


def import_case_pack(
    data: dict[str, object], allowed_ids: set[str]
) -> tuple[list[TestCase], dict[str, RagResponse]]:
    """Import case pack."""
    try:
        pack = CasePackInput.model_validate(data)
    except ValidationError as exc:
        raise ValueError(f"Invalid case pack: {validation_issue(exc)}") from exc
    cases: list[TestCase] = []
    responses: dict[str, RagResponse] = {}
    for entry in pack.cases:
        case = entry.to_domain()
        case.validate()
        if not set(case.relevant_ids).issubset(allowed_ids):
            raise ValueError(f"Unknown relevant ID in {case.id}")
        if not set(case.relevance_grades).issubset(allowed_ids):
            raise ValueError(f"Unknown relevance grade ID in {case.id}")
        if not set(case.context_relevance).issubset(allowed_ids) or any(
            not set(ids).issubset(allowed_ids) for ids in case.point_support.values()
        ):
            raise ValueError(f"Unknown context label ID in {case.id}")
        if case.id not in pack.responses:
            raise ValueError(f"Missing response for {case.id}")
        response = pack.responses[case.id].to_domain()
        if not {item.id for item in response.retrieved}.issubset(allowed_ids):
            raise ValueError(f"Unknown retrieved ID in {case.id}")
        cases.append(case)
        responses[case.id] = response
    if not cases or len({case.id for case in cases}) != len(cases):
        raise ValueError("Case pack must contain unique case IDs")
    return cases, responses
