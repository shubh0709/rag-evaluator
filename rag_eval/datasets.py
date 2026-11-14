"""Portable synthetic case packs without connection credentials."""
from __future__ import annotations

from dataclasses import asdict
from typing import cast

from .core import RagResponse, RetrievedItem, TestCase


def export_case_pack(cases: list[TestCase], responses: dict[str, RagResponse]) -> dict[str, object]:
    return {"schema_version": 1, "cases": [asdict(c) for c in cases],
            "responses": {case.id: asdict(responses[case.id]) for case in cases if case.id in responses}}


def import_case_pack(data: dict[str, object], allowed_ids: set[str]) -> tuple[list[TestCase], dict[str, RagResponse]]:
    if data.get("schema_version") != 1 or not isinstance(data.get("cases"), list) or not isinstance(data.get("responses"), dict):
        raise ValueError("Expected a version 1 case pack with cases and responses")
    cases: list[TestCase] = []
    responses: dict[str, RagResponse] = {}
    entries = cast(list[dict[str, object]], data["cases"])
    response_entries = cast(dict[str, dict[str, object]], data["responses"])
    for entry in entries:
        try:
            relevant_ids = cast(list[object], entry["relevant_ids"])
            if any(not isinstance(i, str) for i in relevant_ids):
                raise ValueError("Relevant IDs must be strings")
            grades = entry.get("relevance_grades", {})
            if not isinstance(grades, dict):
                raise ValueError("Relevance grades must be an ID-to-grade object")
            context_judgments = entry.get("context_relevance", {})
            points = entry.get("reference_points", [])
            support = entry.get("point_support", {})
            if not isinstance(context_judgments, dict) or not isinstance(points, (list, tuple)) or not isinstance(support, dict):
                raise ValueError("Context labels must contain judgments, points, and support mappings")
            if any(not isinstance(ids, (list, tuple)) for ids in support.values()):
                raise ValueError("Point support values must be ID lists")
            case = TestCase(cast(str, entry["id"]), cast(str, entry["name"]), cast(str, entry["question"]),
                            tuple(cast(list[str], relevant_ids)), cast(str, entry["expected_answer"]),
                            cast(str, entry["category"]), cast(str, entry.get("user_id", "employee")),
                            cast(str, entry.get("notes", "")), cast(str, entry.get("supporting_evidence", "")),
                            cast(dict[str, int], grades), cast(dict[str, bool], context_judgments),
                            tuple(cast(list[str], points)),
                            {cast(str, point): tuple(cast(list[str], ids)) for point, ids in support.items()})
            case.validate()
            if not set(case.relevant_ids).issubset(allowed_ids):
                raise ValueError(f"Unknown relevant ID in {case.id}")
            if not set(case.relevance_grades).issubset(allowed_ids):
                raise ValueError(f"Unknown relevance grade ID in {case.id}")
            if not set(case.context_relevance).issubset(allowed_ids) or any(
                not set(ids).issubset(allowed_ids) for ids in case.point_support.values()
            ):
                raise ValueError(f"Unknown context label ID in {case.id}")
            raw = response_entries[case.id]
            ranked = tuple(RetrievedItem(cast(str, item["id"]), cast(str, item.get("title", "")),
                                            cast(str, item.get("text", "")), cast(str, item.get("source_url", "")),
                                            cast(str, item.get("document_id", "")), cast(str, item.get("chunk_id", "")))
                           for item in cast(list[dict[str, object]], raw["retrieved"]))
            if not {r.id for r in ranked}.issubset(allowed_ids):
                raise ValueError(f"Unknown retrieved ID in {case.id}")
            response = RagResponse(cast(str, raw["answer"]), ranked,
                                   tuple(cast(list[str], raw.get("citations", []))))
            response.validate()
        except (KeyError, TypeError, AttributeError) as exc:
            raise ValueError(f"Invalid case pack entry: {exc}") from exc
        cases.append(case)
        responses[case.id] = response
    if not cases or len({c.id for c in cases}) != len(cases):
        raise ValueError("Case pack must contain unique case IDs")
    return cases, responses
