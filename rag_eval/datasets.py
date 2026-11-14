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
            case = TestCase(cast(str, entry["id"]), cast(str, entry["name"]), cast(str, entry["question"]),
                            tuple(cast(list[str], relevant_ids)), cast(str, entry["expected_answer"]),
                            cast(str, entry["category"]), cast(str, entry.get("user_id", "employee")),
                            cast(str, entry.get("notes", "")))
            case.validate()
            if not set(case.relevant_ids).issubset(allowed_ids):
                raise ValueError(f"Unknown relevant ID in {case.id}")
            raw = response_entries[case.id]
            ranked = tuple(RetrievedItem(cast(str, item["id"]), cast(str, item.get("title", "")),
                                            cast(str, item.get("text", "")), cast(str, item.get("source_url", "")))
                           for item in cast(list[dict[str, object]], raw["retrieved"]))
            if not {r.id for r in ranked}.issubset(allowed_ids):
                raise ValueError(f"Unknown retrieved ID in {case.id}")
            response = RagResponse(cast(str, raw["answer"]), ranked,
                                   tuple(cast(list[str], raw.get("citations", []))))
        except (KeyError, TypeError, AttributeError) as exc:
            raise ValueError(f"Invalid case pack entry: {exc}") from exc
        cases.append(case)
        responses[case.id] = response
    if not cases or len({c.id for c in cases}) != len(cases):
        raise ValueError("Case pack must contain unique case IDs")
    return cases, responses
