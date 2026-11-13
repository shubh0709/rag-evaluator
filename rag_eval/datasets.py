"""Portable synthetic case packs without connection credentials."""
from __future__ import annotations

from dataclasses import asdict
from typing import Any

from .core import RagResponse, RetrievedItem, TestCase


def export_case_pack(cases: list[TestCase], responses: dict[str, RagResponse]) -> dict[str, Any]:
    return {"schema_version": 1, "cases": [asdict(c) for c in cases],
            "responses": {case.id: asdict(responses[case.id]) for case in cases if case.id in responses}}


def import_case_pack(data: dict[str, Any], allowed_ids: set[str]) -> tuple[list[TestCase], dict[str, RagResponse]]:
    if data.get("schema_version") != 1 or not isinstance(data.get("cases"), list) or not isinstance(data.get("responses"), dict):
        raise ValueError("Expected a version 1 case pack with cases and responses")
    cases = []
    responses = {}
    for entry in data["cases"]:
        try:
            case = TestCase(entry["id"], entry["name"], entry["question"],
                            tuple(entry["relevant_ids"]), entry["expected_answer"],
                            entry["category"], entry.get("user_id", "employee"), entry.get("notes", ""))
            case.validate()
            if any(not isinstance(i, str) for i in case.relevant_ids):
                raise ValueError("Relevant IDs must be strings")
            if not set(case.relevant_ids).issubset(allowed_ids):
                raise ValueError(f"Unknown relevant ID in {case.id}")
            raw = data["responses"][case.id]
            ranked = tuple(RetrievedItem(item["id"], item.get("title", ""), item.get("text", ""),
                                            item.get("source_url", "")) for item in raw["retrieved"])
            if not {r.id for r in ranked}.issubset(allowed_ids):
                raise ValueError(f"Unknown retrieved ID in {case.id}")
            response = RagResponse(raw["answer"], ranked, tuple(raw.get("citations", [])))
        except (KeyError, TypeError, AttributeError) as exc:
            raise ValueError(f"Invalid case pack entry: {exc}") from exc
        cases.append(case)
        responses[case.id] = response
    if not cases or len({c.id for c in cases}) != len(cases):
        raise ValueError("Case pack must contain unique case IDs")
    return cases, responses
