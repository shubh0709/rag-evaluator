"""Configurable HTTP boundaries. Tokens are passed at call time and never persisted."""

from __future__ import annotations

from typing import Unpack, cast
from urllib.parse import urlparse
import copy

import requests
from pydantic import ValidationError

from .core import RagResponse, Run, TestCase
from .records import Result
from .options import LiveConnectionOptions
from .dto import BotResponseInput, validation_issue


def validate_url(url: str) -> None:
    """Validate url."""
    parsed = urlparse(url)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        raise ValueError("Enter a full http(s) URL")


def auth_headers(
    token: str, header_name: str = "Authorization", prefix: str = "Bearer"
) -> dict[str, str]:
    """Auth headers."""
    if not token:
        return {}
    if "\n" in token or "\r" in token or "\n" in header_name or "\r" in header_name:
        raise ValueError("Invalid token or header name")
    return {header_name: f"{prefix} {token}".strip()}


def check_connection(
    url: str,
    *,
    token: str = "",
    header_name: str = "Authorization",
    prefix: str = "Bearer",
    timeout: int = 10,
) -> dict[str, object]:
    """Check connection."""
    validate_url(url)
    try:
        response = requests.get(
            url, headers=auth_headers(token, header_name, prefix), timeout=timeout
        )
        reachable = response.ok or response.status_code == 405
        message = (
            "Reachable"
            if response.ok
            else (
                "Reachable, but GET is not allowed"
                if response.status_code == 405
                else "Request failed"
            )
        )
        return {"ok": reachable, "status_code": response.status_code, "message": message}
    except requests.RequestException as exc:
        return {"ok": False, "status_code": None, "message": str(exc)}


def run_live_case(case: TestCase, **connection: Unpack[LiveConnectionOptions]) -> RagResponse:
    """Run one case against the configured HTTP endpoint."""
    validate_url(connection["url"])
    payload = {"question": case.question, "user_id": case.user_id}
    response = requests.post(
        connection["url"],
        json=payload,
        headers=auth_headers(connection["token"], connection["header_name"], connection["prefix"]),
        timeout=connection.get("timeout", 30),
    )
    response.raise_for_status()
    raw_data: object = response.json()
    try:
        data = BotResponseInput.model_validate(raw_data)
    except ValidationError as exc:
        for error in exc.errors():
            location = error["loc"]
            if len(location) > 1 and location[0] == "retrieved" and isinstance(location[1], int):
                raise ValueError(
                    f"Invalid retrieved item at rank {location[1] + 1}: {error['msg']}"
                ) from exc
        raise ValueError(f"Invalid bot response: {validation_issue(exc)}") from exc
    return data.to_domain(response_keys=list(cast(dict[str, object], raw_data)))


def persisted_copy(run: Run, *, include_content: bool) -> Run:
    """Remove live question and response text from disk unless explicitly requested."""
    if include_content:
        return run
    redacted = copy.deepcopy(run)
    for result in redacted["results"]:
        _redact_identity(result)
        _redact_metric_details(result)
        _redact_reference_points(result)
        _redact_execution(result)
    redacted["content_redacted"] = True
    return redacted


def _redact_identity(result: Result) -> None:
    """Redact identity from a saved result."""
    if result["error"]:
        result["error"] = "[redacted]"
    result["case"]["name"] = "[redacted]"
    result["case"]["question"] = "[redacted]"
    result["case"]["expected_answer"] = "[redacted]"
    if result["case"].get("exact_match_references"):
        result["case"]["exact_match_references"] = ("[redacted]",)
    result["case"]["notes"] = "[redacted]"
    result["case"]["supporting_evidence"] = "[redacted]"
    if result["case"].get("trap_prior_answer"):
        result["case"]["trap_prior_answer"] = "[redacted]"
        result["case"]["trap_context_answer"] = "[redacted]"


def _redact_metric_details(result: Result) -> None:
    """Redact metric details from a saved result."""
    exact_match = result.get("exact_match_calculation")
    if exact_match:
        exact_match["normalized_answer"] = "[redacted]"
        exact_match["normalized_references"] = ["[redacted]"] * len(
            exact_match["normalized_references"]
        )
    token_f1 = result.get("token_f1_calculation")
    if token_f1:
        token_f1["answer_tokens"] = ["[redacted]"] * len(token_f1["answer_tokens"])
        token_f1["reference_tokens"] = [
            ["[redacted]"] * len(tokens) for tokens in token_f1["reference_tokens"]
        ]
    judgments = result.get("generation_judgments")
    if judgments:
        result["generation_judgments"] = {
            name: {
                "status": "redacted",
                "score": judgment.get("score"),
                "review_status": judgment.get("review_status"),
            }
            for name, judgment in judgments.items()
        }


def _redact_reference_points(result: Result) -> None:
    """Redact reference points from a saved result."""
    points = result["case"].get("reference_points", ())
    point_names = {point: f"[redacted point {index}]" for index, point in enumerate(points, 1)}
    if points:
        result["case"]["reference_points"] = tuple(point_names[point] for point in points)
        support = result["case"].get("point_support", {})
        result["case"]["point_support"] = {
            point_names[point]: ids for point, ids in support.items()
        }
        calculation = result.get("context_recall_calculation")
        if calculation:
            calculation["covered_points"] = [
                point_names[point] for point in calculation["covered_points"]
            ]
            calculation["missed_points"] = [
                point_names[point] for point in calculation["missed_points"]
            ]
            calculation["support_by_point"] = {
                point_names[point]: ids for point, ids in calculation["support_by_point"].items()
            }


def _redact_execution(result: Result) -> None:
    """Redact execution from a saved result."""
    trace = result.get("trace")
    if trace:
        trace["request"]["question"] = "[redacted]"
        for event in trace["events"]:
            if event["status"] == "error":
                event["detail"] = "[redacted]"
    response = result.get("response")
    if response:
        response["answer"] = "[redacted]"
        response["raw"] = {}
        for item in response["retrieved"]:
            item["text"] = "[redacted]"
            item["title"] = "[redacted]"
            item["source_url"] = ""
