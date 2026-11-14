"""Configurable HTTP boundaries. Tokens are passed at call time and never persisted."""
from __future__ import annotations

from typing import TypedDict, cast
from urllib.parse import urlparse

import requests

from .core import RagResponse, RetrievedItem, Run, TestCase


class RetrievedItemData(TypedDict):
    id: str
    title: str
    text: str
    source_url: str


class BotResponseData(TypedDict):
    answer: str
    retrieved: list[RetrievedItemData]
    citations: list[str]


def validate_url(url: str) -> None:
    parsed = urlparse(url)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        raise ValueError("Enter a full http(s) URL")


def auth_headers(token: str, header_name: str = "Authorization", prefix: str = "Bearer") -> dict[str, str]:
    if not token:
        return {}
    if "\n" in token or "\r" in token or "\n" in header_name or "\r" in header_name:
        raise ValueError("Invalid token or header name")
    return {header_name: f"{prefix} {token}".strip()}


def check_connection(url: str, *, token: str = "", header_name: str = "Authorization",
                     prefix: str = "Bearer", timeout: int = 10) -> dict[str, object]:
    validate_url(url)
    try:
        response = requests.get(url, headers=auth_headers(token, header_name, prefix), timeout=timeout)
        reachable = response.ok or response.status_code == 405
        message = ("Reachable" if response.ok else
                   "Reachable, but GET is not allowed" if response.status_code == 405 else
                   "Request failed")
        return {"ok": reachable, "status_code": response.status_code, "message": message}
    except requests.RequestException as exc:
        return {"ok": False, "status_code": None, "message": str(exc)}


def run_live_case(case: TestCase, *, url: str, token: str, header_name: str,
                  prefix: str, timeout: int = 30) -> RagResponse:
    validate_url(url)
    payload = {"question": case.question, "user_id": case.user_id}
    response = requests.post(url, json=payload, headers=auth_headers(token, header_name, prefix), timeout=timeout)
    response.raise_for_status()
    raw_data: object = response.json()
    if not isinstance(raw_data, dict) or not isinstance(raw_data.get("answer"), str):
        raise ValueError("Bot response must contain an answer string")
    retrieved = raw_data.get("retrieved")
    if not isinstance(retrieved, list):
        raise ValueError("Bot response must contain a ranked retrieved list")
    for rank, entry in enumerate(retrieved, 1):
        if (not isinstance(entry, dict) or not isinstance(entry.get("id"), str)
                or not entry["id"] or not all(isinstance(entry.get(key), str)
                                               for key in ("title", "text", "source_url"))):
            raise ValueError(f"Retrieved item at rank {rank} must have string id, title, text, and source_url")
    citations = raw_data.get("citations")
    if not isinstance(citations, list) or not all(isinstance(citation, str) for citation in citations):
        raise ValueError("Bot response must contain a citations string list")
    data: BotResponseData = cast(BotResponseData, raw_data)
    items = tuple(RetrievedItem(item["id"], item["title"], item["text"], item["source_url"])
                  for item in data["retrieved"])
    return RagResponse(data["answer"], items, tuple(data["citations"]), raw={"response_keys": list(data)})


def persisted_copy(run: Run, *, include_content: bool) -> Run:
    """Remove live question and response text from disk unless explicitly requested."""
    if include_content:
        return run
    import copy
    redacted = copy.deepcopy(run)
    for result in redacted["results"]:
        result["case"]["name"] = "[redacted]"
        result["case"]["question"] = "[redacted]"
        result["case"]["expected_answer"] = "[redacted]"
        result["case"]["notes"] = "[redacted]"
        response = result.get("response")
        if response:
            response["answer"] = "[redacted]"
            response["raw"] = {}
            for item in response["retrieved"]:
                item["text"] = "[redacted]"
                item["title"] = "[redacted]"
                item["source_url"] = ""
    redacted["content_redacted"] = True
    return redacted
