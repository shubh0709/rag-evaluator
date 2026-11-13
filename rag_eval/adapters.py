"""Configurable HTTP boundaries. Tokens are passed at call time and never persisted."""
from __future__ import annotations

from typing import Any
from urllib.parse import urlparse

import requests

from .core import RagResponse, RetrievedItem, TestCase


def validate_url(url: str) -> None:
    parsed = urlparse(url)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        raise ValueError("Enter a full http(s) URL")


def path_get(value: Any, path: str, default: Any = None) -> Any:
    if not path:
        return default
    for part in path.split("."):
        try:
            value = value[int(part)] if isinstance(value, list) else value[part]
        except (KeyError, IndexError, ValueError, TypeError):
            return default
    return value


def auth_headers(token: str, header_name: str = "Authorization", prefix: str = "Bearer") -> dict[str, str]:
    if not token:
        return {}
    if "\n" in token or "\r" in token or "\n" in header_name or "\r" in header_name:
        raise ValueError("Invalid token or header name")
    return {header_name: f"{prefix} {token}".strip()}


def check_connection(url: str, *, token: str = "", header_name: str = "Authorization",
                     prefix: str = "Bearer", timeout: int = 10) -> dict[str, Any]:
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
                  prefix: str, question_key: str, user_key: str, answer_path: str,
                  retrieval_path: str, id_path: str, title_path: str, text_path: str,
                  citation_path: str, timeout: int = 30) -> RagResponse:
    validate_url(url)
    payload = {question_key: case.question}
    if user_key:
        payload[user_key] = case.user_id
    response = requests.post(url, json=payload, headers=auth_headers(token, header_name, prefix), timeout=timeout)
    response.raise_for_status()
    data = response.json()
    if not isinstance(data, dict):
        raise ValueError("Bot response must be a JSON object")
    answer = path_get(data, answer_path)
    retrieved = path_get(data, retrieval_path)
    if not isinstance(answer, str) or not isinstance(retrieved, list):
        raise ValueError("Response mapping failed: answer string or ranked retrieval list missing")
    items = []
    for rank, entry in enumerate(retrieved, 1):
        item_id = path_get(entry, id_path)
        if not isinstance(item_id, str) or not item_id:
            raise ValueError(f"Retrieved item at rank {rank} has no string ID")
        items.append(RetrievedItem(item_id, str(path_get(entry, title_path, "") or ""),
                                   str(path_get(entry, text_path, "") or "")))
    citations = path_get(data, citation_path, [])
    if not isinstance(citations, list):
        citations = []
    return RagResponse(answer, tuple(items), tuple(str(c) for c in citations), raw={"response_keys": list(data)})


def persisted_copy(run: dict[str, Any], *, include_content: bool) -> dict[str, Any]:
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
