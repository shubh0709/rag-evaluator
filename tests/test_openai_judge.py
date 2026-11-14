"""Provide test openai judge utilities and tests."""

import json as json_module
from pathlib import Path
from typing import cast

import pytest

from rag_eval.fixtures import CASES, RESPONSES
from rag_eval.generation_judge import judge_payload, validate_judge_response
from rag_eval.openai_judge import load_openai_config, openai_judge


class FakeResponse:
    """Represent fake response data."""

    def __init__(self, data: dict[str, object]) -> None:
        self.data = data

    def raise_for_status(self) -> None:
        """Accept successful fake responses."""
        return None

    def json(self) -> dict[str, object]:
        """Json."""
        return self.data


def test_loads_only_openai_values_from_env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify loads only openai values from env."""
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_MODEL", raising=False)
    path = tmp_path / ".env"
    path.write_text("OPENAI_API_KEY=test-secret\nOPENAI_MODEL=gpt-6-luna\nOTHER_SECRET=ignore\n")
    assert load_openai_config(path) == ("test-secret", "gpt-6-luna")


def test_structured_response_judge_and_store_false(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify structured response judge and store false."""
    payload = judge_payload("faithfulness", CASES[0], RESPONSES["CASE-001"])
    seen: dict[str, object] = {}
    output = {
        "items": [
            {
                "text": "Current limit is $60 per day",
                "verdict": "supported",
                "evidence_ids": ["DOC-101"],
                "supporting_evidence": "up to $60 per day",
                "explanation": "Current policy states the limit",
                "citation_id": "",
            }
        ]
    }

    def fake_post(
        url: str, *, json: dict[str, object], headers: dict[str, str], timeout: int
    ) -> FakeResponse:
        seen.update({"url": url, "body": json, "headers": headers, "timeout": timeout})
        return FakeResponse(
            {
                "id": "resp-test",
                "model": "gpt-6-luna-snapshot",
                "status": "completed",
                "output": [
                    {
                        "type": "message",
                        "content": [{"type": "output_text", "text": json_module.dumps(output)}],
                    }
                ],
            }
        )

    monkeypatch.setattr("rag_eval.openai_judge.requests.post", fake_post)
    raw = openai_judge(payload, api_key="test-secret", model="gpt-6-luna", timeout=9)
    assert seen["url"] == "https://api.openai.com/v1/responses"
    body = cast(dict[str, object], seen["body"])
    assert body["store"] is False
    assert (
        cast(dict[str, object], cast(dict[str, object], body["text"])["format"])["strict"] is True
    )
    assert "test-secret" not in str(seen["body"])
    result = validate_judge_response(
        "faithfulness", raw, payload, model="gpt-6-luna", model_version="gpt-6-luna"
    )
    assert result["score"] == 1
    assert result["resolved_model_version"] == "gpt-6-luna-snapshot"


def test_relevance_uses_embeddings_endpoint(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify relevance uses embeddings endpoint."""
    payload = judge_payload("answer_relevance", CASES[0], RESPONSES["CASE-001"])
    calls: list[str] = []

    def fake_post(
        url: str, *, json: dict[str, object], headers: dict[str, str], timeout: int
    ) -> FakeResponse:
        del headers, timeout
        calls.append(url)
        if url.endswith("/responses"):
            output: dict[str, object] = {
                "items": [
                    {
                        "text": "limit",
                        "verdict": "relevant",
                        "evidence_ids": [],
                        "supporting_evidence": "",
                        "explanation": "Addresses limit",
                        "citation_id": "",
                    }
                ],
                "generated_questions": ["What is the meal allowance?"],
            }
            return FakeResponse(
                {
                    "id": "resp-test",
                    "model": "gpt-6-luna",
                    "status": "completed",
                    "output": [
                        {
                            "type": "message",
                            "content": [{"type": "output_text", "text": json_module.dumps(output)}],
                        }
                    ],
                }
            )
        assert json["input"] == [payload["question"], "What is the meal allowance?"]
        return FakeResponse(
            {
                "model": "text-embedding-3-small",
                "data": [
                    {"index": 0, "embedding": [1.0, 0.0]},
                    {"index": 1, "embedding": [0.8, 0.6]},
                ],
            }
        )

    monkeypatch.setattr("rag_eval.openai_judge.requests.post", fake_post)
    raw = openai_judge(payload, api_key="test-secret", model="gpt-6-luna")
    result = validate_judge_response(
        "answer_relevance", raw, payload, model="gpt-6-luna", model_version="gpt-6-luna"
    )
    assert calls == ["https://api.openai.com/v1/responses", "https://api.openai.com/v1/embeddings"]
    assert result["score"] == pytest.approx(0.8)


def test_refusal_or_incomplete_is_error(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify refusal or incomplete is error."""
    payload = judge_payload("faithfulness", CASES[0], RESPONSES["CASE-001"])

    def incomplete_post(
        url: str, *, json: dict[str, object], headers: dict[str, str], timeout: int
    ) -> FakeResponse:
        del url, json, headers, timeout
        return FakeResponse({"status": "incomplete", "output": []})

    monkeypatch.setattr("rag_eval.openai_judge.requests.post", incomplete_post)
    with pytest.raises(ValueError, match="not completed"):
        openai_judge(payload, api_key="test-secret", model="gpt-6-luna")
