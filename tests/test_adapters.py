"""Provide test adapters utilities and tests."""

import pytest

from rag_eval.adapters import persisted_copy, run_live_case
from rag_eval.core import TestCase as EvalCase, make_run, response_record
from rag_eval.fixtures import CASES, RESPONSES


def test_live_adapter_maps_ranked_response(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify live adapter maps ranked response."""
    fixture = RESPONSES["CASE-001"]

    class Response:
        """Represent response data."""

        def raise_for_status(self) -> None:
            """Accept successful fake responses."""
            return None

        def json(self) -> dict[str, object]:
            """Json."""
            record = response_record(fixture)
            return {
                "answer": record["answer"],
                "retrieved": record["retrieved"],
                "citations": record["citations"],
            }

    captured: dict[str, object] = {}

    def fake_post(
        url: str, json: dict[str, str], headers: dict[str, str], timeout: int
    ) -> Response:
        del timeout
        captured.update({"url": url, "json": json, "headers": headers})
        return Response()

    monkeypatch.setattr("rag_eval.adapters.requests.post", fake_post)
    result = run_live_case(
        EvalCase("T", "T", "Question?", ("DOC-101",), fixture.answer, "test"),
        url="https://bot.example.test/ask",
        token="secret",
        header_name="Authorization",
        prefix="Bearer",
    )
    assert result.answer == fixture.answer
    assert result.retrieved == fixture.retrieved
    assert result.citations == fixture.citations
    assert captured["json"] == {"question": "Question?", "user_id": "employee"}
    assert captured["headers"] == {"Authorization": "Bearer secret"}


def test_live_adapter_rejects_response_outside_fixture_contract(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verify live adapter rejects response outside fixture contract."""

    class Response:
        """Represent response data."""

        def raise_for_status(self) -> None:
            """Accept successful fake responses."""
            return None

        def json(self) -> dict[str, object]:
            """Json."""
            return {"answer": "Yes", "retrieved": [{"id": "DOC-101"}], "citations": []}

    def fake_post(
        url: str, *, json: dict[str, str], headers: dict[str, str], timeout: int
    ) -> Response:
        del url, json, headers, timeout
        return Response()

    monkeypatch.setattr("rag_eval.adapters.requests.post", fake_post)
    with pytest.raises(ValueError, match="rank 1"):
        run_live_case(
            CASES[0],
            url="https://bot.example.test/ask",
            token="",
            header_name="Authorization",
            prefix="Bearer",
        )


def test_live_adapter_rejects_string_citations_without_echoing_answer(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verify live adapter rejects string citations without echoing answer."""

    class Response:
        """Represent response data."""

        def raise_for_status(self) -> None:
            """Accept successful fake responses."""
            return None

        def json(self) -> dict[str, object]:
            """Json."""
            return {"answer": "private answer", "retrieved": [], "citations": "DOC-101"}

    def fake_post(
        url: str, *, json: dict[str, str], headers: dict[str, str], timeout: int
    ) -> Response:
        del url, json, headers, timeout
        return Response()

    monkeypatch.setattr("rag_eval.adapters.requests.post", fake_post)
    with pytest.raises(ValueError, match="citations") as error:
        run_live_case(
            CASES[0],
            url="https://bot.example.test/ask",
            token="",
            header_name="Authorization",
            prefix="Bearer",
        )
    assert "private answer" not in str(error.value)


def test_live_artifact_redacts_content():
    """Verify live artifact redacts content."""
    run = make_run(CASES[:1], RESPONSES, mode="live", k=5, threshold=1, config={})
    redacted = persisted_copy(run, include_content=False)
    assert redacted["results"][0]["case"]["question"] == "[redacted]"
    assert redacted["results"][0]["case"]["name"] == "[redacted]"
    response = redacted["results"][0]["response"]
    assert response is not None
    assert response["answer"] == "[redacted]"
    assert response["retrieved"][0]["text"] == "[redacted]"
    assert response["retrieved"][0]["title"] == "[redacted]"
    assert run["results"][0]["case"]["question"] != "[redacted]"
