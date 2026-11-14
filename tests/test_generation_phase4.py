"""Provide test generation phase4 utilities and tests."""

from copy import deepcopy
from typing import Any, cast

import pytest

from rag_eval.adapters import persisted_copy
from rag_eval.core import RagResponse, RetrievedItem, TestCase as EvalCase, make_run, token_f1
from rag_eval.fixtures import CASES, RESPONSES
from rag_eval.generation_fixtures import MODEL, MODEL_VERSION, fixture_judge
from rag_eval.generation_judge import (
    RUBRICS,
    evaluate_generation_run,
    evaluate_judge_metric,
    judge_payload,
    review_judgment,
    validate_judge_response,
    call_judge_endpoint,
)


@pytest.mark.parametrize(
    ("answer", "references", "score"),
    [
        ("The $60 per day", ("$60 per day",), 1.0),
        ("The current limit is $60 per day.", ("$60 per day",), 2 / 3),
        ("red red blue", ("red blue",), 0.8),
        ("different", ("red blue", "different"), 1.0),
        ("", ("red",), 0.0),
    ],
)
def test_token_f1(answer: str, references: tuple[str, ...], score: float) -> None:
    """Verify token f1."""
    assert token_f1(answer, references)["score"] == pytest.approx(score)


def test_token_f1_missing_and_invalid_labels() -> None:
    """Verify token f1 missing and invalid labels."""
    assert token_f1("answer", ())["status"] == "not_applicable"
    with pytest.raises(ValueError, match="empty after normalization"):
        token_f1("answer", ("the",))


def test_judge_fixture_demo_covers_every_metric_and_error_type() -> None:
    """Verify judge fixture demo covers every metric and error type."""
    run = make_run(CASES, RESPONSES, mode="offline", k=5, threshold=1, config={})
    judged = evaluate_generation_run(
        run, CASES, RESPONSES, fixture_judge, model=MODEL, model_version=MODEL_VERSION
    )
    by_id = {result["case"]["id"]: result for result in judged["results"]}
    judgments = {case_id: result.get("generation_judgments") for case_id, result in by_id.items()}
    assert all(value is not None for value in judgments.values())
    first = cast(dict[str, dict[str, Any]], judgments["CASE-001"])
    seventh = cast(dict[str, dict[str, Any]], judgments["CASE-007"])
    ninth = cast(dict[str, dict[str, Any]], judgments["CASE-009"])
    assert set(first) == set(RUBRICS)
    assert first["faithfulness"]["score"] == 1
    assert seventh["faithfulness"]["items"][0]["verdict"] == "contradicted"
    assert seventh["answer_correctness"]["items"][0]["verdict"] == "false"
    assert seventh["hallucination_despite_retrieval"]["score"] == 1
    assert ninth["faithfulness"]["items"][1]["verdict"] == "unsupported"
    summary = judged.get("generation_summary")
    assert summary is not None
    assert summary["false_confidence_rate"] == 0.5
    assert summary["false_abstention_count"] == 1
    assert first["answer_relevance"]["cosine_similarities"] == [1]


def test_unsupported_claim_is_distinct_from_explicit_contradiction() -> None:
    """Verify unsupported claim is distinct from explicit contradiction."""
    case = EvalCase("x", "x", "Q", (), "", "test")
    response = RagResponse("The limit is $90", (RetrievedItem("A", text="The limit is $60"),))
    payload = judge_payload("faithfulness", case, response)
    raw: dict[str, Any] = {
        "model": MODEL,
        "model_version": MODEL_VERSION,
        "prompt_version": "1.0.0",
        "rubric_version": "1.0.0",
        "items": [
            {
                "text": "The limit is $90",
                "verdict": "unsupported",
                "evidence_ids": [],
                "supporting_evidence": "",
                "explanation": "No cited support",
            }
        ],
    }
    unsupported = validate_judge_response(
        "faithfulness", raw, payload, model=MODEL, model_version=MODEL_VERSION
    )
    assert cast(list[dict[str, object]], unsupported["items"])[0]["verdict"] == "unsupported"
    cast(list[dict[str, object]], raw["items"])[0].update(
        {
            "verdict": "contradicted",
            "evidence_ids": ["A"],
            "supporting_evidence": "The limit is $60",
        }
    )
    contradicted = validate_judge_response(
        "faithfulness", raw, payload, model=MODEL, model_version=MODEL_VERSION
    )
    assert cast(list[dict[str, object]], contradicted["items"])[0]["verdict"] == "contradicted"


def test_judge_rejects_fabricated_quote_and_wrong_versions() -> None:
    """Verify judge rejects fabricated quote and wrong versions."""
    case = CASES[0]
    payload = judge_payload("faithfulness", case, RESPONSES[case.id])
    raw = fixture_judge(payload)
    bad = deepcopy(raw)
    cast(list[dict[str, object]], bad["items"])[0]["supporting_evidence"] = "invented passage"
    with pytest.raises(ValueError, match="not in cited"):
        validate_judge_response(
            "faithfulness", bad, payload, model=MODEL, model_version=MODEL_VERSION
        )
    bad = deepcopy(raw)
    bad["model_version"] = "wrong"
    with pytest.raises(ValueError, match="model/version"):
        validate_judge_response(
            "faithfulness", bad, payload, model=MODEL, model_version=MODEL_VERSION
        )


def test_judge_failure_is_not_quality_zero_and_human_review_is_explicit() -> None:
    """Verify judge failure is not quality zero and human review is explicit."""
    case = CASES[0]
    error = evaluate_judge_metric(
        "faithfulness",
        case,
        RESPONSES[case.id],
        lambda payload: {"items": []},
        model=MODEL,
        model_version=MODEL_VERSION,
    )
    assert error["status"] == "error" and error["score"] is None
    judged = evaluate_judge_metric(
        "faithfulness",
        case,
        RESPONSES[case.id],
        fixture_judge,
        model=MODEL,
        model_version=MODEL_VERSION,
    )
    assert judged["review_status"] == "provisional"
    assert "effective_score" not in judged
    rejected = review_judgment(
        judged,
        reviewer="reviewer-1",
        decision="reject",
        note="Claim decomposition missed a qualifier",
    )
    assert rejected["effective_score"] is None
    assert rejected["review_status"] == "human_rejected"
    accepted = review_judgment(
        judged,
        reviewer="reviewer-1",
        decision="accept",
        note="Checked against the retrieved sentence",
    )
    assert accepted["effective_score"] == 1


def test_live_redaction_removes_judge_content_and_token_strings() -> None:
    """Verify live redaction removes judge content and token strings."""
    run = make_run(
        [CASES[0]], {CASES[0].id: RESPONSES[CASES[0].id]}, mode="live", k=5, threshold=1, config={}
    )
    judged = evaluate_generation_run(
        run,
        [CASES[0]],
        {CASES[0].id: RESPONSES[CASES[0].id]},
        fixture_judge,
        model=MODEL,
        model_version=MODEL_VERSION,
    )
    saved = persisted_copy(judged, include_content=False)["results"][0]
    assert "$60" not in str(saved)
    judgments = saved.get("generation_judgments")
    token_f1_result = saved.get("token_f1_calculation")
    assert judgments is not None and judgments["faithfulness"]["status"] == "redacted"
    assert token_f1_result is not None and token_f1_result["answer_tokens"][0] == "[redacted]"


@pytest.mark.parametrize(
    ("metric", "case_index", "expected"),
    [
        ("faithfulness", 0, 1.0),
        ("citation_precision", 6, 0.0),
        ("answer_relevance", 6, 1.0),
        ("answer_correctness", 6, 0.0),
        ("answer_completeness", 5, 0.0),
        ("abstention_correctness", 5, 0.0),
        ("hallucination_despite_retrieval", 6, 1.0),
        ("faithfulness", 8, 0.5),
        ("citation_precision", 8, 0.5),
        ("abstention_correctness", 7, 0.0),
    ],
)
def test_each_judge_metric_score(metric: str, case_index: int, expected: float) -> None:
    """Verify each judge metric score."""
    case = CASES[case_index]
    judgment = evaluate_judge_metric(
        metric, case, RESPONSES[case.id], fixture_judge, model=MODEL, model_version=MODEL_VERSION
    )
    assert judgment["score"] == expected
    assert judgment["model"] == MODEL
    assert judgment["prompt_version"] == "1.0.0"


def test_citation_judge_must_cover_cited_ids() -> None:
    """Verify citation judge must cover cited ids."""
    case = CASES[2]
    payload = judge_payload("citation_precision", case, RESPONSES[case.id])
    raw = {
        "model": MODEL,
        "model_version": MODEL_VERSION,
        "prompt_version": "1.0.0",
        "rubric_version": "1.0.0",
        "items": [
            {
                "text": "primary has 10 minutes",
                "verdict": "supported",
                "citation_id": "DOC-104",
                "evidence_ids": ["DOC-104"],
                "supporting_evidence": "within 10 minutes",
                "explanation": "Supports timing",
            }
        ],
    }
    with pytest.raises(ValueError, match="omits a cited chunk"):
        validate_judge_response(
            "citation_precision", raw, payload, model=MODEL, model_version=MODEL_VERSION
        )


def test_relevance_rejects_invalid_embeddings() -> None:
    """Verify relevance rejects invalid embeddings."""
    case = CASES[0]
    payload = judge_payload("answer_relevance", case, RESPONSES[case.id])
    raw = fixture_judge(payload)
    raw["generated_question_embeddings"] = [[0.0, 0.0]]
    with pytest.raises(ValueError, match="nonzero"):
        validate_judge_response(
            "answer_relevance", raw, payload, model=MODEL, model_version=MODEL_VERSION
        )


def test_correctness_uncertain_is_not_a_false_score() -> None:
    """Verify correctness uncertain is not a false score."""
    case = CASES[0]
    payload = judge_payload("answer_correctness", case, RESPONSES[case.id])
    raw = fixture_judge(payload)
    cast(list[dict[str, object]], raw["items"])[0]["verdict"] = "uncertain"
    judgment = validate_judge_response(
        "answer_correctness", raw, payload, model=MODEL, model_version=MODEL_VERSION
    )
    assert judgment["status"] == "not_applicable"
    assert judgment["score"] is None


def test_gateway_posts_versioned_payload_and_keeps_token_out_of_payload(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verify gateway posts versioned payload and keeps token out of payload."""
    captured: dict[str, object] = {}

    class FakeResponse:
        """Represent fake response data."""

        def raise_for_status(self) -> None:
            """Accept successful fake responses."""
            return None

        def json(self) -> dict[str, object]:
            """Json."""
            return {"model": "approved-model", "items": []}

    def fake_post(
        url: str, *, json: dict[str, object], headers: dict[str, str], timeout: int
    ) -> FakeResponse:
        captured.update({"url": url, "json": json, "headers": headers, "timeout": timeout})
        return FakeResponse()

    monkeypatch.setattr("rag_eval.generation_judge.requests.post", fake_post)
    payload = judge_payload("faithfulness", CASES[0], RESPONSES["CASE-001"])
    assert (
        call_judge_endpoint(
            payload, url="https://approved.example.test/judge", token="secret", timeout=12
        )["model"]
        == "approved-model"
    )
    assert captured["headers"] == {"Authorization": "Bearer secret"}
    assert captured["timeout"] == 12
    assert "secret" not in str(captured["json"])
