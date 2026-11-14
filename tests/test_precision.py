"""Provide test precision utilities and tests."""

import pytest

from rag_eval.core import TestCase as EvalCase, RagResponse, RetrievedItem, make_run, precision_at_k
from rag_eval.adapters import persisted_copy


@pytest.mark.parametrize(
    "retrieved,relevant,k,expected",
    [
        (["A", "B"], ["A", "B"], 2, 1.0),
        (["A", "X", "B"], ["A", "B"], 2, 0.5),
        (["X", "Y"], ["A"], 2, 0.0),
        ([], ["A"], 2, 0.0),
        (["A"], ["A"], 3, 1 / 3),
        (["A", "A", "B"], ["A", "B"], 3, 2 / 3),
        (["A", "A", "B"], ["A", "B"], 2, 0.5),
    ],
)
def test_precision_formula(
    retrieved: list[str], relevant: list[str], k: int, expected: float
) -> None:
    """Verify precision formula."""
    assert precision_at_k(retrieved, relevant, k)["score"] == pytest.approx(expected)


def test_precision_breakdown_identifies_duplicate_rank() -> None:
    """Verify precision breakdown identifies duplicate rank."""
    result = precision_at_k(["A", "A", "B"], ["A", "B"], 3)
    assert result["relevant_ranks"] == [1, 3]
    assert result["duplicate_ranks"] == [2]
    assert result["numerator"] == 2
    assert result["denominator"] == 3


def test_no_labels_and_invalid_k() -> None:
    """Verify no labels and invalid k."""
    assert precision_at_k(["A"], [], 2)["status"] == "not_applicable"
    assert precision_at_k(["A"], [], 2)["score"] is None
    with pytest.raises(ValueError, match="K must"):
        precision_at_k(["A"], ["A"], 0)


def test_run_preserves_both_metric_results_and_execution_trace() -> None:
    """Verify run preserves both metric results and execution trace."""
    case = EvalCase("A", "A", "Question?", ("A", "B"), "Answer", "test")
    response = RagResponse("Answer", (RetrievedItem("A"), RetrievedItem("X"), RetrievedItem("B")))
    run = make_run(
        [case],
        {"A": response},
        mode="offline",
        k=2,
        threshold=0.6,
        config={"fixture": "test"},
        execution_times={"A": ("start", "finish")},
    )
    result = run["results"][0]
    assert result["calculation"] is not None and result["calculation"]["score"] == 0.5
    precision = result.get("precision_calculation")
    assert precision is not None and precision["score"] == 0.5
    assert result["status"] == result.get("precision_status") == "failed"
    trace = result.get("trace")
    assert trace is not None
    assert trace["request"] == {"question": "Question?", "user_id": "employee"}
    assert trace["started_at"] == "start"
    assert trace["finished_at"] == "finish"
    assert [event["stage"] for event in trace["events"]] == ["request", "response", "metrics"]


def test_live_persistence_redacts_trace_request_and_error_detail() -> None:
    """Verify live persistence redacts trace request and error detail."""
    case = EvalCase("A", "A", "Secret question", ("A",), "Secret answer", "test")
    run = make_run(
        [case], {"A": RuntimeError("Secret response")}, mode="live", k=1, threshold=1.0, config={}
    )
    saved = persisted_copy(run, include_content=False)["results"][0]
    trace = saved.get("trace")
    assert trace is not None
    assert trace["request"]["question"] == "[redacted]"
    assert trace["events"][1]["detail"] == "[redacted]"
    assert saved["error"] == "[redacted]"


def test_live_persistence_redacts_supporting_evidence() -> None:
    """Verify live persistence redacts supporting evidence."""
    case = EvalCase(
        "A",
        "A",
        "Question",
        ("A",),
        "Answer",
        "test",
        supporting_evidence="Sensitive source passage",
    )
    run = make_run(
        [case],
        {"A": RagResponse("Answer", (RetrievedItem("A"),))},
        mode="live",
        k=1,
        threshold=1.0,
        config={},
    )
    saved = persisted_copy(run, include_content=False)["results"][0]
    assert saved["case"]["supporting_evidence"] == "[redacted]"


def test_invalid_response_is_error_not_a_quality_score() -> None:
    """Verify invalid response is error not a quality score."""
    case = EvalCase("A", "A", "Question", ("A",), "Answer", "test")
    invalid = RagResponse("Answer", (RetrievedItem(""),))
    result = make_run([case], {"A": invalid}, mode="offline", k=1, threshold=1.0, config={})[
        "results"
    ][0]
    assert result["status"] == result.get("precision_status") == "error"
    assert result["calculation"] is None and result.get("precision_calculation") is None


def test_case_contract_rejects_empty_relevance_id() -> None:
    """Verify case contract rejects empty relevance id."""
    case = EvalCase("A", "A", "Question", ("",), "Answer", "test")
    with pytest.raises(ValueError, match="Relevant IDs"):
        case.validate()
