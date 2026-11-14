"""Provide test mrr utilities and tests."""

import pytest

from rag_eval.core import (
    RagResponse,
    RetrievedItem,
    TestCase as EvalCase,
    make_run,
    reciprocal_rank,
    with_recomputed_mrr,
)


@pytest.mark.parametrize(
    "retrieved,relevant,expected_rank,expected_score",
    [
        (["A", "B"], ["A"], 1, 1.0),
        (["X", "B", "A"], ["A", "B"], 2, 0.5),
        (["X", "X", "A"], ["A"], 3, 1 / 3),
        (["X", "Y"], ["A"], None, 0.0),
        ([], ["A"], None, 0.0),
    ],
)
def test_reciprocal_rank(
    retrieved: list[str], relevant: list[str], expected_rank: int | None, expected_score: float
) -> None:
    """Verify reciprocal rank."""
    result = reciprocal_rank(retrieved, relevant)
    assert result["score"] == pytest.approx(expected_score)
    assert result["first_relevant_rank"] == expected_rank
    assert result["ranked_ids"] == retrieved


def test_unlabeled_case_is_not_applicable() -> None:
    """Verify unlabeled case is not applicable."""
    result = reciprocal_rank(["A"], [])
    assert result["status"] == "not_applicable"
    assert result["score"] is None


def test_run_mrr_uses_full_ranking_and_excludes_unscored_cases() -> None:
    """Verify run mrr uses full ranking and excludes unscored cases."""
    cases = [
        EvalCase(key, key, "Question?", labels, "Answer", "test")
        for key, labels in (
            ("first", ("A",)),
            ("third", ("B",)),
            ("missing", ("C",)),
            ("unlabeled", ()),
            ("error", ("D",)),
        )
    ]
    responses = {
        "first": RagResponse("Answer", (RetrievedItem("A"),)),
        "third": RagResponse(
            "Answer", (RetrievedItem("X"), RetrievedItem("Y"), RetrievedItem("B"))
        ),
        "missing": RagResponse("Answer", (RetrievedItem("X"),)),
        "unlabeled": RagResponse("Answer", (RetrievedItem("A"),)),
        "error": RuntimeError("No trace"),
    }
    run = make_run(cases, responses, mode="offline", k=1, threshold=0.5, config={})
    results = {result["case"]["id"]: result for result in run["results"]}
    rr = results["third"].get("reciprocal_rank_calculation")
    assert rr is not None and rr["score"] == pytest.approx(1 / 3)
    assert results["third"].get("reciprocal_rank_status") == "failed"
    assert results["unlabeled"].get("reciprocal_rank_status") == "not_applicable"
    assert results["error"].get("reciprocal_rank_status") == "error"
    summary = run.get("mrr_summary")
    assert summary is not None
    assert summary["score"] == pytest.approx((1 + 1 / 3 + 0) / 3)
    assert summary["scored_cases"] == 3
    assert summary["excluded_cases"] == 2


def test_mrr_unavailable_without_scored_cases() -> None:
    """Verify mrr unavailable without scored cases."""
    case = EvalCase("A", "A", "Question?", (), "Answer", "test")
    run = make_run(
        [case], {"A": RagResponse("Answer", ())}, mode="offline", k=1, threshold=1.0, config={}
    )
    summary = run.get("mrr_summary")
    assert summary is not None
    assert summary["score"] is None
    assert summary["scored_cases"] == 0


def test_old_saved_run_mrr_is_recomputed_without_mutating_artifact() -> None:
    """Verify old saved run mrr is recomputed without mutating artifact."""
    case = EvalCase("A", "A", "Question?", ("A",), "Answer", "test")
    run = make_run(
        [case],
        {"A": RagResponse("Answer", (RetrievedItem("X"), RetrievedItem("A")))},
        mode="offline",
        k=1,
        threshold=0.5,
        config={},
    )
    del run["mrr_summary"]
    del run["results"][0]["reciprocal_rank_calculation"]
    del run["results"][0]["reciprocal_rank_status"]
    displayed = with_recomputed_mrr(run)
    rr = displayed["results"][0].get("reciprocal_rank_calculation")
    summary = displayed.get("mrr_summary")
    assert rr is not None and rr["score"] == 0.5
    assert summary is not None and summary["score"] == 0.5
    assert displayed.get("mrr_recomputed") is True
    assert "mrr_summary" not in run
    assert "reciprocal_rank_calculation" not in run["results"][0]
