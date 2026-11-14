import pytest

from rag_eval.core import RagResponse, RetrievedItem, TestCase as EvalCase, make_run, reciprocal_rank, with_recomputed_mrr


@pytest.mark.parametrize("retrieved,relevant,expected_rank,expected_score", [
    (["A", "B"], ["A"], 1, 1.0),
    (["X", "B", "A"], ["A", "B"], 2, 0.5),
    (["X", "X", "A"], ["A"], 3, 1 / 3),
    (["X", "Y"], ["A"], None, 0.0),
    ([], ["A"], None, 0.0),
])
def test_reciprocal_rank(retrieved, relevant, expected_rank, expected_score) -> None:
    result = reciprocal_rank(retrieved, relevant)
    assert result["score"] == pytest.approx(expected_score)
    assert result["first_relevant_rank"] == expected_rank
    assert result["ranked_ids"] == retrieved


def test_unlabeled_case_is_not_applicable() -> None:
    result = reciprocal_rank(["A"], [])
    assert result["status"] == "not_applicable"
    assert result["score"] is None


def test_run_mrr_uses_full_ranking_and_excludes_unscored_cases() -> None:
    cases = [EvalCase(key, key, "Question?", labels, "Answer", "test")
             for key, labels in (("first", ("A",)), ("third", ("B",)),
                                 ("missing", ("C",)), ("unlabeled", ()), ("error", ("D",)))]
    responses = {
        "first": RagResponse("Answer", (RetrievedItem("A"),)),
        "third": RagResponse("Answer", (RetrievedItem("X"), RetrievedItem("Y"), RetrievedItem("B"))),
        "missing": RagResponse("Answer", (RetrievedItem("X"),)),
        "unlabeled": RagResponse("Answer", (RetrievedItem("A"),)),
        "error": RuntimeError("No trace"),
    }
    run = make_run(cases, responses, mode="offline", k=1, threshold=0.5, config={})
    results = {result["case"]["id"]: result for result in run["results"]}
    assert results["third"]["reciprocal_rank_calculation"]["score"] == pytest.approx(1 / 3)
    assert results["third"]["reciprocal_rank_status"] == "failed"
    assert results["unlabeled"]["reciprocal_rank_status"] == "not_applicable"
    assert results["error"]["reciprocal_rank_status"] == "error"
    assert run["mrr_summary"]["score"] == pytest.approx((1 + 1 / 3 + 0) / 3)
    assert run["mrr_summary"]["scored_cases"] == 3
    assert run["mrr_summary"]["excluded_cases"] == 2


def test_mrr_unavailable_without_scored_cases() -> None:
    case = EvalCase("A", "A", "Question?", (), "Answer", "test")
    run = make_run([case], {"A": RagResponse("Answer", ())}, mode="offline", k=1,
                   threshold=1.0, config={})
    assert run["mrr_summary"]["score"] is None
    assert run["mrr_summary"]["scored_cases"] == 0


def test_old_saved_run_mrr_is_recomputed_without_mutating_artifact() -> None:
    case = EvalCase("A", "A", "Question?", ("A",), "Answer", "test")
    run = make_run([case], {"A": RagResponse("Answer", (RetrievedItem("X"), RetrievedItem("A")))},
                   mode="offline", k=1, threshold=0.5, config={})
    del run["mrr_summary"]
    del run["results"][0]["reciprocal_rank_calculation"]
    del run["results"][0]["reciprocal_rank_status"]
    displayed = with_recomputed_mrr(run)
    assert displayed["results"][0]["reciprocal_rank_calculation"]["score"] == 0.5
    assert displayed["mrr_summary"]["score"] == 0.5
    assert displayed["mrr_recomputed"] is True
    assert "mrr_summary" not in run
    assert "reciprocal_rank_calculation" not in run["results"][0]
