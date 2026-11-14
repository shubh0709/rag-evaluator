import pytest

from rag_eval.core import (RagResponse, RetrievedItem, TestCase as EvalCase,
                           average_precision, context_precision, context_recall, make_run)
from rag_eval.adapters import persisted_copy


def test_average_precision_counts_missing_relevant_ids_and_ignores_duplicate_hits() -> None:
    ap = average_precision(["X", "A", "A", "B"], ["A", "B", "C"])
    assert ap["relevant_ranks"] == [2, 4]
    assert ap["precision_at_relevant_ranks"] == [0.5, 0.5]
    assert ap["duplicate_ranks"] == [3]
    assert ap["score"] == pytest.approx(1 / 3)
    assert average_precision([], ["A"])["score"] == 0.0
    assert average_precision(["A"], [])["score"] is None


def test_map_is_mean_of_scored_cases_only() -> None:
    cases = [EvalCase(key, key, "Question", labels, "Answer", "test")
             for key, labels in (("a", ("A",)), ("b", ("B",)), ("c", ()), ("d", ("D",)))]
    responses = {"a": RagResponse("", (RetrievedItem("A"),)),
                 "b": RagResponse("", (RetrievedItem("X"), RetrievedItem("B"))),
                 "c": RagResponse("", ()), "d": RuntimeError("failed")}
    run = make_run(cases, responses, mode="offline", k=1, threshold=0.5, config={})
    assert run["map_summary"]["score"] == pytest.approx(0.75)
    assert run["map_summary"]["scored_cases"] == 2
    assert run["map_summary"]["excluded_cases"] == 2


def test_context_precision_is_rank_sensitive_and_distinct_from_precision_at_k() -> None:
    cp = context_precision(["X", "A", "B"], {"X": False, "A": True, "B": True})
    assert cp["precision_at_relevant_ranks"] == pytest.approx([0.5, 2 / 3])
    assert cp["score"] == pytest.approx((0.5 + 2 / 3) / 2)
    assert context_precision(["X"], {"X": False})["score"] == 0.0
    assert context_precision(["X"], {"A": True})["score"] is None
    assert context_precision(["A", "A"], {"A": True})["duplicate_ranks"] == [2]


def test_context_recall_counts_reference_points_with_any_supporting_context() -> None:
    points = ["p1", "p2", "p3"]
    support = {"p1": ("A", "B"), "p2": ("B",), "p3": ("C",)}
    cr = context_recall(["A", "X", "B"], points, support)
    assert cr["score"] == pytest.approx(2 / 3)
    assert cr["covered_points"] == ["p1", "p2"]
    assert cr["missed_points"] == ["p3"]
    assert cr["support_by_point"]["p1"] == ["A", "B"]
    assert context_recall([], points, support)["score"] == 0.0
    assert context_recall(["A"], [], {})["score"] is None


def test_run_preserves_context_labels_and_skips_missing_judgments() -> None:
    case = EvalCase("a", "a", "Question", ("A",), "Answer", "test",
                    context_relevance={"A": True}, reference_points=("fact",),
                    point_support={"fact": ("A",)})
    response = RagResponse("Answer", (RetrievedItem("A"), RetrievedItem("X")))
    result = make_run([case], {"a": response}, mode="offline", k=2,
                      threshold=0.5, config={})["results"][0]
    assert result["context_precision_status"] == "not_applicable"
    assert result["context_precision_calculation"]["unjudged_ids"] == ["X"]
    assert result["context_recall_calculation"]["score"] == 1.0
    assert result["case"]["point_support"] == {"fact": ("A",)}


def test_live_artifact_redacts_reference_point_text() -> None:
    case = EvalCase("a", "a", "Question", ("A",), "Answer", "test",
                    context_relevance={"A": True}, reference_points=("Secret fact",),
                    point_support={"Secret fact": ("A",)})
    run = make_run([case], {"a": RagResponse("Answer", (RetrievedItem("A"),))},
                   mode="live", k=1, threshold=1.0, config={})
    saved = persisted_copy(run, include_content=False)["results"][0]
    assert saved["case"]["reference_points"] == ("[redacted point 1]",)
    assert "Secret fact" not in str(saved)
