import pytest

from rag_eval.core import RagResponse, RetrievedItem, TestCase as EvalCase, make_run, ndcg_at_k


def test_graded_ndcg_numeric_example() -> None:
    result = ndcg_at_k(["A", "X", "B"], {"A": 3, "B": 2, "X": 0}, 3)
    assert result["ranked_grades"] == [3, 0, 2]
    assert result["ideal_grades"] == [3, 2, 0]
    assert result["discounted_gains"] == pytest.approx([7, 0, 1.5])
    assert result["dcg"] == pytest.approx(8.5)
    assert result["ideal_dcg"] == pytest.approx(7 + 3 / 1.584962500721156)
    assert result["score"] == pytest.approx(0.95583058934618)


@pytest.mark.parametrize("retrieved,grades,k,expected", [
    (["A", "B"], {"A": 3, "B": 2}, 2, 1.0),
    (["B", "A"], {"A": 3, "B": 2}, 2, (3 + 7 / 1.584962500721156) / (7 + 3 / 1.584962500721156)),
    (["X"], {"A": 3}, 2, 0.0),
    ([], {"A": 3}, 2, 0.0),
    (["A", "A"], {"A": 3}, 2, 1.0),
    (["X", "A"], {"A": 3}, 1, 0.0),
])
def test_ndcg_cases(retrieved, grades, k, expected) -> None:
    assert ndcg_at_k(retrieved, grades, k)["score"] == pytest.approx(expected)


def test_duplicate_rank_gets_zero_gain() -> None:
    result = ndcg_at_k(["A", "A", "B"], {"A": 3, "B": 2}, 3)
    assert result["ranked_grades"] == [3, 0, 2]
    assert result["duplicate_ranks"] == [2]


def test_no_positive_grades_and_invalid_grades() -> None:
    assert ndcg_at_k(["A"], {}, 2)["status"] == "not_applicable"
    assert ndcg_at_k(["A"], {"A": 0}, 2)["score"] is None
    with pytest.raises(ValueError, match="K must"):
        ndcg_at_k(["A"], {"A": 1}, 0)
    with pytest.raises(ValueError, match="Relevance grades"):
        ndcg_at_k(["A"], {"A": True}, 1)


def test_run_records_ndcg_and_error_state() -> None:
    case = EvalCase("A", "A", "Question?", ("A",), "Answer", "test",
                    relevance_grades={"A": 3, "B": 2})
    run = make_run([case], {"A": RagResponse("Answer", (RetrievedItem("B"), RetrievedItem("A")))},
                   mode="offline", k=2, threshold=1.0, config={})
    result = run["results"][0]
    assert result["ndcg_calculation"]["score"] < 1
    assert result["ndcg_status"] == "failed"
    error = make_run([case], {"A": RuntimeError("No response")}, mode="offline", k=2,
                     threshold=1.0, config={})["results"][0]
    assert error["ndcg_calculation"] is None
    assert error["ndcg_status"] == "error"
