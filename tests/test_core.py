"""Provide test core utilities and tests."""

import pytest

from rag_eval.core import RagResponse, RetrievedItem, TestCase as EvalCase, make_run, recall_at_k
from rag_eval.fixtures import CASES, DOCUMENTS, RESPONSES


@pytest.mark.parametrize(
    "retrieved,relevant,k,expected",
    [
        (["A", "B"], ["A", "B"], 2, 1.0),
        (["A", "X", "B"], ["A", "B"], 2, 0.5),
        (["X", "Y"], ["A"], 2, 0.0),
        ([], ["A"], 5, 0.0),
        (["A", "A", "B"], ["A", "B"], 2, 0.5),
        (["A", "A", "B"], ["A", "B"], 3, 1.0),
    ],
)
def test_recall_at_k(retrieved: list[str], relevant: list[str], k: int, expected: float) -> None:
    """Verify recall at k."""
    assert recall_at_k(retrieved, relevant, k)["score"] == expected


def test_unlabeled_case_is_not_applicable():
    """Verify unlabeled case is not applicable."""
    result = recall_at_k(["A"], [], 5)
    assert result["score"] is None
    assert result["status"] == "not_applicable"


def test_invalid_k():
    """Verify invalid k."""
    with pytest.raises(ValueError, match="K must"):
        recall_at_k(["A"], ["A"], 0)


def test_synthetic_references_are_consistent():
    """Verify synthetic references are consistent."""
    for case in CASES:
        case.validate()
        assert set(case.relevant_ids).issubset(DOCUMENTS)
        if case.relevant_ids:
            assert case.supporting_evidence
            assert all(doc_id in case.supporting_evidence for doc_id in case.relevant_ids)
    for response in RESPONSES.values():
        assert {item.id for item in response.retrieved}.issubset(DOCUMENTS)


def test_run_records_calculation_and_errors():
    """Verify run records calculation and errors."""
    good = EvalCase("A", "A", "Question?", ("DOC-101",), "Answer", "test")
    bad = EvalCase("B", "B", "Question?", ("DOC-102",), "Answer", "test")
    response = RagResponse("Answer", (RetrievedItem("DOC-101"),))
    run = make_run(
        [good, bad],
        {"A": response, "B": RuntimeError("Bot timeout")},
        mode="offline",
        k=1,
        threshold=1.0,
        config={},
    )
    assert run["summary"] == {"passed": 1, "failed": 0, "not_applicable": 0, "error": 1}
    calculation = run["results"][0]["calculation"]
    assert calculation is not None
    assert calculation["numerator"] == 1
    assert run["results"][1]["error"] == "Bot timeout"
    assert run["dataset_version"]
