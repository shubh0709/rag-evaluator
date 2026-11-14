"""Provide test exact match utilities and tests."""

from typing import cast
import pytest

from rag_eval.adapters import persisted_copy
from rag_eval.core import RagResponse, TestCase as EvalCase, exact_match, make_run
from rag_eval.datasets import export_case_pack, import_case_pack
from rag_eval.fixtures import CASES, DOCUMENTS, RESPONSES


@pytest.mark.parametrize(
    ("answer", "references", "score"),
    [
        ("The $60, per day!", ("$60 per day",), 1.0),
        ("An SSO portal", ("SSO portal",), 1.0),
        ("a DIFFERENT answer", ("first", "different answer"), 1.0),
        ("The current limit is $60 per day.", ("$60 per day",), 0.0),
        ("", ("$60 per day",), 0.0),
    ],
)
def test_exact_match_formula(answer: str, references: tuple[str, ...], score: float) -> None:
    """Verify exact match formula."""
    assert exact_match(answer, references)["score"] == score


def test_exact_match_requires_explicit_usable_reference() -> None:
    """Verify exact match requires explicit usable reference."""
    assert exact_match("answer", ())["status"] == "not_applicable"
    with pytest.raises(ValueError, match="empty after normalization"):
        exact_match("answer", ("the",))
    with pytest.raises(ValueError, match="nonempty strings"):
        exact_match("answer", ("",))


def test_run_demo_and_trace() -> None:
    """Verify run demo and trace."""
    run = make_run(CASES, RESPONSES, mode="offline", k=5, threshold=1.0, config={})
    results = {result["case"]["id"]: result for result in run["results"]}
    assert results["CASE-001"].get("exact_match_status") == "failed"
    first = results["CASE-001"].get("exact_match_calculation")
    assert first is not None and first["normalized_answer"] == "current limit is 60 per day"
    assert results["CASE-002"].get("exact_match_status") == "passed"
    second = results["CASE-002"].get("exact_match_calculation")
    assert second is not None and second["matched_reference_index"] == 0
    assert results["CASE-003"].get("exact_match_status") == "not_applicable"
    trace = results["CASE-002"].get("trace")
    assert trace is not None and "Exact Match" in trace["events"][2]["detail"]


def test_error_and_live_redaction() -> None:
    """Verify error and live redaction."""
    case = EvalCase(
        "EM", "EM", "Question", (), "behavior", "test", exact_match_references=("Sensitive gold",)
    )
    errored = make_run(
        [case], {"EM": RuntimeError("failure")}, mode="live", k=1, threshold=1.0, config={}
    )["results"][0]
    assert errored.get("exact_match_status") == "error"
    run = make_run(
        [case],
        {"EM": RagResponse("Sensitive gold", ())},
        mode="live",
        k=1,
        threshold=1.0,
        config={},
    )
    saved = persisted_copy(run, include_content=False)["results"][0]
    assert saved["case"].get("exact_match_references") == ("[redacted]",)
    calculation = saved.get("exact_match_calculation")
    assert calculation is not None
    assert calculation["normalized_answer"] == "[redacted]"
    assert calculation["normalized_references"] == ["[redacted]"]


def test_case_pack_preserves_and_validates_exact_match_labels() -> None:
    """Verify case pack preserves and validates exact match labels."""
    pack = export_case_pack(CASES, RESPONSES)
    imported, _ = import_case_pack(pack, set(DOCUMENTS))
    assert imported[0].exact_match_references == ("$60 per day",)
    cast(list[dict[str, object]], pack["cases"])[0]["exact_match_references"] = ["the"]
    with pytest.raises(ValueError, match="empty after normalization"):
        import_case_pack(pack, set(DOCUMENTS))
