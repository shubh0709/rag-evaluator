"""Provide test datasets utilities and tests."""

from typing import cast
import pytest

from rag_eval.datasets import export_case_pack, import_case_pack
from rag_eval.fixtures import CASES, DOCUMENTS, RESPONSES


def test_case_pack_round_trip():
    """Verify case pack round trip."""
    cases, responses = import_case_pack(export_case_pack(CASES, RESPONSES), set(DOCUMENTS))
    assert cases == CASES
    assert responses["CASE-003"].retrieved == RESPONSES["CASE-003"].retrieved


def test_case_pack_rejects_unknown_reference():
    """Verify case pack rejects unknown reference."""
    pack = export_case_pack(CASES, RESPONSES)
    cast(list[dict[str, object]], pack["cases"])[0]["relevant_ids"] = ["MISSING"]
    with pytest.raises(ValueError, match="Unknown relevant ID"):
        import_case_pack(pack, set(DOCUMENTS))


def test_case_pack_rejects_invalid_grade():
    """Verify case pack rejects invalid grade."""
    pack = export_case_pack(CASES, RESPONSES)
    cast(list[dict[str, object]], pack["cases"])[0]["relevance_grades"] = {"DOC-101": -1}
    with pytest.raises(ValueError, match="Relevance grades"):
        import_case_pack(pack, set(DOCUMENTS))


def test_case_pack_rejects_nested_type_before_domain_conversion():
    """Verify case pack rejects nested type before domain conversion."""
    pack = export_case_pack(CASES, RESPONSES)
    cast(list[dict[str, object]], pack["cases"])[0]["context_relevance"] = {"DOC-101": "true"}
    with pytest.raises(ValueError, match="cases.0.context_relevance.DOC-101"):
        import_case_pack(pack, set(DOCUMENTS))
