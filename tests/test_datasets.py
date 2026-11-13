import pytest

from rag_eval.datasets import export_case_pack, import_case_pack
from rag_eval.fixtures import CASES, DOCUMENTS, RESPONSES


def test_case_pack_round_trip():
    cases, responses = import_case_pack(export_case_pack(CASES, RESPONSES), set(DOCUMENTS))
    assert cases == CASES
    assert responses["CASE-003"].retrieved == RESPONSES["CASE-003"].retrieved


def test_case_pack_rejects_unknown_reference():
    pack = export_case_pack(CASES, RESPONSES)
    pack["cases"][0]["relevant_ids"] = ["MISSING"]
    with pytest.raises(ValueError, match="Unknown relevant ID"):
        import_case_pack(pack, set(DOCUMENTS))
