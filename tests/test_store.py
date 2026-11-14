from pathlib import Path

import pytest

from rag_eval.core import RagResponse, RetrievedItem, TestCase as EvalCase, make_run
from rag_eval.store import RunStore


def test_saved_run_preserves_inputs_metrics_and_trace(tmp_path: Path) -> None:
    case = EvalCase("A", "Example", "Question?", ("DOC-A",), "Answer", "test")
    response = RagResponse("Answer", (RetrievedItem("DOC-A", document_id="DOC-A"),), ("DOC-A",))
    run = make_run([case], {"A": response}, mode="offline", k=1, threshold=1.0,
                   config={"source": "fixture"}, execution_times={"A": ("start", "finish")})
    store = RunStore(tmp_path)
    path = store.save(run)
    saved = next(item for item in store.list() if item["id"] == run["id"])
    assert path.exists()
    assert saved["schema_version"] == 2
    assert saved["dataset_version"] == run["dataset_version"]
    assert saved["config"] == {"source": "fixture"}
    assert saved["results"][0]["case"]["question"] == "Question?"
    precision = saved["results"][0].get("precision_calculation")
    assert precision is not None and precision["score"] == 1.0
    trace = saved["results"][0].get("trace")
    assert trace is not None and trace["finished_at"] == "finish"


def test_store_rejects_invalid_run_id(tmp_path: Path) -> None:
    case = EvalCase("A", "Example", "Question?", ("DOC-A",), "Answer", "test")
    run = make_run([case], {"A": RagResponse("Answer", ())}, mode="offline", k=1,
                   threshold=1.0, config={})
    run["id"] = "../../outside"
    with pytest.raises(ValueError, match="Invalid run ID"):
        RunStore(tmp_path).save(run)
