from rag_eval.adapters import path_get, persisted_copy, run_live_case
from rag_eval.core import TestCase as EvalCase, make_run
from rag_eval.fixtures import CASES, RESPONSES


def test_dotted_response_mapping():
    assert path_get({"data": {"hits": [{"id": "A"}]}}, "data.hits.0.id") == "A"
    assert path_get({}, "data.missing") is None


def test_live_adapter_maps_ranked_response(monkeypatch):
    class Response:
        def raise_for_status(self):
            pass
        def json(self):
            return {"data": {"answer": "Yes", "hits": [{"chunk": {"id": "A"}}]}}
    captured = {}
    def fake_post(url, json, headers, timeout):
        captured.update({"url": url, "json": json, "headers": headers})
        return Response()
    monkeypatch.setattr("rag_eval.adapters.requests.post", fake_post)
    result = run_live_case(EvalCase("T", "T", "Question?", ("A",), "Yes", "test"),
        url="https://bot.example.test/ask", token="secret", header_name="Authorization", prefix="Bearer",
        question_key="question", user_key="user", answer_path="data.answer",
        retrieval_path="data.hits", id_path="chunk.id", title_path="title", text_path="text",
        citation_path="data.citations")
    assert result.answer == "Yes"
    assert result.retrieved[0].id == "A"
    assert captured["json"] == {"question": "Question?", "user": "employee"}
    assert captured["headers"] == {"Authorization": "Bearer secret"}


def test_live_artifact_redacts_content():
    run = make_run(CASES[:1], RESPONSES, mode="live", k=5, threshold=1, config={})
    redacted = persisted_copy(run, include_content=False)
    assert redacted["results"][0]["case"]["question"] == "[redacted]"
    assert redacted["results"][0]["case"]["name"] == "[redacted]"
    assert redacted["results"][0]["response"]["answer"] == "[redacted]"
    assert redacted["results"][0]["response"]["retrieved"][0]["text"] == "[redacted]"
    assert redacted["results"][0]["response"]["retrieved"][0]["title"] == "[redacted]"
    assert run["results"][0]["case"]["question"] != "[redacted]"
