"""Versioned, auditable generation judgments. Scores remain provisional until reviewed."""

from __future__ import annotations

from collections.abc import Callable
from copy import deepcopy
from math import isfinite, sqrt
from typing import Any, Unpack, cast

import requests
from pydantic import ValidationError

from .adapters import auth_headers, validate_url
from .core import RagResponse, Run, TestCase
from .dto import JudgeResponseInput, validation_issue
from .options import GenerationRunOptions, JudgeIdentityOptions

RUBRIC_VERSION = "1.0.0"
PROMPT_VERSION = "1.0.0"


def _is_json_object(value: object) -> bool:
    return isinstance(value, dict)


VERDICTS = {
    "faithfulness": {"supported": 1.0, "unsupported": 0.0, "contradicted": 0.0},
    "citation_precision": {"supported": 1.0, "unsupported": 0.0, "contradicted": 0.0},
    "answer_relevance": {"relevant": 1.0, "partial": 0.5, "irrelevant": 0.0},
    "answer_correctness": {"correct": 1.0, "partly_correct": 0.5, "false": 0.0, "uncertain": None},
    "answer_completeness": {"present": 1.0, "partial": 0.5, "absent": 0.0},
    "abstention_correctness": {
        "correct_abstention": 1.0,
        "correct_answer": 1.0,
        "false_confidence": 0.0,
        "false_abstention": 0.0,
    },
    "hallucination_despite_retrieval": {
        "followed_context": 0.0,
        "followed_prior": 1.0,
        "other": 0.5,
    },
}
RUBRICS = {
    "faithfulness": (
        "Split the answer into atomic factual claims. Mark each supported "
        "only if retrieved evidence entails it, contradicted only if "
        "evidence explicitly conflicts, otherwise unsupported. Score "
        "supported claims / all claims."
    ),
    "citation_precision": (
        "Split the answer into sentence-citation pairs. Mark supported "
        "only if that cited chunk entails that sentence, contradicted "
        "only on explicit conflict, otherwise unsupported. Score "
        "supported pairs / all pairs. Missing citations are handled "
        "separately."
    ),
    "answer_relevance": (
        "Generate one or more questions that the answer would address. "
        "Return generated_questions, the original question_embedding, and "
        "generated_question_embeddings in the same embedding space. Local "
        "code computes mean cosine similarity, clipped at zero. Mark the "
        "relationship relevant, partial, or irrelevant and explain it; "
        "factual correctness is separate."
    ),
    "answer_correctness": (
        "Compare each factual answer point to the independent reference. "
        "Mark correct, partly_correct, false only with a conflicting "
        "reference, or uncertain when reference cannot decide. Do not "
        "equate lack of retrieved support with falsehood."
    ),
    "answer_completeness": (
        "For every independent reference point, mark present, partial, or "
        "absent in the answer. Score (present + 0.5 partial) / reference "
        "points."
    ),
    "abstention_correctness": (
        "Use the independently labeled answerable flag. Mark "
        "correct_abstention for unanswerable and abstained, "
        "correct_answer for answerable and answered, false_confidence for "
        "unanswerable and answered, or false_abstention for answerable "
        "and abstained."
    ),
    "hallucination_despite_retrieval": (
        "For a planted conflict, compare the answer to the retrieved "
        "context answer and likely prior answer. Mark followed_context, "
        "followed_prior, or other. Score 1 for following prior, 0 for "
        "following context, 0.5 for other pending review. Explain the "
        "exact retrieved passage and answer conflict."
    ),
}


def applicable(metric: str, case: TestCase, response: RagResponse) -> bool:
    """Determine whether a case has the inputs required by a metric."""
    if metric == "citation_precision":
        applicable_to_case = bool(response.citations and response.retrieved)
    elif metric == "answer_correctness":
        applicable_to_case = bool(case.reference_points or case.exact_match_references)
    elif metric == "answer_completeness":
        applicable_to_case = bool(case.reference_points)
    elif metric == "abstention_correctness":
        applicable_to_case = case.answerable is not None
    elif metric == "hallucination_despite_retrieval":
        applicable_to_case = bool(
            case.trap_context_answer and case.trap_prior_answer and response.retrieved
        )
    elif metric == "faithfulness":
        applicable_to_case = bool(response.answer.strip() and response.retrieved)
    else:
        applicable_to_case = bool(response.answer.strip())
    return applicable_to_case


def judge_payload(metric: str, case: TestCase, response: RagResponse) -> dict[str, Any]:
    """Judge payload."""
    if metric not in RUBRICS:
        raise ValueError(f"Unknown judge metric: {metric}")
    return {
        "metric": metric,
        "prompt_version": PROMPT_VERSION,
        "rubric_version": RUBRIC_VERSION,
        "instructions": (
            "Return only JSON with an items array. Each item needs text, verdict, "
            "evidence_ids, supporting_evidence, and explanation. Distinguish "
            "unsupported from contradicted or false. A score is provisional and "
            "must be checked against human labels. " + RUBRICS[metric]
        ),
        "question": case.question,
        "retrieved_evidence": [
            {"id": item.id, "title": item.title, "text": item.text} for item in response.retrieved
        ],
        "generated_answer": response.answer,
        "citations": list(response.citations),
        "reference_points": list(case.reference_points),
        "gold_answers": list(case.exact_match_references),
        "answerable": case.answerable,
        "trap_prior_answer": case.trap_prior_answer,
        "trap_context_answer": case.trap_context_answer,
    }


def _validate_judge_items(
    metric: str, items: list[dict[str, Any]], payload: dict[str, Any]
) -> None:
    """Check verdicts, evidence quotes, citations, and reference coverage."""
    evidence = {item["id"]: item["text"] for item in payload["retrieved_evidence"]}
    for item in items:
        if item["verdict"] not in VERDICTS[metric]:
            raise ValueError("Invalid judge item or verdict")
        if not all(identifier in evidence for identifier in item["evidence_ids"]):
            raise ValueError("Judge item cites unknown evidence")
        quote = item["supporting_evidence"].strip()
        if quote and not any(quote in evidence[identifier] for identifier in item["evidence_ids"]):
            raise ValueError("Judge supporting evidence is not in cited retrieved chunks")
        if (
            metric in {"faithfulness", "citation_precision"}
            and item["verdict"] in {"supported", "contradicted"}
            and not quote
        ):
            raise ValueError("Supported or contradicted claims need an evidence quote")
        if metric == "citation_precision":
            citation_id = item.get("citation_id")
            if citation_id not in payload["citations"] or citation_id not in item["evidence_ids"]:
                raise ValueError("Citation judgment must name its cited chunk")
    if (
        metric == "answer_completeness"
        and [item["text"] for item in items] != payload["reference_points"]
    ):
        raise ValueError("Completeness items must cover every reference point in order")
    if metric == "citation_precision" and not set(payload["citations"]).issubset(
        {item["citation_id"] for item in items}
    ):
        raise ValueError("Citation judgment omits a cited chunk")
    if metric == "abstention_correctness":
        _validate_abstention(items, payload["answerable"])


def _validate_abstention(items: list[dict[str, Any]], answerable: bool | None) -> None:
    """Check the single abstention verdict against the answerability label."""
    if len(items) != 1:
        raise ValueError("Abstention requires one decision")
    verdict = items[0]["verdict"]
    if answerable is False and verdict not in {"correct_abstention", "false_confidence"}:
        raise ValueError("Abstention verdict conflicts with answerability label")
    if answerable is True and verdict not in {"correct_answer", "false_abstention"}:
        raise ValueError("Abstention verdict conflicts with answerability label")


def _relevance_similarities(raw: dict[str, Any]) -> list[float]:
    """Validate embeddings and calculate clipped cosine similarities."""
    questions = raw.get("generated_questions")
    original = raw.get("question_embedding")
    generated = raw.get("generated_question_embeddings")
    if not isinstance(questions, list) or not isinstance(generated, list):
        raise ValueError("Answer relevance needs generated questions and matching embeddings")
    question_values = cast(list[object], questions)
    generated_values = cast(list[object], generated)
    if (
        not question_values
        or not all(isinstance(question, str) and question.strip() for question in question_values)
        or len(generated_values) != len(question_values)
    ):
        raise ValueError("Answer relevance needs generated questions and matching embeddings")
    if not isinstance(original, list):
        raise ValueError("Answer relevance embeddings have invalid dimensions or values")
    original_values = cast(list[object], original)
    if (
        not original_values
        or not all(
            isinstance(value, (int, float)) and not isinstance(value, bool) and isfinite(value)
            for value in original_values
        )
        or any(
            not isinstance(vector, list)
            or len(cast(list[object], vector)) != len(original_values)
            or not all(
                isinstance(value, (int, float)) and not isinstance(value, bool) and isfinite(value)
                for value in cast(list[object], vector)
            )
            for vector in generated_values
        )
    ):
        raise ValueError("Answer relevance embeddings have invalid dimensions or values")
    original = cast(list[float], original)
    generated = cast(list[list[float]], generated)
    questions = cast(list[str], questions)
    original_norm = sqrt(sum(value * value for value in original))
    similarities: list[float] = []
    for vector in generated:
        norm = sqrt(sum(value * value for value in vector))
        if not original_norm or not norm:
            raise ValueError("Answer relevance embeddings must be nonzero")
        similarities.append(
            max(
                0.0,
                min(1.0, sum(a * b for a, b in zip(original, vector)) / (original_norm * norm)),
            )
        )
    if (
        not isinstance(raw.get("embedding_model"), str)
        or not raw["embedding_model"]
        or not isinstance(raw.get("embedding_model_version"), str)
        or not raw["embedding_model_version"]
    ):
        raise ValueError("Answer relevance needs embedding model and version")
    return similarities


def validate_judge_response(
    metric: str, raw: dict[str, Any], payload: dict[str, Any], *, model: str, model_version: str
) -> dict[str, Any]:
    """Reject malformed verdicts and invented evidence; compute score outside the judge."""
    try:
        parsed = JudgeResponseInput.model_validate(raw)
    except ValidationError as exc:
        raise ValueError(f"Invalid judge response: {validation_issue(exc)}") from exc
    if parsed.model != model or parsed.model_version != model_version:
        raise ValueError("Judge model/version does not match configured identity")
    if parsed.prompt_version != PROMPT_VERSION or parsed.rubric_version != RUBRIC_VERSION:
        raise ValueError("Judge prompt/rubric version mismatch")
    items = [item.model_dump(exclude_none=True) for item in parsed.items]
    _validate_judge_items(metric, items, payload)
    scores = [VERDICTS[metric][item["verdict"]] for item in items]
    determined_scores = [value for value in scores if value is not None]
    score = sum(determined_scores) / len(determined_scores) if determined_scores else None
    if metric == "answer_relevance":
        similarities = _relevance_similarities(raw)
        score = sum(similarities) / len(similarities)
    else:
        similarities = []
    return {
        "status": "scored" if score is not None else "not_applicable",
        "score": score,
        "reason": "All reference comparisons uncertain" if score is None else "",
        "review_status": "provisional",
        "model": model,
        "model_version": model_version,
        "resolved_model_version": raw.get("resolved_model_version", model_version),
        "response_id": raw.get("response_id", ""),
        "prompt_version": PROMPT_VERSION,
        "rubric_version": RUBRIC_VERSION,
        "prompt": payload["instructions"],
        "question": payload["question"],
        "retrieved_evidence": payload["retrieved_evidence"],
        "generated_answer": payload["generated_answer"],
        "structured_response": deepcopy(raw),
        "items": deepcopy(items),
        "cosine_similarities": similarities,
        "undetermined_items": len(scores) - len(determined_scores),
        "evidence_scope": "retrieved chunks; exact generation context unavailable",
    }


def evaluate_judge_metric(
    metric: str,
    case: TestCase,
    response: RagResponse,
    judge: Callable[[dict[str, Any]], dict[str, Any]],
    **identity: Unpack[JudgeIdentityOptions],
) -> dict[str, Any]:
    """Evaluate one metric with a configured judge identity."""
    if not applicable(metric, case, response):
        return {
            "status": "not_applicable",
            "score": None,
            "reason": "Required answer, evidence, citation, or independent label is missing",
        }
    payload = judge_payload(metric, case, response)
    try:
        raw = judge(payload)
        if not _is_json_object(raw):
            raise ValueError("Judge must return a JSON object")
        if raw.get("skip") is True:
            return {
                "status": "not_applicable",
                "score": None,
                "reason": "No reviewed synthetic judgment for this case and metric",
            }
        return validate_judge_response(
            metric, raw, payload, model=identity["model"], model_version=identity["model_version"]
        )
    except (ValueError, TypeError, KeyError, requests.RequestException) as exc:
        return {"status": "error", "score": None, "reason": str(exc)}


def evaluate_generation_run(
    run: Run,
    cases: list[TestCase],
    responses: dict[str, RagResponse],
    judge: Callable[[dict[str, Any]], dict[str, Any]],
    **options: Unpack[GenerationRunOptions],
) -> Run:
    """Evaluate selected generation metrics and summarize abstention errors."""
    enriched = deepcopy(run)
    selected_metrics = options.get("metrics")
    if selected_metrics is None:
        selected_metrics = tuple(RUBRICS)
    if any(metric not in RUBRICS for metric in selected_metrics):
        raise ValueError("Unknown generation metric")
    by_id = {case.id: case for case in cases}
    for result in enriched["results"]:
        case_id = result["case"]["id"]
        if case_id not in responses or case_id not in by_id:
            continue
        result["generation_judgments"] = {
            metric: evaluate_judge_metric(
                metric,
                by_id[case_id],
                responses[case_id],
                judge,
                model=options["model"],
                model_version=options["model_version"],
            )
            for metric in selected_metrics
        }
    abstention = [
        result.get("generation_judgments", {}).get("abstention_correctness", {})
        for result in enriched["results"]
    ]
    false_confidence = sum(
        any(
            item["verdict"] == "false_confidence"
            for item in cast(list[dict[str, Any]], detail.get("items", []))
        )
        for detail in abstention
    )
    false_abstention = sum(
        any(
            item["verdict"] == "false_abstention"
            for item in cast(list[dict[str, Any]], detail.get("items", []))
        )
        for detail in abstention
    )
    unanswerable = sum(
        by_id[result["case"]["id"]].answerable is False
        and result.get("generation_judgments", {}).get("abstention_correctness", {}).get("status")
        == "scored"
        for result in enriched["results"]
        if result["case"]["id"] in by_id
    )
    answerable = sum(
        by_id[result["case"]["id"]].answerable is True
        and result.get("generation_judgments", {}).get("abstention_correctness", {}).get("status")
        == "scored"
        for result in enriched["results"]
        if result["case"]["id"] in by_id
    )
    enriched["generation_summary"] = {
        "false_confidence_rate": false_confidence / unanswerable if unanswerable else None,
        "false_confidence_count": false_confidence,
        "unanswerable_scored": unanswerable,
        "false_abstention_rate": false_abstention / answerable if answerable else None,
        "false_abstention_count": false_abstention,
        "answerable_scored": answerable,
    }
    return enriched


def review_judgment(
    judgment: dict[str, Any], *, reviewer: str, decision: str, note: str
) -> dict[str, Any]:
    """Record a human check; rejected scores cannot become effective scores."""
    if judgment.get("status") != "scored":
        raise ValueError("Only scored judgments can be reviewed")
    if decision not in {"accept", "reject"} or not reviewer.strip() or not note.strip():
        raise ValueError("Review requires decision, reviewer, and explanation")
    reviewed = deepcopy(judgment)
    reviewed["review_status"] = "human_accepted" if decision == "accept" else "human_rejected"
    reviewed["reviewer"] = reviewer.strip()
    reviewed["review_note"] = note.strip()
    reviewed["effective_score"] = reviewed["score"] if decision == "accept" else None
    return reviewed


def call_judge_endpoint(
    payload: dict[str, Any], *, url: str, token: str, timeout: int = 60
) -> dict[str, Any]:
    """POST the versioned contract to an explicitly configured approved judge gateway."""
    validate_url(url)
    response = requests.post(url, json=payload, headers=auth_headers(token), timeout=timeout)
    response.raise_for_status()
    data: object = response.json()
    if not isinstance(data, dict):
        raise ValueError("Judge endpoint must return a JSON object")
    return cast(dict[str, Any], data)
