"""Hand-authored synthetic verdicts for exercising judge plumbing, not an LLM."""

from __future__ import annotations

from copy import deepcopy
from typing import Any, Unpack

from .generation_judge import PROMPT_VERSION, RUBRIC_VERSION
from .options import EvidenceOptions

MODEL = "synthetic-human-labels"
MODEL_VERSION = "1.0.0"


def item(
    text: str,
    verdict: str,
    explanation: str,
    **evidence: Unpack[EvidenceOptions],
) -> dict[str, Any]:
    """Build an evaluation item for a fixture."""
    result: dict[str, Any] = {
        "text": text,
        "verdict": verdict,
        "evidence_ids": [evidence.get("evidence_id", "")] if evidence.get("evidence_id") else [],
        "supporting_evidence": evidence.get("quote", ""),
        "explanation": explanation,
    }
    if evidence.get("citation_id"):
        result["citation_id"] = evidence.get("citation_id", "")
    return result


LABELS: dict[tuple[str, str], list[dict[str, Any]]] = {
    ("faithfulness", "What is the current daily meal limit for approved travel?"): [
        item(
            "Current limit is $60 per day",
            "supported",
            "The retrieved current policy states the limit.",
            evidence_id="DOC-101",
            quote="up to $60 per day",
        )
    ],
    ("faithfulness", "When does Project Orion launch?"): [
        item(
            "Project Orion launches in 2029",
            "contradicted",
            "The retrieved note explicitly says 2031, not 2029.",
            evidence_id="DOC-107",
            quote="17 November 2031, not in 2029",
        )
    ],
    ("faithfulness", "What approvals and tools are required for VPN registration?"): [
        item(
            "Use SSO and a hardware key",
            "supported",
            "DOC-103 states both requirements.",
            evidence_id="DOC-103",
            quote="SSO portal and a hardware key",
        ),
        item(
            "Manager approval is required",
            "unsupported",
            "DOC-103 does not mention approval; no conflicting statement is present.",
        ),
    ],
    ("citation_precision", "What is the current daily meal limit for approved travel?"): [
        item(
            "The current limit is $60 per day.",
            "supported",
            "DOC-101 supports the cited sentence.",
            evidence_id="DOC-101",
            quote="up to $60 per day",
            citation_id="DOC-101",
        )
    ],
    ("citation_precision", "When does Project Orion launch?"): [
        item(
            "Project Orion launches in 2029.",
            "contradicted",
            "DOC-107 contradicts the cited year.",
            evidence_id="DOC-107",
            quote="17 November 2031, not in 2029",
            citation_id="DOC-107",
        )
    ],
    ("citation_precision", "What approvals and tools are required for VPN registration?"): [
        item(
            "Use SSO and a hardware key.",
            "supported",
            "DOC-103 supports the tools.",
            evidence_id="DOC-103",
            quote="SSO portal and a hardware key",
            citation_id="DOC-103",
        ),
        item(
            "Manager approval is required.",
            "unsupported",
            "No retrieved passage supports approval.",
            evidence_id="DOC-103",
            citation_id="DOC-103",
        ),
    ],
    ("answer_relevance", "What is the current daily meal limit for approved travel?"): [
        item("The current limit is $60 per day.", "relevant", "Directly answers the asked limit.")
    ],
    ("answer_relevance", "When does Project Orion launch?"): [
        item(
            "Project Orion launches in 2029.",
            "relevant",
            "Addresses the date question, though the date is false.",
        )
    ],
    ("answer_correctness", "What is the current daily meal limit for approved travel?"): [
        item(
            "Current daily meal limit is $60", "correct", "Matches the independent reference point."
        )
    ],
    ("answer_correctness", "When does Project Orion launch?"): [
        item(
            "Orion launches on 17 November 2031",
            "false",
            "The answer says 2029 and conflicts with the reference.",
        )
    ],
    ("answer_completeness", "What is the current daily meal limit for approved travel?"): [
        item("Current daily meal limit is $60", "present", "The answer contains the limit.")
    ],
    ("answer_completeness", "What is the current meal limit?"): [
        item(
            "Current daily meal limit is $60",
            "absent",
            "The answer abstains despite available evidence.",
        )
    ],
    ("abstention_correctness", "What is the daily allowance for Mars travel?"): [
        item(
            "Mars allowance",
            "correct_abstention",
            "No corpus evidence exists and the answer says so.",
        )
    ],
    ("abstention_correctness", "What is the daily Mars travel allowance?"): [
        item(
            "Mars allowance",
            "false_confidence",
            "The corpus has no Mars allowance; the answer invents $500.",
        )
    ],
    ("abstention_correctness", "What is the current meal limit?"): [
        item(
            "Meal limit",
            "false_abstention",
            "The corpus states the limit, but the answer refuses.",
            evidence_id="DOC-101",
            quote="up to $60 per day",
        )
    ],
    ("abstention_correctness", "When does Project Orion launch?"): [
        item(
            "Orion launch",
            "correct_answer",
            "The answer attempts a date; its factual error is scored separately.",
        )
    ],
    ("hallucination_despite_retrieval", "When does Project Orion launch?"): [
        item(
            "Orion launch date",
            "followed_prior",
            "The answer gives 2029 despite the retrieved 2031 note.",
            evidence_id="DOC-107",
            quote="17 November 2031, not in 2029",
        )
    ],
}


def fixture_judge(payload: dict[str, Any]) -> dict[str, Any]:
    """Fixture judge."""
    key = (payload["metric"], payload["question"])
    if key not in LABELS:
        return {"skip": True}
    result: dict[str, Any] = {
        "model": MODEL,
        "model_version": MODEL_VERSION,
        "prompt_version": PROMPT_VERSION,
        "rubric_version": RUBRIC_VERSION,
        "items": deepcopy(LABELS[key]),
    }
    if payload["metric"] == "answer_relevance":
        result.update(
            {
                "generated_questions": [payload["question"]],
                "question_embedding": [1.0, 0.0],
                "generated_question_embeddings": [[1.0, 0.0]],
                "embedding_model": "synthetic-vector",
                "embedding_model_version": "1.0.0",
            }
        )
    return result
