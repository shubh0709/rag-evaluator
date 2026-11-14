"""Typed keyword options for evaluator, adapter, and judge calls."""

from __future__ import annotations

from collections.abc import Mapping
from typing import NotRequired, TypedDict


class RunOptions(TypedDict):
    """Configuration for a retrieval evaluation run."""

    mode: str
    k: int
    threshold: float
    config: dict[str, object]
    execution_times: NotRequired[Mapping[str, tuple[str, str]] | None]


class LiveConnectionOptions(TypedDict):
    """HTTP connection settings for one live case request."""

    url: str
    token: str
    header_name: str
    prefix: str
    timeout: NotRequired[int]


class JudgeIdentityOptions(TypedDict):
    """Configured model identity attached to a judgment."""

    model: str
    model_version: str


class GenerationRunOptions(JudgeIdentityOptions):
    """Judge identity and optional metrics for one generation run."""

    metrics: NotRequired[tuple[str, ...] | None]


class EvidenceOptions(TypedDict, total=False):
    """Optional cited evidence attached to a synthetic verdict."""

    evidence_id: str
    quote: str
    citation_id: str
