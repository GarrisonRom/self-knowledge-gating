"""Typed contracts shared by the gating pipeline components."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from math import isclose
from typing import Any, Mapping, Sequence


class Action(str, Enum):
    """A route selected by the gate or a final user-visible outcome."""

    ANSWER = "answer"
    RETRIEVE = "retrieve"
    CLARIFY = "clarify"
    ABSTAIN = "abstain"


class BoundaryState(str, Enum):
    PARAMETRIC_KNOWN = "parametric_known"
    UNCERTAIN = "uncertain"
    NO_ACCESS = "no_access"
    STALE = "stale"


class ReasonCode(str, Enum):
    AMBIGUOUS_QUESTION = "AMBIGUOUS_QUESTION"
    INSUFFICIENT_EVIDENCE = "INSUFFICIENT_EVIDENCE"
    PARAMETRIC_UNCERTAINTY = "PARAMETRIC_UNCERTAINTY"
    STALE_PARAMETRIC_MEMORY = "STALE_PARAMETRIC_MEMORY"
    CONFLICT_UNRESOLVED = "CONFLICT_UNRESOLVED"
    NO_AUTHORIZED_SOURCE = "NO_AUTHORIZED_SOURCE"
    HIGH_RISK_UNSUPPORTED = "HIGH_RISK_UNSUPPORTED"


def _check_unit_interval(name: str, value: float) -> None:
    if not 0.0 <= value <= 1.0:
        raise ValueError(f"{name} must be in [0, 1], got {value!r}")


@dataclass(frozen=True)
class Question:
    text: str
    question_id: str | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.text.strip():
            raise ValueError("question text must not be empty")


@dataclass(frozen=True)
class Claim:
    key: str
    value: str
    source_refs: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not self.key.strip():
            raise ValueError("claim key must not be empty")
        if not self.value.strip():
            raise ValueError("claim value must not be empty")


@dataclass(frozen=True)
class Answer:
    text: str
    claims: tuple[Claim, ...] = ()

    def __post_init__(self) -> None:
        if not self.text.strip():
            raise ValueError("answer text must not be empty")


@dataclass(frozen=True)
class Evidence:
    evidence_id: str
    text: str
    source_type: str = "unknown"
    claim_key: str | None = None
    claim_value: str | None = None
    reliability: float = 0.5
    freshness: float = 0.5
    relevance: float = 0.0
    source_version: str | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.evidence_id.strip():
            raise ValueError("evidence_id must not be empty")
        if not self.text.strip():
            raise ValueError("evidence text must not be empty")
        for name, value in (
            ("reliability", self.reliability),
            ("freshness", self.freshness),
            ("relevance", self.relevance),
        ):
            _check_unit_interval(name, value)
        if (self.claim_key is None) != (self.claim_value is None):
            raise ValueError("claim_key and claim_value must be provided together")

    @property
    def context_tokens(self) -> int:
        return len(self.text.split())


@dataclass(frozen=True)
class BoundarySignals:
    boundary_probs: Mapping[BoundaryState | str, float]
    self_knowledge: float
    answerability: float
    evidence_sufficiency: float
    conflict_score: float
    retrieval_utility: float
    ambiguity: float = 0.0
    access_allowed: bool = True

    def __post_init__(self) -> None:
        normalized: dict[BoundaryState, float] = {}
        for state, probability in self.boundary_probs.items():
            normalized[BoundaryState(state)] = probability
            _check_unit_interval(f"boundary_probs[{state}]", probability)
        missing = set(BoundaryState) - set(normalized)
        if missing:
            raise ValueError(f"missing boundary probabilities: {sorted(m.value for m in missing)}")
        total = sum(normalized.values())
        if not isclose(total, 1.0, abs_tol=1e-6):
            raise ValueError(f"boundary probabilities must sum to 1, got {total}")
        object.__setattr__(self, "boundary_probs", normalized)
        for name, value in (
            ("self_knowledge", self.self_knowledge),
            ("answerability", self.answerability),
            ("evidence_sufficiency", self.evidence_sufficiency),
            ("conflict_score", self.conflict_score),
            ("ambiguity", self.ambiguity),
        ):
            _check_unit_interval(name, value)

    @property
    def dominant_state(self) -> BoundaryState:
        return max(self.boundary_probs, key=self.boundary_probs.get)


@dataclass(frozen=True)
class VerificationResult:
    evidence_sufficiency: float
    supported_claims: int
    total_claims: int
    unsupported_claims: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        _check_unit_interval("evidence_sufficiency", self.evidence_sufficiency)
        if self.supported_claims < 0 or self.total_claims < 0:
            raise ValueError("claim counts must be non-negative")
        if self.supported_claims > self.total_claims:
            raise ValueError("supported_claims cannot exceed total_claims")


@dataclass(frozen=True)
class ConflictAssessment:
    status: str = "none"
    score: float = 0.0
    conflicting_claims: tuple[str, ...] = ()
    winner: str | None = None
    reason: str = ""

    def __post_init__(self) -> None:
        if self.status not in {"none", "resolved", "unresolved"}:
            raise ValueError(f"unknown conflict status: {self.status!r}")
        _check_unit_interval("conflict score", self.score)


@dataclass(frozen=True)
class ResourceUsage:
    retrieval_calls: int = 0
    context_tokens: int = 0
    output_tokens: int = 0
    latency_ms: float = 0.0

    def __post_init__(self) -> None:
        if self.retrieval_calls < 0 or self.context_tokens < 0 or self.output_tokens < 0:
            raise ValueError("resource counts must be non-negative")
        if self.latency_ms < 0.0:
            raise ValueError("latency_ms must be non-negative")


@dataclass(frozen=True)
class PipelineResult:
    question: Question
    route: Action
    outcome: Action
    reason_code: ReasonCode | None = None
    answer: Answer | None = None
    clarification: str | None = None
    signals: BoundarySignals | None = None
    evidence: tuple[Evidence, ...] = ()
    verification: VerificationResult | None = None
    conflict: ConflictAssessment | None = None
    usage: ResourceUsage = field(default_factory=ResourceUsage)
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.outcome == Action.ANSWER and self.answer is None:
            raise ValueError("an answer outcome requires answer")
        if self.outcome == Action.CLARIFY and not (self.clarification or "").strip():
            raise ValueError("a clarify outcome requires clarification")
        if self.outcome != Action.ANSWER and self.answer is not None:
            raise ValueError("non-answer outcomes cannot expose an answer")


def claims_by_key(claims: Sequence[Claim]) -> dict[str, tuple[Claim, ...]]:
    grouped: dict[str, list[Claim]] = {}
    for claim in claims:
        grouped.setdefault(claim.key, []).append(claim)
    return {key: tuple(values) for key, values in grouped.items()}
