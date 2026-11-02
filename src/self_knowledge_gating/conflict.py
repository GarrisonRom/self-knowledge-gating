"""Conflict detection and adjudication interfaces."""

from __future__ import annotations

from collections import defaultdict
from typing import Protocol, Sequence

from .schemas import Answer, Claim, ConflictAssessment, Evidence, claims_by_key


class ConflictAdjudicator(Protocol):
    def adjudicate(
        self,
        parametric_answer: Answer,
        grounded_answer: Answer,
        evidence: Sequence[Evidence],
    ) -> ConflictAssessment:
        """Detect and resolve conflicts between candidate claims and evidence."""


def _normalize(value: str) -> str:
    return " ".join(value.casefold().split())


class RuleBasedConflictAdjudicator:
    """Auditable source-aware conflict baseline.

    A conflict is resolved only when one evidence value has a clear weighted
    support margin. Otherwise the result remains unresolved and the policy can
    abstain rather than silently selecting a source.
    """

    def __init__(self, resolution_margin: float = 0.20) -> None:
        if not 0.0 <= resolution_margin <= 1.0:
            raise ValueError("resolution_margin must be in [0, 1]")
        self.resolution_margin = resolution_margin

    def adjudicate(
        self,
        parametric_answer: Answer,
        grounded_answer: Answer,
        evidence: Sequence[Evidence],
    ) -> ConflictAssessment:
        parametric = claims_by_key(parametric_answer.claims)
        grounded = claims_by_key(grounded_answer.claims)
        evidence_by_key: dict[str, list[Evidence]] = defaultdict(list)
        for item in evidence:
            if item.claim_key and item.claim_value is not None:
                evidence_by_key[item.claim_key].append(item)

        conflict_keys: set[str] = set()
        for key, claims in parametric.items():
            if key in grounded:
                parametric_values = {_normalize(claim.value) for claim in claims}
                grounded_values = {_normalize(claim.value) for claim in grounded[key]}
                if parametric_values.isdisjoint(grounded_values):
                    conflict_keys.add(key)

        for key, items in evidence_by_key.items():
            values = {_normalize(item.claim_value or "") for item in items}
            if len(values) > 1:
                conflict_keys.add(key)

        if not conflict_keys:
            return ConflictAssessment(reason="no conflicting claims detected")

        unresolved: list[str] = []
        resolved_keys: list[str] = []
        for key in sorted(conflict_keys):
            candidates = evidence_by_key.get(key, [])
            winner = self._evidence_winner(candidates)
            grounded_values = {_normalize(claim.value) for claim in grounded.get(key, ())}
            if winner is None or _normalize(winner) not in grounded_values:
                unresolved.append(key)
            else:
                resolved_keys.append(key)

        score = len(conflict_keys) / max(len(conflict_keys) + len(resolved_keys), 1)
        if unresolved:
            return ConflictAssessment(
                status="unresolved",
                score=min(1.0, score),
                conflicting_claims=tuple(unresolved),
                reason="evidence did not establish a clear, auditable winner",
            )
        return ConflictAssessment(
            status="resolved",
            score=min(1.0, score),
            conflicting_claims=tuple(sorted(conflict_keys)),
            winner="retrieval",
            reason="retrieved evidence had a clear weighted support margin",
        )

    def _evidence_winner(self, evidence: Sequence[Evidence]) -> str | None:
        weights: dict[str, float] = defaultdict(float)
        for item in evidence:
            if item.claim_value is None:
                continue
            value = _normalize(item.claim_value)
            weights[value] += item.reliability * item.freshness * max(item.relevance, 0.01)
        if not weights:
            return None
        ordered = sorted(weights.items(), key=lambda pair: pair[1], reverse=True)
        if len(ordered) == 1:
            return ordered[0][0]
        if ordered[0][1] - ordered[1][1] < self.resolution_margin:
            return None
        return ordered[0][0]
