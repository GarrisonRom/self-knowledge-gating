"""Claim-level evidence verification."""

from __future__ import annotations

from typing import Protocol, Sequence

from .schemas import Answer, Evidence, VerificationResult


class EvidenceVerifier(Protocol):
    def verify(self, answer: Answer, evidence: Sequence[Evidence]) -> VerificationResult:
        """Measure how much of an answer is supported by retrieved evidence."""


def _normalize(value: str) -> str:
    return " ".join(value.casefold().split())


class RuleBasedEvidenceVerifier:
    """Exact claim matching baseline.

    Numeric unit conversion, tolerances and entailment models can be added behind
    this interface. Returning zero for an unstructured answer is intentional: the
    pipeline must not claim evidence support without an auditable claim object.
    """

    def verify(self, answer: Answer, evidence: Sequence[Evidence]) -> VerificationResult:
        if not answer.claims:
            return VerificationResult(
                evidence_sufficiency=0.0,
                supported_claims=0,
                total_claims=1,
                unsupported_claims=("unstructured_answer",),
            )

        supported = 0
        unsupported: list[str] = []
        for claim in answer.claims:
            matched = any(
                item.claim_key == claim.key
                and item.claim_value is not None
                and _normalize(item.claim_value) == _normalize(claim.value)
                for item in evidence
            )
            if matched:
                supported += 1
            else:
                unsupported.append(claim.key)
        total = len(answer.claims)
        return VerificationResult(
            evidence_sufficiency=supported / total,
            supported_claims=supported,
            total_claims=total,
            unsupported_claims=tuple(unsupported),
        )
