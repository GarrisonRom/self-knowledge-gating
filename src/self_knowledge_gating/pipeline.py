"""Main orchestration entry point for self-knowledge gating."""

from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter
from typing import Any, Mapping

from .conflict import ConflictAdjudicator
from .generation import Answerer
from .policy import GatePolicy
from .retrieval import Retriever
from .schemas import (
    Action,
    Answer,
    BoundarySignals,
    ConflictAssessment,
    Evidence,
    PipelineResult,
    Question,
    ReasonCode,
    ResourceUsage,
    VerificationResult,
)
from .signals import BoundarySignalEstimator
from .verification import EvidenceVerifier


@dataclass
class SelfKnowledgeGatingPipeline:
    """Coordinate estimation, routing, retrieval, verification and adjudication.

    The pipeline keeps the route and the user-visible outcome separate. A
    retrieval route can therefore end in an answer when evidence passes checks,
    or in an abstention when evidence is missing or conflict remains unresolved.
    """

    signal_estimator: BoundarySignalEstimator
    retriever: Retriever
    answerer: Answerer
    verifier: EvidenceVerifier
    adjudicator: ConflictAdjudicator
    policy: GatePolicy
    top_k: int = 5

    def __post_init__(self) -> None:
        if self.top_k <= 0:
            raise ValueError("top_k must be positive")

    def run(self, question: Question) -> PipelineResult:
        started = perf_counter()
        signals = self.signal_estimator.estimate(question)
        route_plan = self.policy.route(signals)
        route = route_plan.action
        base_metadata = {
            "route_reason_code": route_plan.reason_code.value if route_plan.reason_code else None,
            "dominant_boundary_state": signals.dominant_state.value,
            "retrieval_used": False,
        }

        if route == Action.CLARIFY:
            clarification = str(
                question.metadata.get(
                    "clarification_prompt",
                    "请补充会影响答案的关键条件。",
                )
            )
            return self._result(
                question=question,
                route=route,
                outcome=Action.CLARIFY,
                reason_code=route_plan.reason_code,
                signals=signals,
                clarification=clarification,
                started=started,
                metadata=base_metadata,
            )

        if route == Action.ABSTAIN:
            return self._result(
                question=question,
                route=route,
                outcome=Action.ABSTAIN,
                reason_code=route_plan.reason_code,
                signals=signals,
                started=started,
                metadata=base_metadata,
            )

        if route == Action.ANSWER:
            answer = self.answerer.generate(question, ())
            return self._result(
                question=question,
                route=route,
                outcome=Action.ANSWER,
                answer=answer,
                signals=signals,
                started=started,
                metadata=base_metadata,
            )

        parametric_answer = self.answerer.generate(question, ())
        evidence = tuple(self.retriever.search(question, top_k=self.top_k))
        grounded_answer = self.answerer.generate(question, evidence)
        verification = self.verifier.verify(grounded_answer, evidence)
        grounded_answer, verification, revised = self._revise_if_needed(
            question, grounded_answer, evidence, verification
        )
        conflict = self.adjudicator.adjudicate(parametric_answer, grounded_answer, evidence)
        final_plan = self.policy.finalize_retrieval(signals, evidence, verification, conflict)
        metadata = dict(base_metadata)
        metadata["retrieval_used"] = True
        metadata["parametric_answer"] = parametric_answer.text
        metadata["grounded_answer"] = grounded_answer.text
        metadata["grounded_answer_revised"] = revised

        return self._result(
            question=question,
            route=route,
            outcome=final_plan.action,
            reason_code=final_plan.reason_code,
            answer=grounded_answer if final_plan.action == Action.ANSWER else None,
            signals=signals,
            evidence=evidence,
            verification=verification,
            conflict=conflict,
            started=started,
            metadata=metadata,
        )

    def _revise_if_needed(self, question, answer, evidence, verification):
        """Give answerers with a revision hook one evidence-constrained retry."""
        if verification.evidence_sufficiency >= self.policy.thresholds.evidence_sufficiency:
            return answer, verification, False
        revise = getattr(self.answerer, "revise_grounded", None)
        if revise is None or not evidence:
            return answer, verification, False
        revised_answer = revise(question, answer, evidence, verification)
        revised_verification = self.verifier.verify(revised_answer, evidence)
        if revised_verification.evidence_sufficiency > verification.evidence_sufficiency:
            return revised_answer, revised_verification, True
        return answer, verification, False

    @staticmethod
    def _result(
        *,
        question: Question,
        route: Action,
        outcome: Action,
        started: float,
        reason_code: ReasonCode | None = None,
        answer: Answer | None = None,
        clarification: str | None = None,
        signals: BoundarySignals | None = None,
        evidence: tuple[Evidence, ...] = (),
        verification: VerificationResult | None = None,
        conflict: ConflictAssessment | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> PipelineResult:
        context_tokens = sum(item.context_tokens for item in evidence)
        output_tokens = len(answer.text.split()) if answer is not None else 0
        usage = ResourceUsage(
            retrieval_calls=1 if evidence else 0,
            context_tokens=context_tokens,
            output_tokens=output_tokens,
            latency_ms=(perf_counter() - started) * 1000.0,
        )
        return PipelineResult(
            question=question,
            route=route,
            outcome=outcome,
            reason_code=reason_code,
            answer=answer,
            clarification=clarification,
            signals=signals,
            evidence=tuple(evidence),
            verification=verification,
            conflict=conflict,
            usage=usage,
            metadata=metadata or {},
        )
