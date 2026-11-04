"""Inference-time routing baselines.

These policies share the production ``SelfKnowledgeGatingPipeline`` contract.
They intentionally change only the routing rule so comparisons can keep the
same model, corpus, prompt, verifier and resource budget.
"""

from __future__ import annotations

from self_knowledge_gating.policy import GatePolicy, GateThresholds, PolicyDecision
from self_knowledge_gating.schemas import Action, BoundarySignals, ReasonCode


class AlwaysAnswerPolicy(GatePolicy):
    name = "always-answer"

    def route(self, signals: BoundarySignals, question=None) -> PolicyDecision:
        del signals, question
        return PolicyDecision(Action.ANSWER)


class AlwaysRAGPolicy(GatePolicy):
    name = "always-rag"

    def route(self, signals: BoundarySignals, question=None) -> PolicyDecision:
        del signals, question
        return PolicyDecision(Action.RETRIEVE, ReasonCode.PARAMETRIC_UNCERTAINTY)


class UncalibratedConfidencePolicy(GatePolicy):
    """Route on raw self-knowledge only; ignores boundary decomposition."""

    threshold: float = 0.80
    name = "uncalibrated-confidence"

    def __init__(self, threshold: float = 0.80):
        super().__init__()
        self.threshold = threshold

    def route(self, signals: BoundarySignals, question=None) -> PolicyDecision:
        del question
        if signals.self_knowledge >= self.threshold:
            return PolicyDecision(Action.ANSWER)
        return PolicyDecision(Action.RETRIEVE, ReasonCode.PARAMETRIC_UNCERTAINTY)


class ComplexityRouterPolicy(GatePolicy):
    """No-retrieval for short queries and retrieval for complex queries."""

    token_threshold: int = 12
    name = "complexity-router"

    def __init__(self, token_threshold: int = 12):
        super().__init__()
        self.token_threshold = token_threshold

    def route(self, signals: BoundarySignals, question=None) -> PolicyDecision:
        complexity = None
        if question is not None:
            complexity = question.metadata.get("complexity")
            if complexity is None:
                complexity = "complex" if len(question.text.split()) > self.token_threshold else "simple"
        if complexity == "simple":
            return PolicyDecision(Action.ANSWER)
        return PolicyDecision(Action.RETRIEVE, ReasonCode.PARAMETRIC_UNCERTAINTY)


class ConformalAbstentionPolicy(GatePolicy):
    """Post-hoc abstention adapter using a pre-calibrated risk metadata field.

    A real conformal calibration step must produce ``conformal_risk`` on each
    question. Falling back to ``1 - self_knowledge`` is explicit and useful for
    smoke comparisons, but is not a conformal guarantee.
    """

    risk_target: float = 0.05
    name = "conformal-abstention"

    def __init__(self, risk_target: float = 0.05):
        super().__init__()
        self.risk_target = risk_target

    def route(self, signals: BoundarySignals, question=None) -> PolicyDecision:
        del question
        risk = 1.0 - signals.self_knowledge
        if risk > self.risk_target:
            return PolicyDecision(Action.ABSTAIN, ReasonCode.HIGH_RISK_UNSUPPORTED)
        return PolicyDecision(Action.ANSWER)


def build_policy(name: str, **kwargs) -> GatePolicy:
    normalized = name.casefold().replace("_", "-")
    if normalized in {"calibrated-boundary-gate", "gate", "production"}:
        return GatePolicy(GateThresholds(**kwargs)) if kwargs else GatePolicy()
    if normalized == "always-answer":
        return AlwaysAnswerPolicy()
    if normalized in {"always-rag", "always-retrieve"}:
        return AlwaysRAGPolicy()
    if normalized in {"uncalibrated-confidence", "confidence"}:
        return UncalibratedConfidencePolicy(**kwargs)
    if normalized in {"complexity-router", "adaptive-rag"}:
        return ComplexityRouterPolicy(**kwargs)
    if normalized in {"conformal-abstention", "conformal"}:
        return ConformalAbstentionPolicy(**kwargs)
    raise ValueError(f"unknown routing baseline: {name}")
