"""Boundary-signal estimators and a deterministic metadata baseline."""

from __future__ import annotations

from typing import Any, Callable, Mapping, Protocol

from .schemas import BoundarySignals, BoundaryState, Question


class BoundarySignalEstimator(Protocol):
    def estimate(self, question: Question) -> BoundarySignals:
        """Estimate the state before retrieval is attempted."""


def _one_hot(state: BoundaryState, confidence: float = 1.0) -> dict[BoundaryState, float]:
    remainder = (1.0 - confidence) / (len(BoundaryState) - 1)
    return {candidate: confidence if candidate == state else remainder for candidate in BoundaryState}


class MetadataSignalEstimator:
    """Deterministic baseline for controlled fixtures and pipeline integration.

    A model-backed estimator can implement the same protocol. This baseline reads
    optional labels from ``Question.metadata`` and otherwise stays conservative by
    assigning most probability to ``uncertain``.
    """

    def __init__(self, default_confidence: float = 0.55) -> None:
        if not 0.25 <= default_confidence <= 1.0:
            raise ValueError("default_confidence must be in [0.25, 1.0]")
        self.default_confidence = default_confidence

    def estimate(self, question: Question) -> BoundarySignals:
        metadata = question.metadata
        supplied_probs = metadata.get("boundary_probs")
        state_value = metadata.get("boundary_state")

        if supplied_probs is not None:
            boundary_probs = self._coerce_probs(supplied_probs)
            dominant = max(boundary_probs, key=boundary_probs.get)
        elif state_value is not None:
            dominant = BoundaryState(state_value)
            boundary_probs = _one_hot(dominant, self.default_confidence)
        else:
            dominant = BoundaryState.UNCERTAIN
            boundary_probs = _one_hot(dominant, self.default_confidence)

        access_allowed = bool(metadata.get("access_allowed", dominant != BoundaryState.NO_ACCESS))
        defaults = {
            BoundaryState.PARAMETRIC_KNOWN: (0.95, 0.95, -0.10),
            BoundaryState.UNCERTAIN: (0.40, 0.75, 0.50),
            BoundaryState.NO_ACCESS: (0.10, 0.00, 0.00),
            BoundaryState.STALE: (0.25, 0.85, 0.70),
        }
        self_knowledge, answerability, retrieval_utility = defaults[dominant]
        if not access_allowed:
            answerability = min(answerability, 0.25)
            retrieval_utility = 0.0

        return BoundarySignals(
            boundary_probs=boundary_probs,
            self_knowledge=float(metadata.get("self_knowledge", self_knowledge)),
            answerability=float(metadata.get("answerability", answerability)),
            evidence_sufficiency=float(metadata.get("evidence_sufficiency", 0.0)),
            conflict_score=float(metadata.get("conflict_score", 0.0)),
            retrieval_utility=float(metadata.get("retrieval_utility", retrieval_utility)),
            ambiguity=float(metadata.get("ambiguity", 0.0)),
            access_allowed=access_allowed,
        )

    @staticmethod
    def _coerce_probs(values: Mapping[BoundaryState | str, float]) -> dict[BoundaryState, float]:
        converted = {BoundaryState(key): float(value) for key, value in values.items()}
        missing = set(BoundaryState) - set(converted)
        if missing:
            raise ValueError(f"boundary_probs is missing: {sorted(state.value for state in missing)}")
        total = sum(converted.values())
        if total <= 0.0:
            raise ValueError("boundary_probs must have a positive sum")
        return {state: value / total for state, value in converted.items()}


class ProbeSignalEstimator:
    """Boundary signals derived from interpretability probes.

    ``probe_fn`` takes a Question and returns a dict with at least
    ``p_ik`` (calibrated self-knowledge probability) and ``semantic_entropy``
    (normalized sampling entropy in [0, 1]). ``no_access``/``stale`` mass comes
    from question metadata so controlled fixtures can still inject them; the
    remaining mass is split between ``parametric_known`` and ``uncertain``.
    """

    def __init__(
        self,
        probe_fn: Callable[[Question], Mapping[str, Any]],
        default_confidence: float = 0.55,
    ) -> None:
        if not 0.25 <= default_confidence <= 1.0:
            raise ValueError("default_confidence must be in [0.25, 1.0]")
        self.probe_fn = probe_fn
        self.default_confidence = default_confidence

    def estimate(self, question: Question) -> BoundarySignals:
        probes = dict(self.probe_fn(question))
        p_ik = float(probes.get("p_ik", 0.5))
        entropy = float(probes.get("semantic_entropy", {}).get("entropy", 0.5)
                        if isinstance(probes.get("semantic_entropy"), Mapping)
                        else probes.get("entropy", 0.5))
        p_ik = min(max(p_ik, 0.0), 1.0)
        entropy = min(max(entropy, 0.0), 1.0)

        metadata = question.metadata
        p_no_access = float(metadata.get("no_access_prob", 0.02))
        p_stale = float(metadata.get("stale_prob", 0.03))
        p_no_access = min(max(p_no_access, 0.0), 0.9)
        p_stale = min(max(p_stale, 0.0), 0.9)

        # remaining mass split by probe evidence
        remainder = max(1.0 - p_no_access - p_stale, 0.0)
        w_known = max(p_ik * (1.0 - entropy), 1e-4)
        w_uncertain = max((1.0 - p_ik) * (1.0 - 0.5 * entropy), 1e-4)
        p_known = remainder * w_known / (w_known + w_uncertain)
        p_uncertain = remainder * w_uncertain / (w_known + w_uncertain)

        boundary_probs = {
            BoundaryState.PARAMETRIC_KNOWN: p_known,
            BoundaryState.UNCERTAIN: p_uncertain,
            BoundaryState.NO_ACCESS: p_no_access,
            BoundaryState.STALE: p_stale,
        }
        self_knowledge = float(metadata.get(
            "self_knowledge", 0.25 + 0.70 * p_ik * (1.0 - entropy)))
        answerability = float(metadata.get("answerability", 0.75))
        retrieval_utility = float(metadata.get(
            "retrieval_utility", 0.85 * (1.0 - self_knowledge)))
        access_allowed = bool(metadata.get("access_allowed", True))

        return BoundarySignals(
            boundary_probs=boundary_probs,
            self_knowledge=self_knowledge,
            answerability=answerability,
            evidence_sufficiency=float(metadata.get("evidence_sufficiency", 0.0)),
            conflict_score=float(metadata.get("conflict_score", 0.0)),
            retrieval_utility=retrieval_utility,
            ambiguity=float(metadata.get("ambiguity", 0.0)),
            access_allowed=access_allowed,
        )
