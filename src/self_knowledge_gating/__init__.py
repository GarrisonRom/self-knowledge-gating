"""Calibrated self-knowledge gating pipeline."""

from .pipeline import SelfKnowledgeGatingPipeline
from .schemas import (
    Action,
    Answer,
    BoundarySignals,
    BoundaryState,
    Claim,
    Evidence,
    PipelineResult,
    Question,
    ReasonCode,
)

__all__ = [
    "Action",
    "Answer",
    "BoundarySignals",
    "BoundaryState",
    "Claim",
    "Evidence",
    "PipelineResult",
    "Question",
    "ReasonCode",
    "SelfKnowledgeGatingPipeline",
]

__version__ = "0.1.0"
