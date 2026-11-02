"""Run a dependency-free pipeline demonstration with ``python -m``."""

from __future__ import annotations

import json
from dataclasses import asdict

from .conflict import RuleBasedConflictAdjudicator
from .generation import MetadataAnswerer
from .pipeline import SelfKnowledgeGatingPipeline
from .policy import GatePolicy
from .retrieval import InMemoryRetriever
from .schemas import Evidence, PipelineResult, Question
from .signals import MetadataSignalEstimator
from .verification import RuleBasedEvidenceVerifier


def _pipeline(corpus: tuple[Evidence, ...]) -> SelfKnowledgeGatingPipeline:
    return SelfKnowledgeGatingPipeline(
        signal_estimator=MetadataSignalEstimator(),
        retriever=InMemoryRetriever(corpus),
        answerer=MetadataAnswerer(),
        verifier=RuleBasedEvidenceVerifier(),
        adjudicator=RuleBasedConflictAdjudicator(),
        policy=GatePolicy(),
    )


def _render(result: PipelineResult) -> dict[str, object]:
    return {
        "route": result.route.value,
        "outcome": result.outcome.value,
        "reason_code": result.reason_code.value if result.reason_code else None,
        "answer": result.answer.text if result.answer else None,
        "clarification": result.clarification,
        "evidence": [item.evidence_id for item in result.evidence],
        "verification": asdict(result.verification) if result.verification else None,
        "conflict": asdict(result.conflict) if result.conflict else None,
        "usage": asdict(result.usage),
    }


def main() -> None:
    evidence = (
        Evidence(
            evidence_id="mp-149-band-gap",
            text="Materials Project mp-149 reports the band gap as 1.42 eV.",
            source_type="materials_project",
            claim_key="band_gap",
            claim_value="1.42 eV",
            reliability=0.95,
            freshness=0.95,
            relevance=0.95,
            source_version="demo-v1",
        ),
    )
    conflicting_evidence = (
        Evidence(
            evidence_id="source-a",
            text="Materials Project reports the band gap as 1.42 eV.",
            source_type="materials_project",
            claim_key="band_gap",
            claim_value="1.42 eV",
            reliability=0.90,
            freshness=0.90,
            relevance=0.90,
            source_version="demo-v1",
        ),
        Evidence(
            evidence_id="source-b",
            text="Materials Project reports the band gap as 2.10 eV.",
            source_type="external",
            claim_key="band_gap",
            claim_value="2.10 eV",
            reliability=0.90,
            freshness=0.90,
            relevance=0.90,
            source_version="demo-v1",
        ),
    )
    scenarios = [
        _pipeline(()).run(
            Question(
                text="What is the stored band gap?",
                question_id="known",
                metadata={
                    "boundary_state": "parametric_known",
                    "parametric_answer": "The stored value is 1.42 eV.",
                    "parametric_claims": [{"key": "band_gap", "value": "1.42 eV"}],
                },
            )
        ),
        _pipeline(evidence).run(
            Question(
                text="What is the Materials Project band gap?",
                question_id="retrieve",
                metadata={
                    "boundary_state": "uncertain",
                    "parametric_answer": "The stored value may be 1.10 eV.",
                    "parametric_claims": [{"key": "band_gap", "value": "1.10 eV"}],
                },
            )
        ),
        _pipeline(()).run(
            Question(
                text="What is the private batch identifier?",
                question_id="no-access",
                metadata={"boundary_state": "no_access", "access_allowed": False},
            )
        ),
        _pipeline(()).run(
            Question(
                text="Which value should be used?",
                question_id="ambiguous",
                metadata={
                    "boundary_state": "uncertain",
                    "ambiguity": 0.9,
                    "clarification_prompt": "请说明材料、属性和数据版本。",
                },
            )
        ),
        _pipeline(()).run(
            Question(
                text="What is the unknown property?",
                question_id="insufficient-evidence",
                metadata={"boundary_state": "uncertain"},
            )
        ),
        _pipeline(conflicting_evidence).run(
            Question(
                text="What is the Materials Project band gap?",
                question_id="conflict",
                metadata={
                    "boundary_state": "uncertain",
                    "parametric_claims": [{"key": "band_gap", "value": "1.10 eV"}],
                },
            )
        ),
    ]
    print(json.dumps([_render(result) for result in scenarios], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
