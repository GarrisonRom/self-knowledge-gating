"""Answerer interfaces used by the orchestration pipeline."""

from __future__ import annotations

from typing import Protocol, Sequence

from .schemas import Answer, Claim, Evidence, Question


class Answerer(Protocol):
    def generate(self, question: Question, evidence: Sequence[Evidence] = ()) -> Answer:
        """Generate an answer from parameter memory and optional evidence."""


class MetadataAnswerer:
    """Deterministic answerer for controlled fixtures.

    A real adapter can call an LLM and return the same ``Answer`` contract. The
    metadata adapter makes the pipeline runnable before a model provider exists.
    """

    def generate(self, question: Question, evidence: Sequence[Evidence] = ()) -> Answer:
        if evidence:
            structured = next(
                (item for item in evidence if item.claim_key and item.claim_value),
                None,
            )
            if structured is not None:
                claim = Claim(
                    key=structured.claim_key or "",
                    value=structured.claim_value or "",
                    source_refs=(structured.evidence_id,),
                )
                return Answer(
                    text=f"According to {structured.evidence_id}, {claim.key} is {claim.value}.",
                    claims=(claim,),
                )
            return Answer(text=evidence[0].text)

        claims = tuple(
            Claim(
                key=str(item["key"]),
                value=str(item["value"]),
                source_refs=tuple(str(ref) for ref in item.get("source_refs", ())),
            )
            for item in question.metadata.get("parametric_claims", ())
        )
        answer_text = str(question.metadata.get("parametric_answer", "No parametric answer configured."))
        return Answer(text=answer_text, claims=claims)
