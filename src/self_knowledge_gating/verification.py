"""Claim-level evidence verification."""

from __future__ import annotations

import re
from typing import Any, Protocol, Sequence

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


class EmbeddingEvidenceVerifier:
    """Lightweight claim-evidence support check with the dense encoder.

    For each atomic claim, encode ``"key: value"`` and score it against every
    retrieved evidence text (cosine, fine-tuned domain encoder). A claim is
    supported when its best evidence similarity crosses ``threshold``. This is
    the auditable light entailment check named by the README, not an LLM judge.

    ``last_details`` exposes the per-claim matrix for probe recording; the
    ``verification`` probe payload gains an optional ``claim_details`` field.
    """

    def __init__(
        self,
        encoder_path: str,
        threshold: float = 0.55,
        max_length: int = 512,
        recorder: Any = None,
        device: str = "cuda",
    ) -> None:
        self.encoder_path = encoder_path
        self.threshold = threshold
        self.max_length = max_length
        self.recorder = recorder
        self.device = device
        self.last_details: list[dict] = []
        self._tok = None
        self._model = None

    def _lazy_load(self) -> None:
        if self._model is not None:
            return
        import torch
        from transformers import AutoModel, AutoTokenizer

        self._tok = AutoTokenizer.from_pretrained(self.encoder_path)
        self._model = AutoModel.from_pretrained(
            self.encoder_path, dtype=torch.bfloat16, device_map=self.device)
        self._model.eval()

    def _encode(self, texts: list[str], max_length: int | None = None):
        import torch
        import torch.nn.functional as F

        self._lazy_load()
        limit = max_length or self.max_length
        outs = []
        for i in range(0, len(texts), 32):
            enc = self._tok(texts[i:i + 32], padding=True, truncation=True,
                            max_length=limit,
                            return_tensors="pt").to(self.device)
            with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                h = self._model(**enc).last_hidden_state[:, 0]
            outs.append(F.normalize(h.float(), dim=-1).cpu())
        return torch.cat(outs).numpy()

    def verify(self, answer: Answer,
               evidence: Sequence[Evidence]) -> VerificationResult:
        self.last_details = []
        if not answer.claims:
            return VerificationResult(0.0, 0, 1, ("unstructured_answer",))
        if not evidence:
            return VerificationResult(
                0.0, 0, len(answer.claims),
                tuple(c.key for c in answer.claims))

        claim_texts = [f"{c.key}: {c.value}" for c in answer.claims]
        # sentence-level matching: short claims vs whole abstracts is too coarse
        sent_rows: list[tuple[str, str]] = []  # (evidence_id, sentence)
        for e in evidence:
            for s in re.split(r"(?<=[.!?])\s+", e.text):
                w = len(s.split())
                if 5 <= w <= 80:
                    sent_rows.append((e.evidence_id, s))
        if not sent_rows:
            for e in evidence:
                sent_rows.append((e.evidence_id, e.text[:600]))
        ce = self._encode(claim_texts, max_length=128)
        se = self._encode([s for _, s in sent_rows], max_length=256)
        sims = ce @ se.T

        supported, unsupported = 0, []
        for i, claim in enumerate(answer.claims):
            j = int(sims[i].argmax())
            best = float(sims[i, j])
            ok = best >= self.threshold
            self.last_details.append({
                "claim_key": claim.key,
                "claim_value": claim.value,
                "best_evidence_id": sent_rows[j][0],
                "best_sentence": sent_rows[j][1][:160],
                "similarity": round(best, 4),
                "supported": ok,
            })
            if ok:
                supported += 1
            else:
                unsupported.append(claim.key)
        total = len(answer.claims)
        if self.recorder is not None and self.last_details:
            qid = getattr(self, "_current_question_id", "?")
            self.recorder.record(qid, "claim_support_detail",
                                 {"claims": self.last_details,
                                  "threshold": self.threshold})
        return VerificationResult(
            evidence_sufficiency=supported / total,
            supported_claims=supported,
            total_claims=total,
            unsupported_claims=tuple(unsupported),
        )
