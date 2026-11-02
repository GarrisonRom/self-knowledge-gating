# Research plan

## 1. Problem formulation

For a question `q`, the system estimates a state vector:

```text
s(q) = {
  self_knowledge,
  answerability,
  evidence_sufficiency,
  conflict_score,
  retrieval_utility,
  ambiguity
}
```

The policy selects one action from:

```text
a(q) in {answer, retrieve, clarify, abstain}
```

The action is evaluated jointly on answer quality, unsupported-answer risk, retrieval cost and the usefulness of clarification. A single raw model confidence is not treated as a sufficient estimate of any of these quantities.

## 2. Gate design

The initial policy is an auditable ordered policy rather than an opaque learned router:

1. Detect whether the question is materially ambiguous. If a missing constraint can change the answer, prefer `clarify`.
2. Estimate self-knowledge and answerability separately. Low answerability cannot be repaired by high verbal confidence.
3. Estimate retrieval utility from expected evidence gain, cost, latency and noise risk.
4. If retrieval is selected, score evidence for relevance, coverage, source reliability and freshness.
5. Compare the parametric candidate with retrieved evidence. Record each conflict and its resolution status.
6. Emit `answer` only when the required evidence and risk thresholds are met. Otherwise emit `abstain` or ask for clarification.

The thresholds in `configs/base.yaml` are placeholders for calibration experiments. They must be selected on a validation split and frozen before testing.

## 3. Calibration protocol

- Collect examples from known, unknown, stale, ambiguous and conflict-controlled slices.
- Keep calibration data separate from the final test set.
- Compare temperature scaling, isotonic regression and an uncalibrated baseline when sample size permits.
- Report ECE, Brier score, NLL and reliability diagrams for every signal that is interpreted probabilistically.
- Report calibration by slice; an acceptable aggregate score can hide failure on unknown or conflict cases.

## 4. Conflict arbitration

Each candidate claim should retain:

- claim text and normalized answer;
- source type: parameter memory, retrieved document or multiple retrieved documents;
- source identity, timestamp and reliability metadata;
- entailment or contradiction score;
- freshness and agreement indicators;
- final resolution: `supported`, `parameter_only`, `retrieval_only`, `conflicted` or `unresolved`.

The first baseline should use explicit rules based on source reliability, freshness and cross-source agreement. A learned resolver can be added only after the rule-based baseline has a complete audit trail. An unresolved high-risk conflict must lead to `abstain` or `clarify`, not a silent winner.

## 5. Experiment matrix

| Axis | Planned levels |
| --- | --- |
| Router | always-answer, always-RAG, random, uncalibrated gate, calibrated gate |
| Evidence | no retrieval, clean evidence, incomplete evidence, stale evidence, conflicting evidence |
| Question type | known, unknown, time-sensitive, ambiguous, high-risk |
| Calibration | none, temperature scaling, isotonic regression |
| Action set | answer only, answer/abstain, answer/retrieve/abstain, all four actions |

Every comparison should use the same model, prompt budget, retrieval budget and evaluation split where possible.

## 6. Metrics

### Answer quality

Use accuracy, EM/F1 or a task-specific judge. Preserve raw model outputs and evidence references so judged scores can be audited.

### Selective prediction

Report coverage-risk curves, AURC, risk at fixed coverage, abstention precision/recall and unsupported-answer rate.

### Retrieval efficiency

Report retrieval rate, unnecessary retrieval rate, token cost, latency and evidence hit rate. A lower retrieval rate is useful only when quality and risk remain acceptable.

### Clarification and conflict

Report clarification success after one turn, user-effort proxy, conflict resolution accuracy and unresolved-conflict rate.

## 7. Milestones

1. Freeze schema and build a manually auditable development slice.
2. Implement signal collection and a deterministic four-action baseline.
3. Add calibration and reliability reports.
4. Add a replaceable retriever and evidence snapshots.
5. Add conflict arbitration and controlled conflict evaluation.
6. Run ablations, sensitivity analysis and final test evaluation.

## 8. Risk register

- Self-reports may be well calibrated on familiar data but fail under distribution shift; report slice-level results.
- Retrieval can add plausible but unsupported text; evidence sufficiency must check entailment, not only relevance.
- A conflict score can hide source quality assumptions; preserve raw source metadata and disagreement details.
- Abstention can improve risk by refusing too much; report coverage and utility together.
- LLM-as-judge scores can be unstable; use deterministic metrics where possible and retain judge prompts and versions.
