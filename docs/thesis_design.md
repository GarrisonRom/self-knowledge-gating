# Master's thesis design: Materials-science case study

Working title:

> **科学问答场景下大语言模型可靠性的评测、归因与增强方法研究——以材料领域为例**

This document turns the project idea into an experiment plan. The core benchmark is designed to use structured data and deterministic evaluators, so the main labels do not depend on manual judgments.

## 1. Four research questions

### RQ1: Can a model distinguish four knowledge-boundary states?

For a fixed model checkpoint and a question, can the system distinguish:

1. `parametric_known`: the answer is available in parameter memory and is correct under the current reference snapshot;
2. `uncertain`: the question is answerable and accessible, but the model's answer or self-assessment is unstable or poorly calibrated;
3. `no_access`: a reliable answer exists only behind a withheld, private or permission-denied source;
4. `stale`: the parameter snapshot contains an older answer while the current authoritative source has changed.

The state is conditional on the model checkpoint, reference snapshot and access policy. It is not treated as an intrinsic label of a question. The test split must keep the checkpoint, source version and access condition fixed before the self-report is evaluated.

### RQ2: Does boundary-aware retrieval gating dominate fixed retrieval policies?

Under matched retrieval calls, context tokens and latency budgets, compare a calibrated four-state gate with `always-answer`, `always-RAG`, random routing and a single-score confidence gate. The primary output is a Pareto frontier, not one accuracy number.

The frontier should expose at least:

- coverage versus selective risk;
- normalized retrieval and context-token cost versus selective risk;
- p50/p95 latency versus coverage;
- unsupported-answer rate versus retrieval rate.

Each policy is evaluated over a threshold sweep. A point is retained when no other point is no worse on every reported resource and quality dimension and strictly better on at least one.

### RQ3: Is explicit conflict adjudication better than fixed trust?

Construct controlled cases where parameter memory and retrieved evidence are:

- consistent;
- numerically conflicting;
- entity-conflicting through formula or name substitution;
- temporally conflicting;
- mixed with irrelevant or misleading documents;
- split across multiple sources with no clear majority.

Compare `trust-parameter`, `trust-retrieval`, majority/score heuristics and an explicit claim-level adjudicator. The adjudicator must expose source, timestamp, reliability, entailment and resolution status. A silent choice is counted as an adjudication failure.

### RQ4: Can reason-coded abstention protect high-risk cases without over-refusal?

Evaluate `abstain` on unknown, stale, unresolved-conflict, insufficient-evidence and permission-denied cases. The response carries a reason code and a user-safe explanation, but never repeats protected content or infers the hidden answer. Report both safety and coverage:

- incorrect-answer rate;
- permission/secret leakage rate;
- abstention precision and recall;
- coverage at a fixed risk level;
- utility after one clarification or retrieval turn;
- distribution of reason codes by slice.

## 2. Unified decision model

For a question `q`, model checkpoint `m`, access policy `a`, and source snapshot `t`, the system records:

```text
state(q, m, a, t) = {
  boundary_probs: {
    parametric_known,
    uncertain,
    no_access,
    stale
  },
  ambiguity,
  evidence_sufficiency,
  conflict_score,
  retrieval_utility,
  risk_score
}
```

The action is one of `answer`, `retrieve`, `clarify` and `abstain`. The output contract also includes `reason_code`, `evidence_refs`, `source_versions`, `cost_tokens`, `retrieval_calls` and `latency_ms`.

Boundary signals are evaluated separately before they are combined. This prevents a high verbal confidence score from being mistaken for source access, freshness or evidence sufficiency.

## 3. Research content one: benchmark and automatic judgment

### 3.1 Materials Project source

Use Materials Project records and versioned snapshots as the structured source for material identity, formula and selected properties. The source manifest records API version or dump version, retrieval date, field precision, license and content hash.

### 3.2 Three required benchmark slices

1. **Common property questions**: formation energy, band gap, stability, density or other fields with clear units and deterministic values. These probe parametric knowledge.
2. **Long-tail material questions**: rare compositions, uncommon structures or properties with sparse training exposure. These measure where retrieval can add value.
3. **No-answer canaries**: nonexistent material identifiers, unsupported properties, withheld records and questions whose answer is intentionally unavailable under the access policy. Any fabricated answer is a failure.

Add temporal and conflict slices without changing the question text:

- `stale`: ask against a current snapshot while the closed-book baseline is fixed to an older snapshot;
- `conflict`: insert a controlled perturbation in a retrieved value, formula or source timestamp;
- `no_access`: remove or permission-deny the only source that contains the target value.

### 3.3 Automatic evaluation pipeline

1. Parse the model response into a typed claim object. LLM extraction is allowed only as a parser; it cannot decide correctness.
2. Normalize units and numerical formats using a deterministic conversion layer.
3. Apply a field-specific absolute or relative tolerance recorded in the manifest. Do not reuse one tolerance for all physical quantities.
4. Apply exact or canonicalized matching for categorical properties and material identifiers.
5. Compare every atomic claim with the versioned reference record and assign failure labels.

The core benchmark therefore needs no human gold labels. A small manual spot audit may be used to discover generator bugs, but audited examples are not silently mixed into the automatic gold set.

### 3.4 Failure taxonomy

- `retrieval_failure`: the target evidence was not returned or was inaccessible.
- `generation_use_failure`: evidence was present but the answer did not use it or contradicted it.
- `generation_copy_failure`: a supported value was copied with a unit, decimal or entity error.
- `generation_fabrication`: the answer contains an unsupported claim or invented material.
- `abstention_failure`: the system answered when it should abstain, or abstained on a supported low-risk case.
- `permission_failure`: the response reveals protected content or infers it from a denied source.

## 4. Research content two: factors and failure attribution

Use four system levels:

1. no retrieval;
2. naive RAG;
3. hybrid BM25 plus dense retrieval with reranking;
4. oracle retrieval, where the correct source is supplied but generation remains unchanged.

The oracle condition isolates generation and evidence-use failures from retrieval failures. It is not an upper bound on the entire system because context length and source ambiguity can still cause errors.

Run a preregistered factorial intervention over:

- evidence relevance: high versus low;
- evidence freshness: current versus stale;
- source agreement: agreement versus controlled conflict;
- context length and document count;
- evidence position in the context;
- parameter-memory availability;
- access permission.

For each cell, report retrieval miss, evidence-use failure, copying failure, fabrication and abstention failure separately. Do not attribute an error to generation when the target record never reached the context.

## 5. Research content three: repair of evidence-use failures

The first enhancement line targets evidence-use failure without retraining the main model:

1. decompose the response into atomic claims;
2. retrieve or align evidence for each claim;
3. run deterministic numeric and entity checks;
4. use an evidence-recite prompt and a citation constraint as separate interventions;
5. revise only unsupported or conflicting claims;
6. replace an expensive judge with a lightweight claim-evidence verifier where calibration permits.

All repair methods must be compared under matched extra-token and extra-latency budgets. Report the frontier shift, not only the best repaired accuracy.

## 6. Research content four: selective abstention

Candidate risk signals are:

- semantic entropy or sampled-answer disagreement;
- verbalized confidence after calibration;
- retrieval score and evidence coverage;
- claim-evidence contradiction;
- source freshness and conflict status;
- access and permission state.

Compare temperature scaling, isotonic regression and a conformal abstention rule. Calibration is fitted on a validation split and frozen before the final test split. The conformal rule should target a predeclared hallucination or unsupported-answer risk, while coverage remains a reported outcome.

Reason codes are part of the output contract:

```text
AMBIGUOUS_QUESTION
INSUFFICIENT_EVIDENCE
PARAMETRIC_UNCERTAINTY
STALE_PARAMETRIC_MEMORY
CONFLICT_UNRESOLVED
NO_AUTHORIZED_SOURCE
HIGH_RISK_UNSUPPORTED
```

The reason must be safe to display to a user. It should describe the decision state and next action, not disclose the hidden record or private source.

## 7. Pareto frontier protocol

For each router, sweep thresholds and retrieval depth. Record one row per run:

```text
model_id, data_version, access_policy, threshold_id,
coverage, selective_risk, unsupported_rate,
retrieval_calls, context_tokens, output_tokens,
latency_p50_ms, latency_p95_ms, abstention_rate
```

Normalize cost against the same model and hardware. Plot non-dominated points and report frontier area or hypervolume only as a secondary summary. The main claim must state where the proposed gate improves the frontier and where it does not.

## 8. Optional research-content five: modular knowledge storage

This is a doctoral-level extension, not required for the master's acceptance criteria. Compare parameter memory, RAG and a modular external knowledge store on update cost, factuality, forgetting after edits and conflict behavior. Keep it out of the core benchmark until the four-state boundary and frontier protocol are stable.

## 9. Acceptance criteria

The thesis is complete when it provides:

1. a versioned materials benchmark with deterministic judgments and canary cases;
2. a four-state boundary evaluation with calibration and slice-level results;
3. Pareto frontiers under matched retrieval/token/latency budgets;
4. a controlled comparison of explicit conflict adjudication and fixed-trust baselines;
5. a reason-coded abstention analysis covering safety, permission leakage and over-refusal;
6. a mechanism-level account that links observed failures to evidence access, freshness, conflict, context use or generation.
