# Mechanism-oriented literature review

Search date: 2026-11-02. Metadata was checked against arXiv, Crossref or ACL Anthology pages. The references below are used to form hypotheses and controls; they are not evidence that the proposed system will outperform any baseline.

## 1. Self-knowledge and uncertainty

| Reference | Mechanism contribution | Design implication |
| --- | --- | --- |
| Kadavath et al., **Language Models (Mostly) Know What They Know**, 2022. [arXiv:2207.05221](https://arxiv.org/abs/2207.05221) | Separates `P(True)` for a proposed answer from `P(IK)` for whether the model knows. Self-evaluation can be calibrated in familiar settings, while `P(IK)` generalizes less reliably to new tasks. | Treat answer correctness, knowledge availability and task transfer as separate labels. The proposed `parametric_known` signal cannot be only a verbal confidence score. |
| Kuhn, Gal & Farquhar, **Semantic Uncertainty: Linguistic Invariances for Uncertainty Estimation in Natural Language Generation**, 2023. [arXiv:2302.09664](https://arxiv.org/abs/2302.09664) | Semantic entropy groups different surface forms that express the same meaning, making uncertainty less sensitive to wording. | Use semantic uncertainty and sampled-answer disagreement as boundary signals, but evaluate them against deterministic material-property correctness. |
| Manakul, Liusie & Gales, **SelfCheckGPT: Zero-Resource Black-Box Hallucination Detection for Generative Large Language Models**, 2023. [arXiv:2303.08896](https://arxiv.org/abs/2303.08896) | Sampling inconsistency is a black-box signal for hallucination detection. | Include sample disagreement as one component of risk, not as a proxy for source access or freshness. |

### Mechanism synthesis

These papers imply at least three separable latent variables: whether the model can produce a correct answer, whether its sampled meanings are stable, and whether the answer remains valid under a new task or source snapshot. That motivates the four-state protocol instead of a one-dimensional confidence threshold. The missing test is an intervention that holds answerability fixed while independently changing access permission and source freshness.

## 2. Adaptive retrieval and evidence use

| Reference | Mechanism contribution | Design implication |
| --- | --- | --- |
| Asai et al., **Self-RAG: Learning to Retrieve, Generate, and Critique through Self-Reflection**, 2023. [arXiv:2310.11511](https://arxiv.org/abs/2310.11511) | Trains a model to retrieve on demand and emit reflection tokens for retrieval and generation critique. Fixed retrieval can hurt when passages are unnecessary or irrelevant. | Use adaptive retrieval as a baseline, but separate learned reflection from the explicit boundary-state gate proposed here. |
| Jeong et al., **Adaptive-RAG: Learning to Adapt Retrieval-Augmented Large Language Models through Question Complexity**, 2024. [NAACL DOI](https://doi.org/10.18653/v1/2024.naacl-long.389) | Routes questions to no retrieval, single-step retrieval or iterative retrieval using estimated question complexity. | Reproduce the routing idea under matched budgets, then test whether boundary state, freshness and access signals explain gains beyond complexity. |
| Gao et al., **RARR: Researching and Revising What Language Models Say, Using Language Models**, 2022. [arXiv:2210.08726](https://arxiv.org/abs/2210.08726) | Searches for attribution and revises unsupported output while preserving supported content. | Use claim-level attribution and revision as a repair condition for generation-use failure, not as a replacement for retrieval gating. |
| Ru et al., **RAGChecker: A Fine-grained Framework for Diagnosing Retrieval-Augmented Generation**, 2024. [arXiv:2408.08067](https://arxiv.org/abs/2408.08067) | Separates retrieval and generation diagnostics and reports fine-grained metrics for both modules. | Adopt the retrieval-versus-generation decomposition and add explicit access, freshness and conflict labels. |

### Mechanism synthesis

Adaptive retrieval work mainly routes by query complexity or learned reflection. It does not by itself identify whether a missing answer is uncertain, inaccessible or stale. RAGChecker and RARR support a claim-level audit, which is necessary for distinguishing retrieval failure from evidence-use failure and unsupported revision.

## 3. Knowledge conflict and source adjudication

| Reference | Mechanism contribution | Design implication |
| --- | --- | --- |
| Xie et al., **Adaptive Chameleon or Stubborn Sloth: Revealing the Behavior of Large Language Models in Knowledge Conflicts**, 2023. [arXiv:2305.13300](https://arxiv.org/abs/2305.13300) | Models can accept coherent external evidence, but also show confirmation bias when some context agrees with their parametric memory. | Conflict tests must vary coherence and partial agreement independently; a fixed trust rule cannot explain both behaviors. |
| Wang et al., **Resolving Knowledge Conflicts in Large Language Models**, 2023. [arXiv:2310.00935](https://arxiv.org/abs/2310.00935) | Defines conflict goals: detect that a conflict exists, identify the conflicting spans and produce distinct answers or viewpoints. | Evaluate conflict detection, localization and resolution separately. A single final-answer accuracy hides the failure mode. |
| Jin et al., **Tug-of-War Between Knowledge: Exploring and Resolving Knowledge Conflicts in Retrieval-Augmented Language Models**, 2024. [arXiv:2402.14409](https://arxiv.org/abs/2402.14409) | Reports Dunning-Kruger-like preference for faulty internal memory, availability bias, majority rule and confirmation bias; proposes conflict-disentangle contrastive decoding. | Build controlled numerical and entity conflicts, and compare explicit adjudication with trust-parameter, trust-retrieval and majority baselines. |
| Wang et al., **Astute RAG: Overcoming Imperfect Retrieval Augmentation and Knowledge Conflicts for Large Language Models**, 2024. [arXiv:2410.07176](https://arxiv.org/abs/2410.07176) | Treats imperfect retrieval and internal/external conflict jointly, using source-aware consolidation rather than blindly following context. | Include irrelevant, misleading and malicious evidence conditions, and make source reliability and freshness first-class inputs to adjudication. |

### Mechanism synthesis

The conflict literature predicts that the model's response is a function of both evidence quality and overlap with its prior. Therefore the correct intervention is not simply “trust retrieval more.” The thesis should estimate a claim-level reliability vector and test whether explicit adjudication changes the conflict-resolution frontier without increasing unsupported answers on clean evidence.

## 4. Selective prediction, abstention and risk control

| Reference | Mechanism contribution | Design implication |
| --- | --- | --- |
| Xin et al., **The Art of Abstention: Selective Prediction and Error Regularization for Natural Language Processing**, ACL 2021. [DOI](https://doi.org/10.18653/v1/2021.acl-long.84) | Studies abstention and confidence estimation for NLP, including an error-regularization method that improves confidence without a large computation increase. | Report risk-coverage curves and confidence quality, not only accuracy. |
| Yadkori et al., **Mitigating LLM Hallucinations via Conformal Abstention**, 2024. [arXiv:2405.01563](https://arxiv.org/abs/2405.01563) | Uses sampled-response similarity and conformal calibration to bound hallucination risk while controlling abstention. | Add a conformal rule as a statistical baseline, and test conditional guarantees separately on stale, no-access and conflict slices. |
| Srinivasan et al., **Selective “Selective Prediction”: Reducing Unnecessary Abstention in Vision-Language Reasoning**, 2024. [DOI](https://doi.org/10.18653/v1/2024.findings-acl.767) | Studies the cost of unnecessary abstention and the need to preserve answerable coverage. | Treat over-refusal as a first-class failure and optimize safety and coverage jointly. |

### Mechanism synthesis

Abstention is a decision problem over a calibrated risk signal, not a generic refusal prompt. The proposed reason codes add an audit dimension: the system should distinguish uncertainty, stale memory, missing authorization and unresolved conflict. The key test is whether each code maps to a measurable intervention and whether high-risk codes reduce leakage without collapsing coverage.

## 5. Materials science and automatic evaluation

| Reference | Mechanism contribution | Design implication |
| --- | --- | --- |
| Jain et al., **The Materials Project: A materials genome approach to accelerating materials innovation**, 2013. [DOI](https://doi.org/10.1063/1.4812323) | Provides a large, structured and versionable source of computed materials properties. | Use versioned Materials Project snapshots as the authoritative source for numeric and categorical checks. |
| Wan et al., **SciQAG: A Framework for Auto-Generated Science Question Answering Dataset with Fine-grained Evaluation**, 2024. [arXiv:2405.09939](https://arxiv.org/abs/2405.09939) | Shows a framework for automatically generating science QA and fine-grained evaluation from scientific literature. | Reuse the generator/evaluator separation while replacing open-ended LLM judging with deterministic property, unit and entity checks for the core benchmark. |

## 6. Mechanism-level gaps and proposed tests

### Gap A: four boundary states are not one confidence axis

**Prediction.** A model can be confident while having no authorized source or stale memory. Self-evaluation should fail selectively on access and freshness unless those variables are made explicit.

**Test.** Hold question wording and gold answer fixed; vary only access permission, source snapshot and context availability. Compare verbal confidence, semantic entropy, self-consistency and a boundary probe.

### Gap B: retrieval gains may be mediated by evidence use

**Prediction.** Oracle retrieval will not eliminate errors when the model ignores, miscopies or overgeneralizes evidence.

**Test.** Compare no retrieval, standard retrieval and oracle retrieval under fixed context budgets; classify residual errors with RAGChecker-style retrieval/generation labels.

### Gap C: conflict behavior depends on prior overlap and source reliability

**Prediction.** Coherent evidence that agrees with part of the parameter memory will be over-trusted or selectively assimilated; isolated conflicting evidence will be resisted. An explicit adjudicator should reduce this asymmetry.

**Test.** Use a crossed intervention of source reliability, freshness, agreement overlap and numerical distance. Compare fixed trust, majority and claim-level adjudication.

### Gap D: safe abstention needs a reason, not only a threshold

**Prediction.** A common abstention threshold either leaks on permission-denied cases or over-refuses clean, answerable cases. Reason-coded policies should improve conditional risk-coverage trade-offs.

**Test.** Evaluate leakage canaries, stale records, clean known questions and ambiguous questions under the same overall risk target; report conditional frontiers.

## 7. What this literature does not establish

The cited papers motivate mechanisms and baselines. They do not establish that a four-state gate will outperform all RAG systems, that Materials Project QA transfers to other domains, or that reason codes are truthful explanations. Those claims require the controlled benchmark and Pareto analysis in `docs/thesis_design.md`.
