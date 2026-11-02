# Annotation schema

The development set is stored as JSONL. One line represents one question and its decision-relevant labels. Labels should be assigned independently of the system prediction.

| Field | Type | Meaning |
| --- | --- | --- |
| `sample_id` | string | Stable unique identifier |
| `question` | string | User question presented to the model |
| `reference_answer` | string/null | Gold answer when one is defined |
| `answerability` | enum | `answerable`, `conditionally_answerable`, `unanswerable`, `time_sensitive` |
| `ambiguity` | enum | `none`, `resolvable`, `material` |
| `retrieval_needed` | boolean | Whether external evidence is required for a reliable answer |
| `evidence_refs` | array | Source identifiers that can support the answer |
| `parametric_conflict` | enum | `none`, `possible`, `confirmed`, `not_applicable` |
| `risk_level` | enum | `low`, `medium`, `high` |
| `acceptable_actions` | array | Actions acceptable for this sample |
| `annotation_notes` | string | Rationale and edge-case notes |

Example:

```json
{
  "sample_id": "conflict-0001",
  "question": "Which policy version is currently in effect?",
  "reference_answer": "Version 3.2",
  "answerability": "time_sensitive",
  "ambiguity": "none",
  "retrieval_needed": true,
  "evidence_refs": ["policy-2026-03-01"],
  "parametric_conflict": "confirmed",
  "risk_level": "high",
  "acceptable_actions": ["retrieve", "abstain"],
  "annotation_notes": "The older version is intentionally present in the model-memory baseline."
}
```

For conflict cases, annotate the evidence version and timestamp separately in the source manifest. Do not place private documents or credentials in this repository.
