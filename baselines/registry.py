"""Single registry for reproducible baseline names and citations."""

BASELINES = {
    "always-answer": {
        "kind": "routing",
        "description": "closed-book answer on every question",
    },
    "always-rag": {
        "kind": "routing",
        "description": "retrieve on every question, then use the common verifier",
    },
    "uncalibrated-confidence": {
        "kind": "routing",
        "description": "route only on raw self-knowledge confidence",
    },
    "complexity-router": {
        "kind": "routing",
        "description": "route by query complexity, Adaptive-RAG style",
    },
    "conformal-abstention": {
        "kind": "routing",
        "description": "abstain from a pre-calibrated risk score",
    },
    "trust-parameter": {
        "kind": "conflict",
        "description": "always prefer parametric memory when claims conflict",
    },
    "trust-retrieval": {
        "kind": "conflict",
        "description": "always prefer retrieved evidence when claims conflict",
    },
    "majority-rule": {
        "kind": "conflict",
        "description": "prefer the most frequent evidence value",
    },
    "claim-level": {
        "kind": "conflict",
        "description": "source-aware claim-level adjudication used by the proposed system",
    },
}
