"""End-to-end run of the full gating research system.

Wires the real components together:
  ProbeSignalEstimator (probe-driven boundary signals)
  BGERetriever (fine-tuned bge-m3 encoder + bge-reranker-v2-m3)
  QwenAnswerer (local Qwen3-8B, claim extraction, logprob probes)
  RuleBasedEvidenceVerifier / RuleBasedConflictAdjudicator / GatePolicy

Every decision-relevant signal is persisted through ProbeRecorder into
results/runs/<run_id>/ (probes.jsonl, registry.json, PROBES.md, results.jsonl,
summary.json).

Usage:
  python scripts/run_e2e.py --n 30 --corpus 8000 [--hidden-states]
"""

from __future__ import annotations

import argparse
import json
import os
import time
from dataclasses import asdict
from typing import Any

from self_knowledge_gating.bge_retrieval import BGERetriever
from self_knowledge_gating.conflict import RuleBasedConflictAdjudicator
from self_knowledge_gating.llm import QwenAnswerer
from self_knowledge_gating.pipeline import SelfKnowledgeGatingPipeline
from self_knowledge_gating.policy import GatePolicy
from self_knowledge_gating.probes import PROBE_POINTS, ProbeRecorder
from self_knowledge_gating.schemas import Evidence, Question
from self_knowledge_gating.signals import ProbeSignalEstimator
from self_knowledge_gating.verification import EmbeddingEvidenceVerifier

ROOT = "/root/autodl-tmp/self-knowledge-gating"
MODELS = f"{ROOT}/models/modelscope/models"
FT_ENCODER = f"{ROOT}/models/finetuned/bge-m3-matsci-v1"
RERANKER = f"{MODELS}/BAAI--bge-reranker-v2-m3/snapshots/master"
LLM = f"{MODELS}/Qwen--Qwen3-8B/snapshots/master"
INTERIM = f"{ROOT}/data/interim/openalex_clean.jsonl"
TEST_PAIRS = f"{ROOT}/data/processed/pairs_test.jsonl"
QUESTION_CACHE = f"{ROOT}/data/processed/e2e_questions.jsonl"


def load_corpus(limit: int) -> list[Evidence]:
    evidence = []
    with open(INTERIM, encoding="utf-8") as f:
        for i, line in enumerate(f):
            if i >= limit:
                break
            r = json.loads(line)
            age = max(0, 2025 - int(r["year"] or 2025))
            freshness = max(0.3, 1.0 - 0.07 * age)
            reliability = min(0.98, 0.55 + (r["cited_by_count"] or 0) / 3000.0)
            evidence.append(Evidence(
                evidence_id=r["work_id"], text=r["abstract"],
                source_type="openalex",
                reliability=round(reliability, 3),
                freshness=round(freshness, 3),
                metadata={"venue": r["venue"], "year": r["year"],
                          "title": r["title"]},
            ))
    return evidence


def make_questions(answerer: QwenAnswerer, n: int) -> list[Question]:
    """Turn held-out paper titles into natural questions (cached)."""
    cache: dict[str, str] = {}
    if os.path.exists(QUESTION_CACHE):
        with open(QUESTION_CACHE, encoding="utf-8") as f:
            for line in f:
                r = json.loads(line)
                cache[r["work_id"]] = r["question"]
    rows = []
    with open(TEST_PAIRS, encoding="utf-8") as f:
        for line in f:
            rows.append(json.loads(line))
    rows = rows[:n]
    out = []
    for r in rows:
        wid = r["positive_id"]
        if wid not in cache:
            prompt = (
                "Turn the following materials-science paper title into one "
                "specific scientific question that the paper answers. Output "
                "only the question, with a question mark.\n\nTitle: "
                f"{r['query']}")
            text, _ = answerer._chat(prompt, greedy=True)
            q = text.strip().split("\n")[0].strip()
            if not q.endswith("?"):
                q = q.rstrip(".") + "?"
            cache[wid] = q
            with open(QUESTION_CACHE, "a", encoding="utf-8") as f:
                f.write(json.dumps({"work_id": wid, "question": q,
                                    "title": r["query"]},
                                   ensure_ascii=False) + "\n")
            time.sleep(0.1)
        out.append(Question(text=cache[wid], question_id=wid,
                            metadata={"source_work_id": wid,
                                      "venue": r["meta"]["venue"],
                                      "year": r["meta"]["year"]}))
    return out


def serialize_result(qid: str, result) -> dict[str, Any]:
    return {
        "question_id": qid,
        "route": result.route.value,
        "outcome": result.outcome.value,
        "reason_code": result.reason_code.value if result.reason_code else None,
        "answer": result.answer.text if result.answer else None,
        "signals": {
            "boundary_probs": {k.value: v for k, v in
                               result.signals.boundary_probs.items()},
            "self_knowledge": result.signals.self_knowledge,
            "answerability": result.signals.answerability,
        } if result.signals else None,
        "verification": asdict(result.verification) if result.verification else None,
        "conflict": asdict(result.conflict) if result.conflict else None,
        "usage": asdict(result.usage),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=30)
    ap.add_argument("--corpus", type=int, default=8000)
    ap.add_argument("--hidden-states", action="store_true")
    ap.add_argument("--run-id", default=None)
    args = ap.parse_args()

    run_id = args.run_id or time.strftime("e2e-%Y%m%d-%H%M%S")
    run_dir = f"{ROOT}/results/runs/{run_id}"
    recorder = ProbeRecorder(run_dir)
    print("run_dir:", run_dir, flush=True)

    t0 = time.time()
    corpus = load_corpus(args.corpus)
    print(f"corpus: {len(corpus)} evidence docs", flush=True)

    retriever = BGERetriever(
        encoder_path=FT_ENCODER, reranker_path=RERANKER, corpus=corpus,
        candidate_top_k=40, final_top_k=5, batch_size=16, max_length=512,
        use_fp16=True)
    answerer = QwenAnswerer(LLM, recorder=recorder,
                            hidden_states=args.hidden_states)
    print(f"models loaded in {time.time()-t0:.0f}s", flush=True)

    probe_cache: dict[str, dict] = {}

    def probe_fn(question: Question) -> dict:
        qid = question.question_id or "?"
        if qid in probe_cache:
            return probe_cache[qid]
        p_ik = answerer.probe_self_knowledge(question)
        ent = answerer.probe_semantic_entropy(question)
        p_true = answerer.probe_p_true(question)
        probe_cache[qid] = {"p_ik": p_ik["p_ik"], "semantic_entropy": ent,
                            "p_true": p_true["p_true"]}
        return probe_cache[qid]

    estimator = ProbeSignalEstimator(probe_fn)
    verifier = EmbeddingEvidenceVerifier(
        FT_ENCODER, threshold=0.55, recorder=recorder)
    pipeline = SelfKnowledgeGatingPipeline(
        retriever=retriever, signal_estimator=estimator,
        answerer=answerer, verifier=verifier,
        adjudicator=RuleBasedConflictAdjudicator(), policy=GatePolicy(),
        top_k=5)

    questions = make_questions(answerer, args.n)
    print(f"questions: {len(questions)}", flush=True)

    results_path = os.path.join(run_dir, "results.jsonl")
    from collections import Counter
    routes: Counter = Counter()
    outcomes: Counter = Counter()
    lat = []
    with open(results_path, "w", encoding="utf-8") as rf:
        for i, question in enumerate(questions):
            recorder.record(question.question_id, "question_profile", {
                "question_id": question.question_id,
                "question_text": question.text,
                "source_work_id": question.metadata["source_work_id"],
                "venue": question.metadata["venue"],
                "year": question.metadata["year"],
            })
            # pre-run probes so 'signals' probe sees the same pass
            probe_fn(question)
            verifier._current_question_id = question.question_id or "?"
            result = pipeline.run(question)
            recorder.record(question.question_id, "signals", {
                "boundary_probs": {k.value: round(v, 6) for k, v in
                                   result.signals.boundary_probs.items()},
                "self_knowledge": result.signals.self_knowledge,
                "answerability": result.signals.answerability,
                "retrieval_utility": result.signals.retrieval_utility,
                "ambiguity": result.signals.ambiguity,
                "access_allowed": result.signals.access_allowed,
            })
            recorder.record(question.question_id, "route", {
                "action": result.route.value,
                "reason_code": result.reason_code.value
                if result.reason_code else None,
            })
            if result.evidence:
                recorder.record(question.question_id, "retrieval", {
                    "top_k": len(result.evidence),
                    "evidence": [{
                        "evidence_id": e.evidence_id,
                        "relevance": e.relevance,
                        "dense_score": e.metadata.get("dense_score"),
                        "reranker_score": e.metadata.get("reranker_score"),
                        "reliability": e.reliability,
                        "freshness": e.freshness,
                    } for e in result.evidence],
                })
            if result.verification:
                recorder.record(question.question_id, "verification",
                                asdict(result.verification))
            if result.conflict:
                recorder.record(question.question_id, "conflict",
                                asdict(result.conflict))
            recorder.record(question.question_id, "outcome", {
                "route": result.route.value,
                "outcome": result.outcome.value,
                "reason_code": result.reason_code.value
                if result.reason_code else None,
                "latency_ms": result.usage.latency_ms,
                "context_tokens": result.usage.context_tokens,
                "output_tokens": result.usage.output_tokens,
                "retrieval_calls": result.usage.retrieval_calls,
            })
            routes[result.route.value] += 1
            outcomes[result.outcome.value] += 1
            lat.append(result.usage.latency_ms)
            rf.write(json.dumps(serialize_result(question.question_id, result),
                                ensure_ascii=False) + "\n")
            rf.flush()
            print(f"[{i+1}/{len(questions)}] route={result.route.value} "
                  f"outcome={result.outcome.value} "
                  f"lat={result.usage.latency_ms:.0f}ms", flush=True)

    summary = {
        "run_id": run_id,
        "n_questions": len(questions),
        "corpus_size": len(corpus),
        "route_distribution": dict(routes),
        "outcome_distribution": dict(outcomes),
        "latency_p50_ms": sorted(lat)[len(lat) // 2] if lat else 0,
        "latency_p95_ms": sorted(lat)[int(len(lat) * 0.95)] if lat else 0,
        "probe_points": list(PROBE_POINTS),
        "probe_records": recorder.records,
        "wall_time_s": round(time.time() - t0, 1),
        "models": {"encoder": FT_ENCODER, "reranker": RERANKER, "llm": LLM},
    }
    with open(os.path.join(run_dir, "summary.json"), "w",
              encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    recorder.close()
    print(json.dumps(summary, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
