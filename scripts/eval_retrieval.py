"""Evaluate a dense encoder (bge-m3 baseline or fine-tuned checkpoint).

Deterministic title->abstract retrieval on a split of processed pairs.
Metrics: Recall@10/20, nDCG@10, MRR@10, hard-negative distinction rate.
Usage: python scripts/eval_retrieval.py --model-path <hf-or-local-path> --split test
"""
from __future__ import annotations

import argparse
import json

import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer

ROOT = "/root/autodl-tmp/self-knowledge-gating"
QUERY_INSTR = "Represent this sentence for searching relevant passages: "


def encode(model, tok, texts, device, max_len, batch_size=32, is_query=False):
    model.eval()
    outs = []
    for i in range(0, len(texts), batch_size):
        chunk = texts[i:i + batch_size]
        if is_query:
            chunk = [QUERY_INSTR + t for t in chunk]
        enc = tok(chunk, padding=True, truncation=True, max_length=max_len,
                  return_tensors="pt").to(device)
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            h = model(**enc).last_hidden_state[:, 0]  # CLS
        outs.append(torch.nn.functional.normalize(h.float(), dim=-1).cpu())
    return torch.cat(outs).numpy()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-path", required=True)
    ap.add_argument("--split", default="test")
    ap.add_argument("--pairs", default=None)
    ap.add_argument("--max-query-len", type=int, default=128)
    ap.add_argument("--max-passage-len", type=int, default=512)
    ap.add_argument("--tag", default=None)
    args = ap.parse_args()

    pairs_path = args.pairs or f"{ROOT}/data/processed/pairs_{args.split}.jsonl"
    rows = [json.loads(l) for l in open(pairs_path, encoding="utf-8")]
    queries = [r["query"] for r in rows]
    corpus = [r["positive"] for r in rows]
    qid2cid = {i: i for i in range(len(rows))}
    # venue-mate hard negatives must be in the corpus to be distinguishable
    id2cid = {r["positive_id"]: i for i, r in enumerate(rows)}
    hard_pairs = []
    for r in rows:
        for hn_id in r.get("hard_negative_ids", []):
            if hn_id in id2cid:
                hard_pairs.append((r["positive_id"], hn_id))

    device = "cuda"
    tok = AutoTokenizer.from_pretrained(args.model_path)
    model = AutoModel.from_pretrained(args.model_path, dtype=torch.bfloat16,
                                      device_map=device)

    Q = encode(model, tok, queries, device, args.max_query_len, is_query=True)
    D = encode(model, tok, corpus, device, args.max_passage_len)
    scores = Q @ D.T

    def metrics_at(k):
        recall = 0.0
        mrr = 0.0
        ndcg = 0.0
        for i in range(len(rows)):
            topk = np.argsort(-scores[i])[:k]
            hit_rank = None
            for rank, j in enumerate(topk, 1):
                if j == qid2cid[i]:
                    hit_rank = rank
                    break
            if hit_rank:
                recall += 1
                mrr += 1.0 / hit_rank
                ndcg += 1.0 / np.log2(hit_rank + 1)
        n = len(rows)
        return recall / n, ndcg / n, mrr / n

    r10, n10, m10 = metrics_at(10)
    r20, _, _ = metrics_at(20)

    hn_hit = 0
    if hard_pairs:
        pos_scores = {}
        neg_scores = {}
        for i, r in enumerate(rows):
            pos_scores[r["positive_id"]] = scores[i][id2cid[r["positive_id"]]]
        for pid, nid in hard_pairs:
            # negative's similarity under its own query row is not defined;
            # use max over all query rows as the retrieval-style score
            j = id2cid[nid]
            neg_scores[nid] = scores[:, j].max()
        hn_hit = sum(1 for pid, nid in hard_pairs
                     if pos_scores[pid] > neg_scores[nid]) / len(hard_pairs)

    result = {
        "model": args.model_path,
        "tag": args.tag or args.model_path,
        "split": args.split,
        "n_queries": len(rows),
        "n_hard_pairs": len(hard_pairs),
        "recall_at_10": round(r10, 4),
        "recall_at_20": round(r20, 4),
        "ndcg_at_10": round(n10, 4),
        "mrr_at_10": round(m10, 4),
        "hard_negative_distinction": round(hn_hit, 4),
    }
    print(json.dumps(result, indent=2))
    out = f"{ROOT}/results/runs/eval_{args.split}.jsonl"
    import os
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "a", encoding="utf-8") as f:
        f.write(json.dumps(result, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
