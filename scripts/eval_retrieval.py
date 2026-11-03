"""Evaluate a dense encoder on a versioned title-to-abstract split.

Metrics: Recall@10/20, nDCG@10, MRR@10 and query-local hard-negative
distinction. The module is also imported by ``train_embedding.py`` so training
and final testing use exactly the same evaluator.
"""

from __future__ import annotations

import argparse
from contextlib import nullcontext
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer


ROOT = Path(__file__).resolve().parents[1]
QUERY_INSTRUCTION = "Represent this sentence for searching relevant passages: "


def encode(
    model,
    tokenizer,
    texts: list[str],
    device: str,
    max_length: int,
    batch_size: int,
) -> np.ndarray:
    vectors = []
    model.eval()
    for start in range(0, len(texts), batch_size):
        encoded = tokenizer(
            texts[start : start + batch_size],
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        ).to(device)
        autocast = (
            torch.autocast("cuda", dtype=torch.bfloat16)
            if device == "cuda"
            else nullcontext()
        )
        with torch.inference_mode(), autocast:
            hidden = model(**encoded).last_hidden_state[:, 0]
            vectors.append(torch.nn.functional.normalize(hidden.float(), dim=-1).cpu())
    return torch.cat(vectors).numpy()


def load_rows(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.open(encoding="utf-8")]


def evaluate_model(
    *,
    model_path: Path,
    pairs_path: Path,
    split: str,
    batch_size: int,
    query_max_length: int,
    passage_max_length: int,
    tag: str,
) -> dict[str, Any]:
    rows = load_rows(pairs_path)
    queries = [QUERY_INSTRUCTION + row["query"] for row in rows]
    passages = [row["positive"] for row in rows]
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModel.from_pretrained(
        model_path,
        dtype=torch.bfloat16 if device == "cuda" else torch.float32,
    ).to(device)
    query_vectors = encode(model, tokenizer, queries, device, query_max_length, batch_size)
    passage_vectors = encode(model, tokenizer, passages, device, passage_max_length, batch_size)
    scores = query_vectors @ passage_vectors.T

    def metrics_at(k: int) -> tuple[float, float, float]:
        hits = 0
        reciprocal_rank = 0.0
        ndcg = 0.0
        for row_index in range(len(rows)):
            top_k = np.argsort(-scores[row_index])[:k]
            ranks = np.flatnonzero(top_k == row_index)
            if len(ranks):
                rank = int(ranks[0]) + 1
                hits += 1
                reciprocal_rank += 1.0 / rank
                ndcg += 1.0 / np.log2(rank + 1)
        count = max(len(rows), 1)
        return hits / count, ndcg / count, reciprocal_rank / count

    recall10, ndcg10, mrr10 = metrics_at(10)
    recall20, _, _ = metrics_at(20)

    hard_text_by_id: dict[str, str] = {}
    hard_pairs: list[tuple[int, str]] = []
    for row_index, row in enumerate(rows):
        ids = row.get("hard_negative_ids", [])
        texts = row.get("hard_negatives", [])
        for negative_id, negative_text in zip(ids, texts):
            if negative_id in hard_text_by_id and hard_text_by_id[negative_id] != negative_text:
                raise ValueError(f"hard negative id has conflicting text: {negative_id}")
            hard_text_by_id[negative_id] = negative_text
            hard_pairs.append((row_index, negative_id))

    if hard_pairs:
        hard_ids = list(hard_text_by_id)
        hard_vectors = encode(
            model,
            tokenizer,
            hard_ids and [hard_text_by_id[item] for item in hard_ids] or [],
            device,
            passage_max_length,
            batch_size,
        )
        hard_index = {negative_id: index for index, negative_id in enumerate(hard_ids)}
        hard_scores = query_vectors @ hard_vectors.T
        hard_hits = 0
        for row_index, negative_id in hard_pairs:
            if scores[row_index, row_index] > hard_scores[row_index, hard_index[negative_id]]:
                hard_hits += 1
        hard_negative_distinction = hard_hits / len(hard_pairs)
    else:
        hard_negative_distinction = None

    result = {
        "model": str(model_path),
        "tag": tag,
        "split": split,
        "n_queries": len(rows),
        "n_hard_pairs": len(hard_pairs),
        "hard_negative_definition": "query-local positive score greater than each hard-negative score",
        "recall_at_10": round(recall10, 6),
        "recall_at_20": round(recall20, 6),
        "ndcg_at_10": round(ndcg10, 6),
        "mrr_at_10": round(mrr10, 6),
        "hard_negative_distinction": (
            round(hard_negative_distinction, 6)
            if hard_negative_distinction is not None
            else None
        ),
    }
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", required=True, type=Path)
    parser.add_argument("--pairs", required=True, type=Path)
    parser.add_argument("--split", default="test")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--query-max-length", type=int, default=128)
    parser.add_argument("--passage-max-length", type=int, default=512)
    parser.add_argument("--tag", required=True)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    result = evaluate_model(
        model_path=args.model_path,
        pairs_path=args.pairs,
        split=args.split,
        batch_size=args.batch_size,
        query_max_length=args.query_max_length,
        passage_max_length=args.passage_max_length,
        tag=args.tag,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(result, ensure_ascii=False) + "\n")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
