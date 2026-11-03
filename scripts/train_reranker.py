"""Reproducible full-data bge-reranker-v2-m3 fine-tuning and evaluation.

Training is listwise over each query's positive passage and hard negatives.
The same script evaluates the frozen baseline and the fine-tuned reranker on
the held-out split. Results are written under one named run directory.

Usage:
    python scripts/train_reranker.py --config configs/bge_reranker_train.yaml
"""

from __future__ import annotations

import argparse
import json
import math
import random
import shutil
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from train_embedding import (
    ROOT,
    Tee,
    autocast_context,
    environment_record,
    git_commit,
    load_config,
    resolve_path,
    select_dtype,
    set_seed,
    sha256_file,
    write_json,
)


class RerankerGroupDataset(Dataset):
    def __init__(self, path: Path, max_negatives: int, max_queries: int | None, seed: int):
        rows = [json.loads(line) for line in path.open(encoding="utf-8")]
        rows = [
            row for row in rows
            if len(row.get("hard_negatives", [])) >= max_negatives
        ]
        if max_queries is not None and len(rows) > max_queries:
            rows = random.Random(seed).sample(rows, max_queries)
        self.rows = rows
        self.max_negatives = max_negatives

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> tuple[str, tuple[str, ...]]:
        row = self.rows[index]
        passages = (row["positive"], *row["hard_negatives"][: self.max_negatives])
        return row["query"], tuple(passages)


def collate_groups(batch, tokenizer, max_length: int):
    queries, passage_groups = zip(*batch)
    group_width = len(passage_groups[0])
    if any(len(group) != group_width for group in passage_groups):
        raise ValueError("all reranker groups must have the same width")
    flat_queries = [query for query in queries for _ in range(group_width)]
    flat_passages = [passage for group in passage_groups for passage in group]
    encoded = tokenizer(
        list(flat_queries),
        list(flat_passages),
        padding=True,
        truncation=True,
        max_length=max_length,
        return_tensors="pt",
    )
    return encoded, len(queries), group_width


def move_batch(encoded, device: str):
    return {key: value.to(device) for key, value in encoded.items()}


def logits_for_batch(model, encoded, group_count: int, group_width: int, dtype: torch.dtype):
    with autocast_context(dtype):
        logits = model(**encoded).logits
    if logits.ndim == 2:
        logits = logits[:, 0]
    return logits.reshape(group_count, group_width).float()


@torch.inference_mode()
def quick_dev_accuracy(model, tokenizer, dataset: RerankerGroupDataset, limit: int, max_length: int, batch_queries: int, dtype: torch.dtype) -> float:
    rows = dataset.rows[:limit] if limit else dataset.rows
    if not rows:
        return 0.0
    model.eval()
    hits = 0
    for start in range(0, len(rows), batch_queries):
        batch = [(row["query"], (row["positive"], *row["hard_negatives"][: dataset.max_negatives])) for row in rows[start : start + batch_queries]]
        encoded, group_count, group_width = collate_groups(batch, tokenizer, max_length)
        scores = logits_for_batch(model, move_batch(encoded, "cuda"), group_count, group_width, dtype)
        hits += int((scores.argmax(dim=1) == 0).sum().item())
    return hits / len(rows)


@torch.inference_mode()
def evaluate_model(model_path: Path, rows: list[dict[str, Any]], max_negatives: int, max_length: int, batch_queries: int, tag: str) -> dict[str, Any]:
    rows = [row for row in rows if len(row.get("hard_negatives", [])) >= max_negatives]
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if device == "cuda" else torch.float32
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSequenceClassification.from_pretrained(model_path, dtype=dtype).to(device)
    model.eval()
    ranks: list[int] = []
    pair_total = 0
    pair_hits = 0
    for start in range(0, len(rows), batch_queries):
        batch_rows = rows[start : start + batch_queries]
        batch = [
            (row["query"], (row["positive"], *row["hard_negatives"][:max_negatives]))
            for row in batch_rows
        ]
        encoded, group_count, group_width = collate_groups(batch, tokenizer, max_length)
        scores = logits_for_batch(model, move_batch(encoded, device), group_count, group_width, dtype).cpu().numpy()
        for row_index, row_scores in enumerate(scores):
            order = np.argsort(-row_scores)
            rank = int(np.flatnonzero(order == 0)[0]) + 1
            ranks.append(rank)
            pair_total += group_width - 1
            pair_hits += int(np.sum(row_scores[0] > row_scores[1:]))

    count = max(len(ranks), 1)
    result = {
        "model": str(model_path),
        "tag": tag,
        "n_queries": len(ranks),
        "candidates_per_query": max_negatives + 1,
        "n_hard_pairs": pair_total,
        "hard_negative_definition": "positive score greater than each hard-negative score within the same query group",
        "top1_accuracy": round(sum(rank == 1 for rank in ranks) / count, 6),
        "mrr_at_candidates": round(sum(1.0 / rank for rank in ranks) / count, 6),
        "ndcg_at_candidates": round(sum(1.0 / np.log2(rank + 1) for rank in ranks) / count, 6),
        "hard_negative_distinction": round(pair_hits / max(pair_total, 1), 6),
    }
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return result


def train(config_path: Path) -> tuple[Path, Path]:
    config = load_config(config_path)
    seed = int(config["seed"])
    run_id = str(config["run"]["id"])
    run_dir = resolve_path(config["run"]["output_dir"])
    if run_dir.exists() and any(run_dir.iterdir()) and not config["run"].get("overwrite", False):
        raise FileExistsError(f"run directory is not empty: {run_dir}")
    run_dir.mkdir(parents=True, exist_ok=True)
    log_handle = (run_dir / "train.log").open("w", encoding="utf-8")
    original_stdout = sys.stdout
    sys.stdout = Tee(original_stdout, log_handle)
    started = time.time()
    try:
        shutil.copy2(config_path, run_dir / "config.yaml")
        set_seed(seed, bool(config["training"].get("deterministic_algorithms", True)))
        if not torch.cuda.is_available():
            raise RuntimeError("full reranker training requires CUDA")

        data_cfg = config["data"]
        model_cfg = config["models"]
        train_path = resolve_path(data_cfg["train_pairs"])
        dev_path = resolve_path(data_cfg["dev_pairs"])
        base_path = resolve_path(model_cfg["base_reranker"])
        output_model = resolve_path(model_cfg["finetuned_reranker"])
        if output_model.exists() and any(output_model.iterdir()):
            raise FileExistsError(f"model output already exists: {output_model}")

        training_cfg = config["training"]
        dtype = select_dtype(str(training_cfg["precision"]))
        tokenizer = AutoTokenizer.from_pretrained(base_path)
        model = AutoModelForSequenceClassification.from_pretrained(base_path, dtype=dtype)
        if training_cfg.get("gradient_checkpointing", True):
            model.gradient_checkpointing_enable()
        model.train().to("cuda")

        max_negatives = int(data_cfg["hard_negatives_per_query"])
        dataset = RerankerGroupDataset(
            train_path,
            max_negatives=max_negatives,
            max_queries=data_cfg.get("max_train_queries"),
            seed=seed,
        )
        dev_dataset = RerankerGroupDataset(dev_path, max_negatives, None, seed)
        loader_generator = torch.Generator()
        loader_generator.manual_seed(seed)
        loader = DataLoader(
            dataset,
            batch_size=int(training_cfg["queries_per_batch"]),
            shuffle=True,
            generator=loader_generator,
            num_workers=int(training_cfg.get("dataloader_workers", 0)),
            drop_last=True,
            collate_fn=lambda batch: collate_groups(batch, tokenizer, int(training_cfg["max_length"])),
        )
        accumulation = int(training_cfg["gradient_accumulation"])
        target_micro_steps = math.ceil(len(loader) * float(training_cfg["epochs"]))
        total_steps = math.ceil(target_micro_steps / accumulation)
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=float(training_cfg["learning_rate"]),
            weight_decay=float(training_cfg["weight_decay"]),
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_steps)
        manifest = {
            "run_id": run_id,
            "config_version": config.get("config_version"),
            "seed": seed,
            "git_commit": git_commit(),
            "config_sha256": sha256_file(config_path),
            "data": {
                "train_pairs": str(train_path),
                "dev_pairs": str(dev_path),
                "train_sha256": sha256_file(train_path),
                "dev_sha256": sha256_file(dev_path),
                "train_queries_used": len(dataset),
                "hard_negatives_per_query": max_negatives,
            },
            "models": {"base_reranker": str(base_path), "finetuned_reranker": str(output_model)},
            "training": training_cfg,
            "environment": environment_record(),
            "started_at_epoch": started,
        }
        write_json(run_dir / "manifest.json", manifest)
        print(json.dumps({"event": "start", **manifest}, ensure_ascii=False), flush=True)
        print(f"train_queries={len(dataset)} optimizer_steps={total_steps} candidates={max_negatives + 1}", flush=True)

        step = 0
        micro_step = 0
        running_losses: list[float] = []
        model.zero_grad(set_to_none=True)
        while micro_step < target_micro_steps:
            for encoded, group_count, group_width in loader:
                if micro_step >= target_micro_steps:
                    break
                scores = logits_for_batch(model, move_batch(encoded, "cuda"), group_count, group_width, dtype)
                targets = torch.zeros(group_count, dtype=torch.long, device="cuda")
                loss = torch.nn.functional.cross_entropy(scores, targets)
                (loss / accumulation).backward()
                running_losses.append(float(loss.detach().cpu()))
                micro_step += 1
                if micro_step % accumulation == 0 or micro_step >= target_micro_steps:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad(set_to_none=True)
                    step += 1
                    if step % 20 == 0 or step == total_steps:
                        print(f"step={step}/{total_steps} loss={np.mean(running_losses):.6f} progress={micro_step}/{target_micro_steps}", flush=True)
                        running_losses = []
                    if step % int(training_cfg["eval_every_steps"]) == 0 or step == total_steps:
                        dev_acc = quick_dev_accuracy(
                            model, tokenizer, dev_dataset,
                            int(data_cfg.get("dev_quick_limit") or 0),
                            int(training_cfg["max_length"]),
                            int(training_cfg["queries_per_batch"]), dtype,
                        )
                        print(f"dev_top1_accuracy={dev_acc:.6f}", flush=True)
                        model.train()

        output_model.mkdir(parents=True, exist_ok=True)
        model.save_pretrained(output_model)
        tokenizer.save_pretrained(output_model)
        summary = {
            "run_id": run_id,
            "status": "trained",
            "seed": seed,
            "train_queries": len(dataset),
            "optimizer_steps": total_steps,
            "elapsed_seconds": round(time.time() - started, 3),
            "output_model": str(output_model),
            "candidates_per_query": max_negatives + 1,
        }
        write_json(run_dir / "training_summary.json", summary)
        manifest["finished_at_epoch"] = time.time()
        manifest["training_summary"] = summary
        write_json(run_dir / "manifest.json", manifest)
        print(json.dumps({"event": "training_complete", **summary}, ensure_ascii=False), flush=True)
        return run_dir, output_model
    finally:
        sys.stdout = original_stdout
        log_handle.close()


def evaluate_after_training(config_path: Path, run_dir: Path, finetuned_model: Path) -> None:
    config = load_config(config_path)
    eval_cfg = config["evaluation"]
    test_rows = [json.loads(line) for line in resolve_path(config["data"]["test_pairs"]).open(encoding="utf-8")]
    max_negatives = int(config["data"]["hard_negatives_per_query"])
    metrics_path = run_dir / "reranker_metrics.jsonl"
    if metrics_path.exists():
        metrics_path.unlink()
    models = []
    if bool(eval_cfg.get("include_baseline", True)):
        models.append((resolve_path(config["models"]["base_reranker"]), "bge-reranker-v2-m3-baseline"))
    models.append((finetuned_model, "bge-reranker-v2-m3-matsci-v1-seed42"))
    for model_path, tag in models:
        result = evaluate_model(
            model_path=model_path,
            rows=test_rows,
            max_negatives=max_negatives,
            max_length=int(eval_cfg["max_length"]),
            batch_queries=int(eval_cfg["batch_queries"]),
            tag=tag,
        )
        with metrics_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(result, ensure_ascii=False) + "\n")
        print(json.dumps({"event": "evaluation_complete", **result}, ensure_ascii=False), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=ROOT / "configs/bge_reranker_train.yaml")
    parser.add_argument("--skip-test", action="store_true")
    args = parser.parse_args()
    run_dir, finetuned_model = train(args.config.resolve())
    if not args.skip_test:
        evaluate_after_training(args.config.resolve(), run_dir, finetuned_model)


if __name__ == "__main__":
    main()
