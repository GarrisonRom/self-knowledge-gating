"""Reproducible full-data BGE-M3 fine-tuning and test evaluation.

The configuration is the source of truth. A run writes its config snapshot,
manifest, training summary, log and retrieval metrics under one directory:

    results/runs/bge-m3-matsci-v1-seed42/

Usage:
    python scripts/train_embedding.py --config configs/bge_m3_train.yaml
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import random
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModel, AutoTokenizer, get_cosine_schedule_with_warmup


ROOT = Path(__file__).resolve().parents[1]
QUERY_INSTRUCTION = "Represent this sentence for searching relevant passages: "


class Tee:
    def __init__(self, *streams):
        self.streams = streams

    def write(self, text: str) -> None:
        for stream in self.streams:
            stream.write(text)
            stream.flush()

    def flush(self) -> None:
        for stream in self.streams:
            stream.flush()


class PairDataset(Dataset):
    def __init__(self, path: Path, tokenizer, query_length: int, passage_length: int):
        self.rows = [json.loads(line) for line in path.open(encoding="utf-8")]
        self.tokenizer = tokenizer
        self.query_length = query_length
        self.passage_length = passage_length

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int):
        row = self.rows[index]
        query = self.tokenizer(
            QUERY_INSTRUCTION + row["query"],
            truncation=True,
            max_length=self.query_length,
        )
        passage = self.tokenizer(
            row["positive"],
            truncation=True,
            max_length=self.passage_length,
        )
        return query["input_ids"], passage["input_ids"]


def collate_pairs(batch, pad_token_id: int):
    queries, passages = zip(*batch)

    def pad(sequences):
        max_length = max(len(sequence) for sequence in sequences)
        ids = torch.full(
            (len(sequences), max_length), pad_token_id, dtype=torch.long
        )
        attention = torch.zeros_like(ids)
        for row, sequence in enumerate(sequences):
            ids[row, : len(sequence)] = torch.tensor(sequence, dtype=torch.long)
            attention[row, : len(sequence)] = 1
        return ids, attention

    query_ids, query_attention = pad(queries)
    passage_ids, passage_attention = pad(passages)
    return query_ids, query_attention, passage_ids, passage_attention


def load_config(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if not isinstance(config, dict):
        raise ValueError("training config must be a YAML mapping")
    return config


def resolve_path(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def git_commit() -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def set_seed(seed: int, deterministic: bool) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
    if deterministic:
        torch.use_deterministic_algorithms(True, warn_only=True)


def select_dtype(precision: str) -> torch.dtype:
    if precision == "bf16":
        if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
            raise RuntimeError("bf16 training requires a CUDA device with bf16 support")
        return torch.bfloat16
    if precision == "fp16":
        if not torch.cuda.is_available():
            raise RuntimeError("fp16 training requires CUDA")
        return torch.float16
    if precision == "fp32":
        return torch.float32
    raise ValueError(f"unsupported precision: {precision}")


def autocast_context(dtype: torch.dtype):
    if dtype == torch.float32:
        return torch.autocast(device_type="cuda", enabled=False)
    return torch.autocast(device_type="cuda", dtype=dtype)


def cls_embed(model, input_ids, attention_mask):
    hidden = model(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state[:, 0]
    return F.normalize(hidden.float(), dim=-1)


def batch_embed(model, tokenizer, texts: Sequence[str], max_length: int, device: str, dtype: torch.dtype):
    vectors = []
    for start in range(0, len(texts), 32):
        encoded = tokenizer(
            list(texts[start : start + 32]),
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        ).to(device)
        with torch.inference_mode(), autocast_context(dtype):
            vectors.append(cls_embed(model, encoded["input_ids"], encoded["attention_mask"]).cpu())
    return torch.cat(vectors).numpy()


def quick_dev_accuracy(model, tokenizer, path: Path, limit: int, query_length: int, passage_length: int, dtype: torch.dtype) -> float:
    rows = [json.loads(line) for line in path.open(encoding="utf-8")]
    rows = rows[:limit] if limit else rows
    queries = [QUERY_INSTRUCTION + row["query"] for row in rows]
    passages = [row["positive"] for row in rows]
    model.eval()
    query_vectors = batch_embed(model, tokenizer, queries, query_length, "cuda", dtype)
    passage_vectors = batch_embed(model, tokenizer, passages, passage_length, "cuda", dtype)
    scores = query_vectors @ passage_vectors.T
    return float(np.mean(np.argmax(scores, axis=1) == np.arange(len(rows))))


def environment_record() -> dict[str, Any]:
    record: dict[str, Any] = {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "torch": torch.__version__,
        "cuda_runtime": torch.version.cuda,
        "cuda_available": torch.cuda.is_available(),
    }
    if torch.cuda.is_available():
        record["gpu"] = torch.cuda.get_device_name(0)
    for module_name in ("transformers", "modelscope", "FlagEmbedding"):
        try:
            module = __import__(module_name)
            record[module_name] = getattr(module, "__version__", "available")
        except Exception as exc:  # pragma: no cover - diagnostic only
            record[module_name] = f"missing:{type(exc).__name__}"
    return record


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def train(config_path: Path) -> tuple[Path, Path]:
    config = load_config(config_path)
    seed = int(config["seed"])
    run_id = str(config["run"]["id"])
    run_dir = resolve_path(config["run"]["output_dir"])
    if run_dir.exists() and any(run_dir.iterdir()) and not config["run"].get("overwrite", False):
        raise FileExistsError(
            f"run directory is not empty: {run_dir}; choose another run.id or set overwrite=true"
        )
    run_dir.mkdir(parents=True, exist_ok=True)
    log_handle = (run_dir / "train.log").open("w", encoding="utf-8")
    original_stdout = sys.stdout
    sys.stdout = Tee(original_stdout, log_handle)
    started = time.time()
    try:
        shutil.copy2(config_path, run_dir / "config.yaml")
        set_seed(seed, bool(config["training"].get("deterministic_algorithms", True)))
        if not torch.cuda.is_available():
            raise RuntimeError("full BGE training requires CUDA")

        data_cfg = config["data"]
        model_cfg = config["models"]
        train_path = resolve_path(data_cfg["train_pairs"])
        dev_path = resolve_path(data_cfg["dev_pairs"])
        base_path = resolve_path(model_cfg["base_encoder"])
        output_model = resolve_path(model_cfg["finetuned_encoder"])
        if output_model.exists() and any(output_model.iterdir()):
            raise FileExistsError(f"model output already exists: {output_model}")

        training_cfg = config["training"]
        dtype = select_dtype(str(training_cfg["precision"]))
        tokenizer = AutoTokenizer.from_pretrained(base_path)
        model = AutoModel.from_pretrained(base_path, dtype=dtype)
        if training_cfg.get("gradient_checkpointing", True):
            model.gradient_checkpointing_enable()
        model.train().to("cuda")

        dataset = PairDataset(
            train_path,
            tokenizer,
            int(training_cfg["query_max_length"]),
            int(training_cfg["passage_max_length"]),
        )
        max_train = data_cfg.get("max_train_pairs")
        if max_train is not None and len(dataset) > int(max_train):
            sampler = random.Random(seed)
            indices = sampler.sample(range(len(dataset)), int(max_train))
            dataset = torch.utils.data.Subset(dataset, indices)
        loader_generator = torch.Generator()
        loader_generator.manual_seed(seed)
        loader = DataLoader(
            dataset,
            batch_size=int(training_cfg["batch_pairs"]),
            shuffle=True,
            generator=loader_generator,
            num_workers=int(training_cfg.get("dataloader_workers", 0)),
            drop_last=True,
            collate_fn=lambda batch: collate_pairs(batch, tokenizer.pad_token_id),
        )
        accumulation = int(training_cfg["gradient_accumulation"])
        target_micro_steps = math.ceil(len(loader) * float(training_cfg["epochs"]))
        total_steps = math.ceil(target_micro_steps / accumulation)
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=float(training_cfg["learning_rate"]),
            weight_decay=float(training_cfg["weight_decay"]),
        )
        scheduler = get_cosine_schedule_with_warmup(
            optimizer,
            int(float(training_cfg["warmup_fraction"]) * total_steps),
            total_steps,
        )
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
                "train_pairs_used": len(dataset),
            },
            "models": {"base_encoder": str(base_path), "finetuned_encoder": str(output_model)},
            "training": training_cfg,
            "environment": environment_record(),
            "started_at_epoch": started,
        }
        write_json(run_dir / "manifest.json", manifest)
        print(json.dumps({"event": "start", **manifest}, ensure_ascii=False), flush=True)
        print(f"train_pairs={len(dataset)} optimizer_steps={total_steps}", flush=True)

        step = 0
        micro_step = 0
        running_losses: list[float] = []
        model.zero_grad(set_to_none=True)
        while micro_step < target_micro_steps:
            for query_ids, query_attention, passage_ids, passage_attention in loader:
                if micro_step >= target_micro_steps:
                    break
                query_ids = query_ids.cuda(non_blocking=True)
                query_attention = query_attention.cuda(non_blocking=True)
                passage_ids = passage_ids.cuda(non_blocking=True)
                passage_attention = passage_attention.cuda(non_blocking=True)
                with autocast_context(dtype):
                    query_vectors = cls_embed(model, query_ids, query_attention)
                    passage_vectors = cls_embed(model, passage_ids, passage_attention)
                    logits = query_vectors @ passage_vectors.T / float(training_cfg["temperature"])
                    labels = torch.arange(len(query_vectors), device="cuda")
                    loss = F.cross_entropy(logits, labels)
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
                        print(
                            f"step={step}/{total_steps} loss={np.mean(running_losses):.6f} "
                            f"progress={micro_step}/{target_micro_steps}",
                            flush=True,
                        )
                        running_losses = []
                    if step % int(training_cfg["eval_every_steps"]) == 0 or step == total_steps:
                        dev_acc = quick_dev_accuracy(
                            model,
                            tokenizer,
                            dev_path,
                            int(data_cfg.get("dev_quick_limit") or 0),
                            int(training_cfg["query_max_length"]),
                            int(training_cfg["passage_max_length"]),
                            dtype,
                        )
                        print(f"dev_argmax_acc={dev_acc:.6f}", flush=True)
                        model.train()

        output_model.mkdir(parents=True, exist_ok=True)
        model.save_pretrained(output_model)
        tokenizer.save_pretrained(output_model)
        elapsed = time.time() - started
        summary = {
            "run_id": run_id,
            "status": "trained",
            "seed": seed,
            "train_pairs": len(dataset),
            "optimizer_steps": total_steps,
            "elapsed_seconds": round(elapsed, 3),
            "output_model": str(output_model),
            "loss_temperature": float(training_cfg["temperature"]),
            "pooling": "CLS",
            "normalization": "L2",
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
    sys.path.insert(0, str(ROOT / "scripts"))
    from eval_retrieval import evaluate_model

    config = load_config(config_path)
    data_path = resolve_path(config["data"]["test_pairs"])
    eval_cfg = config["evaluation"]
    metrics_path = run_dir / "retrieval_metrics.jsonl"
    if metrics_path.exists():
        metrics_path.unlink()
    models = []
    if bool(eval_cfg.get("include_baseline", True)):
        models.append((resolve_path(config["models"]["base_encoder"]), "bge-m3-baseline"))
    models.append((finetuned_model, "bge-m3-matsci-v1-seed42"))
    for model_path, tag in models:
        result = evaluate_model(
            model_path=model_path,
            pairs_path=data_path,
            split=str(eval_cfg.get("split", "test")),
            batch_size=int(eval_cfg.get("batch_size", 32)),
            query_max_length=int(eval_cfg.get("query_max_length", 128)),
            passage_max_length=int(eval_cfg.get("passage_max_length", 512)),
            tag=tag,
        )
        with metrics_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(result, ensure_ascii=False) + "\n")
        print(json.dumps({"event": "evaluation_complete", **result}, ensure_ascii=False), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=ROOT / "configs/bge_m3_train.yaml")
    parser.add_argument("--skip-test", action="store_true")
    args = parser.parse_args()
    run_dir, finetuned_model = train(args.config.resolve())
    if not args.skip_test:
        evaluate_after_training(args.config.resolve(), run_dir, finetuned_model)


if __name__ == "__main__":
    main()
