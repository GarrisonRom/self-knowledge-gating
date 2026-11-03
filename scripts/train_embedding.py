"""Fine-tune bge-m3 on materials-science title->abstract pairs.

Contrastive InfoNCE with in-batch negatives. bf16 + grad checkpointing.
Saves an HF-format checkpoint loadable by plain transformers (eval script).
Usage: python scripts/train_embedding.py [--epochs 1] [--batch-pairs 32]
"""
from __future__ import annotations

import argparse
import json
import math
import os
import random
import time

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from transformers import (AutoModel, AutoTokenizer, get_cosine_schedule_with_warmup)

ROOT = "/root/autodl-tmp/self-knowledge-gating"
BASE = f"{ROOT}/models/modelscope/models/BAAI--bge-m3/snapshots/master"
OUT = f"{ROOT}/models/finetuned/bge-m3-matsci-v1"
QUERY_INSTR = "Represent this sentence for searching relevant passages: "


class PairDataset(Dataset):
    def __init__(self, path, tok, q_len, p_len):
        self.rows = [json.loads(l) for l in open(path, encoding="utf-8")]
        self.tok, self.q_len, self.p_len = tok, q_len, p_len

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        r = self.rows[i]
        q = self.tok(QUERY_INSTR + r["query"], truncation=True,
                     max_length=self.q_len)
        p = self.tok(r["positive"], truncation=True, max_length=self.p_len)
        return q["input_ids"], p["input_ids"]


def collate(batch, pad_id):
    qs, ps = zip(*batch)

    def pad(seqs, max_len):
        out = torch.full((len(seqs), max_len), pad_id, dtype=torch.long)
        att = torch.zeros((len(seqs), max_len), dtype=torch.long)
        for i, s in enumerate(seqs):
            s = s[:max_len]
            out[i, :len(s)] = torch.tensor(s)
            att[i, :len(s)] = 1
        return out, att

    q_ids, q_att = pad(qs, max(len(s) for s in qs))
    p_ids, p_att = pad(ps, max(len(s) for s in ps))
    return q_ids, q_att, p_ids, p_att


def cls_embed(model, ids, att):
    h = model(input_ids=ids, attention_mask=att).last_hidden_state[:, 0]
    return F.normalize(h.float(), dim=-1)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=float, default=1.0)
    ap.add_argument("--batch-pairs", type=int, default=32)
    ap.add_argument("--accum", type=int, default=2)
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--q-len", type=int, default=128)
    ap.add_argument("--p-len", type=int, default=512)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--max-train", type=int, default=48000)
    ap.add_argument("--eval-every", type=int, default=200)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    random.seed(args.seed)
    device = "cuda"
    tok = AutoTokenizer.from_pretrained(BASE)
    model = AutoModel.from_pretrained(BASE, dtype=torch.bfloat16)
    model.gradient_checkpointing_enable()
    model.train().to(device)

    ds = PairDataset(f"{ROOT}/data/processed/pairs_train.jsonl", tok,
                     args.q_len, args.p_len)
    if args.max_train and len(ds) > args.max_train:
        idx = random.sample(range(len(ds)), args.max_train)
        ds = torch.utils.data.Subset(ds, idx)
    dl = DataLoader(ds, batch_size=args.batch_pairs, shuffle=True,
                    collate_fn=lambda b: collate(b, tok.pad_token_id),
                    num_workers=2, drop_last=True)

    steps_per_epoch = len(dl) // args.accum
    total_steps = int(steps_per_epoch * args.epochs)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    sched = get_cosine_schedule_with_warmup(opt, int(0.05 * total_steps),
                                            total_steps)
    print(f"train pairs={len(ds)} opt steps={total_steps}", flush=True)

    def dev_quick():
        rows = [json.loads(l) for l in open(
            f"{ROOT}/data/processed/pairs_dev.jsonl", encoding="utf-8")][:300]
        qs = [QUERY_INSTR + r["query"] for r in rows]
        ps = [r["positive"] for r in rows]
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            qe = batch_embed(model, tok, qs, args.q_len, device)
            pe = batch_embed(model, tok, ps, args.p_len, device)
        s = qe @ pe.T
        hits = sum(1 for i in range(len(rows))
                   if np.argmax(s[i]) == i)
        return hits / len(rows)

    def batch_embed(m, t, texts, max_len, dev):
        outs = []
        for i in range(0, len(texts), 32):
            enc = t(texts[i:i + 32], padding=True, truncation=True,
                    max_length=max_len, return_tensors="pt").to(dev)
            outs.append(cls_embed(m, enc["input_ids"], enc["attention_mask"]).cpu())
        return torch.cat(outs).numpy()

    t0, step, running = time.time(), 0, []
    micro = 0
    done = False
    for epoch in range(math.ceil(args.epochs)):
        if done:
            break
        for q_ids, q_att, p_ids, p_att in dl:
            q_ids, q_att = q_ids.to(device), q_att.to(device)
            p_ids, p_att = p_ids.to(device), p_att.to(device)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                qe = cls_embed(model, q_ids, q_att)
                pe = cls_embed(model, p_ids, p_att)
                logits = qe @ pe.T / 0.05
                labels = torch.arange(len(qe), device=device)
                loss = F.cross_entropy(logits, labels)
            (loss / args.accum).backward()
            running.append(loss.item())
            micro += 1
            if micro % args.accum == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
                sched.step()
                opt.zero_grad(set_to_none=True)
                step += 1
                if step % 20 == 0:
                    el = time.time() - t0
                    print(f"step {step}/{total_steps} loss {np.mean(running):.4f} "
                          f"{el:.0f}s eta {el/max(step,1)*(total_steps-step):.0f}s",
                          flush=True)
                    running = []
                if step % args.eval_every == 0 or step == total_steps:
                    acc = dev_quick()
                    print(f"  [dev argmax-acc@{len(ds) and 300}] {acc:.4f}",
                          flush=True)
                    model.train()
                if step >= total_steps:
                    done = True
                    break

    os.makedirs(OUT, exist_ok=True)
    model.save_pretrained(OUT)
    tok.save_pretrained(OUT)
    meta = {"base": BASE, "recipe": vars(args), "train_pairs": len(ds),
            "opt_steps": total_steps, "loss_temp": 0.05,
            "note": "InfoNCE in-batch negatives; CLS pool; L2 norm"}
    with open(f"{OUT}/finetune_meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    print("saved to", OUT, flush=True)


if __name__ == "__main__":
    main()
