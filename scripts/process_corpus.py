"""Secondary processing of the OpenAlex materials corpus.

raw -> interim : reconstruct abstracts, clean, filter, dedupe.
interim -> processed : build contrastive pairs (title -> abstract) with
venue/year hard negatives, and deterministic train/dev/test splits.

Outputs (all inside data/, no copies of the same content):
  interim/openalex_clean.jsonl          one cleaned record per work
  processed/pairs_train.jsonl           {query, positive, hard_negatives[], meta}
  processed/pairs_dev.jsonl / pairs_test.jsonl
  processed/manifest.json               versions, counts, split seed, field map
"""
from __future__ import annotations

import hashlib
import json
import random
import re
import unicodedata
from collections import defaultdict

ROOT = "/root/autodl-tmp/self-knowledge-gating"
RAW = f"{ROOT}/data/external/openalex/raw/works.jsonl"
INTERIM = f"{ROOT}/data/interim/openalex_clean.jsonl"
PROCESSED = f"{ROOT}/data/processed"
SEED = 42
MIN_ABS_WORDS = 40
MAX_ABS_WORDS = 320

random.seed(SEED)

_WS = re.compile(r"\s+")
_LATEX = re.compile(r"\$[^$]*\$|\\[a-zA-Z]+{[^}]*}")
_URL = re.compile(r"https?://\S+")


def reconstruct(inv: dict) -> str:
    if not inv:
        return ""
    pos: dict[int, str] = {}
    for word, idxs in inv.items():
        for i in idxs:
            pos[i] = word
    return " ".join(pos[i] for i in sorted(pos))


def clean(text: str) -> str:
    text = unicodedata.normalize("NFKC", text or "")
    text = _LATEX.sub(" ", text)
    text = _URL.sub(" ", text)
    text = _WS.sub(" ", text).strip()
    return text


def norm_key(title: str) -> str:
    t = re.sub(r"[^a-z0-9 ]", "", title.lower())
    return hashlib.md5(_WS.sub(" ", t).strip().encode()).hexdigest()


def main() -> None:
    seen_doi: set[str] = set()
    seen_title: set[str] = set()
    kept, drop_noabs, drop_len, drop_dup = 0, 0, 0, 0
    records = []
    with open(RAW, encoding="utf-8") as f:
        for line in f:
            w = json.loads(line)
            title = clean(w.get("title") or "")
            abstract = clean(reconstruct(w.get("abstract_inverted_index")))
            words = len(abstract.split())
            if words < 5:
                drop_noabs += 1
                continue
            if not (MIN_ABS_WORDS <= words <= MAX_ABS_WORDS):
                drop_len += 1
                continue
            doi = (w.get("doi") or "").lower().rsplit("/", 1)[-1]
            tkey = norm_key(title)
            if (doi and doi in seen_doi) or tkey in seen_title:
                drop_dup += 1
                continue
            if doi:
                seen_doi.add(doi)
            seen_title.add(tkey)
            src = (w.get("primary_location") or {}).get("source") or {}
            records.append({
                "work_id": w["id"].rsplit("/", 1)[-1],
                "doi": doi,
                "title": title,
                "abstract": abstract,
                "year": w.get("publication_year"),
                "venue": src.get("display_name") or "unknown",
                "cited_by_count": w.get("cited_by_count") or 0,
            })
            kept += 1
    print(f"kept={kept} noabs={drop_noabs} len={drop_len} dup={drop_dup}")

    with open(INTERIM, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    # ---- build pairs with venue/year hard negatives ----
    by_bucket: dict[tuple, list[int]] = defaultdict(list)
    for i, r in enumerate(records):
        by_bucket[(r["venue"], r["year"])].append(i)

    rng = random.Random(SEED)
    order = list(range(len(records)))
    rng.shuffle(order)
    n_test = n_dev = max(500, int(0.1 * len(records)))
    test_idx = set(order[:n_test])
    dev_idx = set(order[n_test:n_test + n_dev])
    train_idx = set(order[n_test + n_dev:])

    def write_pairs(idxs, path):
        with open(path, "w", encoding="utf-8") as f:
            for i in sorted(idxs):
                r = records[i]
                mates = [j for j in by_bucket[(r["venue"], r["year"])] if j != i]
                hard = rng.sample(mates, min(3, len(mates)))
                f.write(json.dumps({
                    "query": r["title"],
                    "positive": r["abstract"],
                    "positive_id": r["work_id"],
                    "hard_negatives": [records[j]["abstract"] for j in hard],
                    "hard_negative_ids": [records[j]["work_id"] for j in hard],
                    "meta": {"venue": r["venue"], "year": r["year"],
                             "cited_by_count": r["cited_by_count"]},
                }, ensure_ascii=False) + "\n")

    import os
    os.makedirs(PROCESSED, exist_ok=True)
    write_pairs(train_idx, f"{PROCESSED}/pairs_train.jsonl")
    write_pairs(dev_idx, f"{PROCESSED}/pairs_dev.jsonl")
    write_pairs(test_idx, f"{PROCESSED}/pairs_test.jsonl")

    manifest = {
        "source": "OpenAlex works API",
        "filter": "concepts.id:C192562407, has_abstract, article, en, 2014-2025",
        "seed": SEED,
        "counts": {"raw_kept": kept, "train": len(train_idx),
                   "dev": len(dev_idx), "test": len(test_idx)},
        "constraints": {"min_abs_words": MIN_ABS_WORDS,
                        "max_abs_words": MAX_ABS_WORDS},
        "fields": {"query": "paper title", "positive": "own abstract",
                   "hard_negative": "same venue+year, different work"},
    }
    with open(f"{PROCESSED}/manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)
    print("splits:", len(train_idx), len(dev_idx), len(test_idx))
    print("manifest written")


if __name__ == "__main__":
    main()
