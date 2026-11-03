"""Fetch materials-science abstracts from OpenAlex into data/external/openalex/raw/.

OpenAlex requires a free API key for cursor paging, so this fetcher partitions
by publication year and uses page-based paging, which is still key-free.
Raw layer: one immutable JSONL per run, polite rate limit, resumable.
Usage: python scripts/fetch_openalex.py [target_count]
"""
import json
import os
import sys
import time

import requests

BASE = "https://api.openalex.org/works"
FILTER_TMPL = (
    "concepts.id:C192562407,has_abstract:true,type:article,"
    "language:en,publication_year:{year}"
)
SELECT = (
    "id,doi,title,publication_year,primary_location,"
    "abstract_inverted_index,cited_by_count,concepts"
)
TARGET = int(sys.argv[1]) if len(sys.argv) > 1 else 60000
ROOT = "/root/autodl-tmp/self-knowledge-gating"
OUT = os.path.join(ROOT, "data/external/openalex/raw/works.jsonl")
LOG = os.path.join(ROOT, "data/external/openalex/fetch.log")
os.makedirs(os.path.dirname(OUT), exist_ok=True)

session = requests.Session()
session.headers.update({"User-Agent": "Mozilla/5.0 (research; materials-embedding)"})

start_n = 0
if os.path.exists(OUT):
    with open(OUT, encoding="utf-8") as f:
        start_n = sum(1 for _ in f)
    print(f"resuming from {start_n} existing records", flush=True)

n = 0
t0 = time.time()
years = list(range(2025, 2013, -1))  # newest first
with open(OUT, "a", encoding="utf-8") as f:
    for year in years:
        if n + start_n >= TARGET:
            break
        page = 1 + start_n // 200 if year == years[0] and start_n else 1
        while n + start_n < TARGET:
            params = {
                "filter": FILTER_TMPL.format(year=year),
                "per-page": 200,
                "select": SELECT,
                "page": page,
            }
            d = None
            for attempt in range(5):
                try:
                    r = session.get(BASE, params=params, timeout=60)
                    if r.status_code == 200:
                        d = r.json()
                        break
                    print(f"retry {attempt} year={year} page={page} "
                          f"status={r.status_code} {r.text[:120]}", flush=True)
                except Exception as e:  # noqa: BLE001
                    print(f"retry {attempt} {e!r}", flush=True)
                time.sleep(5 * (attempt + 1))
            if d is None:
                print("FAILED, stopping", flush=True)
                sys.exit(1)
            results = d["results"]
            if not results:
                break
            for w in results:
                f.write(json.dumps(w, ensure_ascii=False) + "\n")
            n += len(results)
            total = n + start_n
            if total % 2000 < 200:
                el = time.time() - t0
                print(f"{total}/{TARGET} elapsed {el:.0f}s", flush=True)
            if len(results) < 200:
                break
            page += 1
            time.sleep(0.15)
print("DONE", n + start_n, "works in", round(time.time() - t0), "s", flush=True)
