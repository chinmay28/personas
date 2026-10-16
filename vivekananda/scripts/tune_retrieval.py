"""Sweep fusion parameters on the question-for-passage set without re-embedding: rank distribution of the gold
paragraph, candidate depth, RRF k, BM25 header weight, multi-model fusion, and 'near-duplicate' credit.
  .venv/bin/python scripts/tune_retrieval.py [--models bge-m3/h,nomic-embed-text] [--limit N]
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections import Counter

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import vklib  # noqa: E402

QFILE = os.path.join(vklib.ROOT, "data", "qfp", "questions.jsonl")


def norm(t):
    return " ".join(t.lower().split())[:300]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", default="bge-m3/h,bge-m3,nomic-embed-text")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--questions", default=QFILE)
    args = ap.parse_args()
    con = vklib.connect()
    qs = [json.loads(l) for l in open(args.questions)]
    if args.limit:
        qs = qs[:args.limit]
    models = args.models.split(",")
    # near-duplicate map: paragraphs sharing the first 300 normalised chars count as the same passage
    dup = {}
    groups = {}
    for pid, text in con.execute("SELECT id, text FROM paragraphs WHERE length(text) >= 200"):
        groups.setdefault(norm(text), []).append(pid)
    for g in groups.values():
        for pid in g:
            dup[pid] = set(g)
    print(f"{len(qs)} questions; near-duplicate groups with >1 member: {sum(1 for g in groups.values() if len(g) > 1)}")
    t0 = time.time()
    qvecs = {}
    for m in models:
        v = []
        for i in range(0, len(qs), 64):
            v.extend(vklib.embed([q["question"] for q in qs[i:i + 64]], m, "query"))
        qvecs[m] = v
    print(f"query embeddings in {time.time() - t0:.0f}s")
    DEPTH = 200
    dense_lists = {m: [] for m in models}
    for m in models:
        slug = vklib.model_slug(m)
        for qv in qvecs[m]:
            dense_lists[m].append([r[0] for r in con.execute(f"SELECT id FROM vec_para_{slug} WHERE embedding MATCH ? ORDER BY distance LIMIT ?", (vklib.pack(qv), DEPTH))])
    bm_lists = {}
    for w in (0.0, 1.0, 3.0, 8.0, 20.0):
        bm_lists[w] = [[r[0] for r in con.execute(f"SELECT id FROM fts_para WHERE fts_para MATCH ? ORDER BY bm25(fts_para, 0.0, {w}, 1.0) LIMIT ?", (vklib.fts_query(q["question"]), DEPTH))] for q in qs]
    print(f"candidate lists in {time.time() - t0:.0f}s")

    def score(lists_per_q, k_rrf=60, cand=50, label=""):
        ranks, dranks = [], []
        for q, lists in zip(qs, lists_per_q):
            fused = [h.id for h in vklib.rrf([l[:cand] for l in lists], k=k_rrf)]
            r = fused.index(q["para_id"]) + 1 if q["para_id"] in fused else None
            ranks.append(r)
            dr = next((i + 1 for i, pid in enumerate(fused) if pid in dup.get(q["para_id"], {q["para_id"]})), None)
            dranks.append(dr)
        n = len(qs)
        def rec(rs, k): return sum(1 for r in rs if r and r <= k) / n
        print(f"  {label:48s} @1={rec(ranks,1):.3f} @5={rec(ranks,5):.3f} @10={rec(ranks,10):.3f} @20={rec(ranks,20):.3f} @50={rec(ranks,50):.3f} | dup-credit @5={rec(dranks,5):.3f} @10={rec(dranks,10):.3f}")
        return ranks

    print("\n== single model hybrid (dense + bm25 header w=3), cand=50, k=60")
    base = {}
    for m in models:
        base[m] = score([[d, b] for d, b in zip(dense_lists[m], bm_lists[3.0])], label=m)
    best = models[0]
    print(f"\n== {best}: bm25 header weight")
    for w in (0.0, 1.0, 3.0, 8.0, 20.0):
        score([[d, b] for d, b in zip(dense_lists[best], bm_lists[w])], label=f"header w={w}")
    print(f"\n== {best}: candidate depth and RRF k")
    for cand in (50, 100, 200):
        for k in (10, 60):
            score([[d, b] for d, b in zip(dense_lists[best], bm_lists[3.0])], k_rrf=k, cand=cand, label=f"cand={cand} k={k}")
    print("\n== multi-model fusion (cand=100, k=60)")
    for combo in ([models[0], models[1]], models):
        score([[dense_lists[m][i] for m in combo] + [bm_lists[3.0][i]] for i in range(len(qs))], cand=100, label="+".join(combo) + "+bm25")
    print("\n== dense only / bm25 only, best model, cand=200")
    score([[d] for d in dense_lists[best]], cand=200, label=f"{best} dense only")
    score([[b] for b in bm_lists[3.0]], cand=200, label="bm25 only (w=3)")
    print("\n== by question kind and genre (best, hybrid cand=100)")
    ranks = score([[d, b] for d, b in zip(dense_lists[best], bm_lists[3.0])], cand=100, label="all")
    for key in ("kind", "genre"):
        groups_ = Counter(q[key] for q in qs)
        for g, cnt in groups_.most_common():
            rs = [r for q, r in zip(qs, ranks) if q[key] == g]
            print(f"    {key}={g:14s} n={cnt:4d} @5={sum(1 for r in rs if r and r<=5)/cnt:.3f} @10={sum(1 for r in rs if r and r<=10)/cnt:.3f}")


if __name__ == "__main__":
    main()
