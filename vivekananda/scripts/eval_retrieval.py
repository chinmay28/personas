"""Retrieval eval on the question-for-passage set: recall@k at paragraph and section level, per model and method.

  .venv/bin/python scripts/eval_retrieval.py [--models nomic-embed-text,bge-m3,...] [--limit N]
Writes eval/results/retrieval_<timestamp>.json and prints a table.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import vklib  # noqa: E402

QFILE = os.path.join(vklib.ROOT, "data", "qfp", "questions.jsonl")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", default=",".join(vklib.EMBED_MODELS))
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--cand", type=int, default=50)
    ap.add_argument("--questions", default=QFILE)
    args = ap.parse_args()
    con = vklib.connect()
    have = {r[0].replace("vectors_", "") for r in con.execute("SELECT key FROM meta WHERE key LIKE 'vectors_%'")}
    models = [m for m in args.models.split(",") if vklib.model_slug(m) in have]
    qs = [json.loads(l) for l in open(args.questions)]
    if args.limit:
        qs = qs[:args.limit]
    print(f"{len(qs)} questions, models {models}")
    results = {}
    for model in models:
        if "/" in model and vklib.model_slug(model) not in have:
            continue
        slug = vklib.model_slug(model)
        t0 = time.time()
        qvecs = []
        for i in range(0, len(qs), 64):
            qvecs.extend(vklib.embed([q["question"] for q in qs[i:i + 64]], model, "query"))
        hits = defaultdict(lambda: defaultdict(int))  # method -> metric -> count
        by_kind = defaultdict(lambda: defaultdict(int))
        for q, qv in zip(qs, qvecs):
            dense = [r[0] for r in con.execute(f"SELECT id FROM vec_para_{slug} WHERE embedding MATCH ? ORDER BY distance LIMIT ?", (vklib.pack(qv), args.cand))]
            bm = [r[0] for r in con.execute("SELECT id FROM fts_para WHERE fts_para MATCH ? ORDER BY bm25(fts_para) LIMIT ?", (vklib.fts_query(q["question"]), args.cand))]
            hyb = [h.id for h in vklib.rrf([dense, bm])]
            secs = [h.id for h in vklib.search_sections(con, q["question"], model, k=10, cand=args.cand, qvec=qv)]
            for name, lst in (("dense", dense), ("bm25", bm), ("hybrid", hyb)):
                for k in (1, 5, 10):
                    ok = q["para_id"] in lst[:k]
                    hits[name][f"para@{k}"] += ok
                    by_kind[(name, q["kind"])][f"para@{k}"] += ok
                    # section-level via paragraph list: any paragraph of the gold section
                    sec_ok = any(con.execute("SELECT section_id FROM paragraphs WHERE id = ?", (pid,)).fetchone()[0] == q["section_id"] for pid in lst[:k])
                    hits[name][f"sec_via_para@{k}"] += sec_ok
            for k in (1, 5, 10):
                hits["hybrid"][f"section@{k}"] += q["section_id"] in secs[:k]
        n = len(qs)
        results[model] = {m: {k: round(v / n, 4) for k, v in d.items()} for m, d in hits.items()}
        results[model]["by_kind"] = {f"{m}/{kind}": {k: round(v / sum(1 for q in qs if q["kind"] == kind), 4) for k, v in d.items()}
                                     for (m, kind), d in by_kind.items()}
        results[model]["seconds"] = round(time.time() - t0, 1)
        print(f"\n== {model} ({results[model]['seconds']}s)")
        for m in ("dense", "bm25", "hybrid"):
            r = results[model][m]
            print(f"  {m:7s} para@1={r['para@1']:.3f} para@5={r['para@5']:.3f} para@10={r['para@10']:.3f} | sec_via_para@5={r['sec_via_para@5']:.3f}" +
                  (f" | section@5={r['section@5']:.3f} section@10={r['section@10']:.3f}" if m == "hybrid" else ""))
        for kind in ("specific", "general"):
            r = results[model]["by_kind"].get(f"hybrid/{kind}", {})
            if r:
                print(f"  hybrid/{kind:8s} para@5={r['para@5']:.3f} para@10={r['para@10']:.3f}")
    out = os.path.join(vklib.ROOT, "eval", "results", f"retrieval_{time.strftime('%Y%m%d_%H%M%S')}.json")
    json.dump({"n_questions": len(qs), "cand": args.cand, "results": results}, open(out, "w"), indent=1)
    print("wrote", out)


if __name__ == "__main__":
    main()
