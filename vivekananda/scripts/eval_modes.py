"""Phase 3 acceptance: oracle verbatim rate on N questions, scholar citation coverage on M questions.

  .venv/bin/python scripts/eval_modes.py --oracle 100 --scholar 50 [--model qwen3:14b]
Questions are drawn (seeded) from data/qfp/questions.jsonl ("general" kind). Results -> eval/results/modes_<ts>.json
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import modes  # noqa: E402
import vklib  # noqa: E402

QFILE = os.path.join(vklib.ROOT, "data", "qfp", "questions.jsonl")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--oracle", type=int, default=100)
    ap.add_argument("--scholar", type=int, default=50)
    ap.add_argument("--model", default=modes.DEV_MODEL)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--no-expand", action="store_true")
    args = ap.parse_args()
    con = vklib.connect()
    qs = [json.loads(l) for l in open(QFILE)]
    qs = [q for q in qs if q["kind"] == "general"]
    rng = random.Random(args.seed)
    rng.shuffle(qs)
    out = {"model": args.model, "oracle": [], "scholar": []}
    t0 = time.time()
    for i, q in enumerate(qs[:args.oracle], 1):
        r = modes.ask(con, q["question"], "oracle", args.model, expand=not args.no_expand)
        g = r.get("quote_guard") or []
        out["oracle"].append({"question": q["question"], "gold_para": q["para_id"], "gold_section": q["section_id"], "mode": r["mode"],
                              "fell_through": bool(r.get("cascade")), "quotes": len(g), "verified": sum(1 for x in g if x["verified"]),
                              "quoted_words": r.get("quoted_words"), "gold_para_quoted": any(x["para_id"] == q["para_id"] for x in r.get("quotes", [])),
                              "gold_section_cited": any(c["section_id"] == q["section_id"] for c in r.get("citations", [])), "seconds": r["seconds"]})
        if i % 10 == 0:
            print(f"oracle {i}/{args.oracle} {time.time() - t0:.0f}s", flush=True)
    for i, q in enumerate(qs[args.oracle:args.oracle + args.scholar], 1):
        r = modes.ask(con, q["question"], "scholar", args.model, expand=not args.no_expand)
        s = r.get("stats", {})
        g = r.get("quote_guard") or []
        out["scholar"].append({"question": q["question"], "gold_section": q["section_id"], "sentences": s.get("sentences"), "cited": s.get("sentences_cited"),
                               "invalid_tags": s.get("invalid_tags"), "dropped_uncited": s.get("dropped_uncited", 0), "quotes": len(g), "verified": sum(1 for x in g if x["verified"]),
                               "gold_section_cited": any(c["section_id"] == q["section_id"] for c in r.get("citations", [])),
                               "no_record": "no record" in r["text"].lower(), "seconds": r["seconds"]})
        if i % 10 == 0:
            print(f"scholar {i}/{args.scholar} {time.time() - t0:.0f}s", flush=True)
    o = out["oracle"]; sc = out["scholar"]
    summary = {
        "oracle": {"n": len(o), "fallthrough_rate": round(sum(x["fell_through"] for x in o) / max(len(o), 1), 3),
                   "quotes": sum(x["quotes"] for x in o), "verified": sum(x["verified"] for x in o),
                   "verbatim_rate": round(sum(x["verified"] for x in o) / max(sum(x["quotes"] for x in o), 1), 4),
                   "gold_para_quoted_rate": round(sum(x["gold_para_quoted"] for x in o) / max(len(o), 1), 3),
                   "gold_section_cited_rate": round(sum(x["gold_section_cited"] for x in o) / max(len(o), 1), 3),
                   "mean_quoted_words": round(sum(x["quoted_words"] or 0 for x in o if not x["fell_through"]) / max(sum(1 for x in o if not x["fell_through"]), 1), 1),
                   "over_cap": sum(1 for x in o if (x["quoted_words"] or 0) > modes.QUOTE_WORD_CAP), "mean_seconds": round(sum(x["seconds"] for x in o) / max(len(o), 1), 1)},
        "scholar": {"n": len(sc), "sentences": sum(x["sentences"] or 0 for x in sc), "cited": sum(x["cited"] or 0 for x in sc),
                    "citation_coverage": round(sum(x["cited"] or 0 for x in sc) / max(sum(x["sentences"] or 0 for x in sc), 1), 3),
                    "invalid_tags": sum(x["invalid_tags"] or 0 for x in sc), "dropped_uncited": sum(x["dropped_uncited"] or 0 for x in sc), "quotes": sum(x["quotes"] for x in sc), "verified": sum(x["verified"] for x in sc),
                    "gold_section_cited_rate": round(sum(x["gold_section_cited"] for x in sc) / max(len(sc), 1), 3),
                    "no_record_rate": round(sum(x["no_record"] for x in sc) / max(len(sc), 1), 3), "mean_seconds": round(sum(x["seconds"] for x in sc) / max(len(sc), 1), 1)},
    }
    out["summary"] = summary
    path = os.path.join(vklib.ROOT, "eval", "results", f"modes_{time.strftime('%Y%m%d_%H%M%S')}.json")
    json.dump(out, open(path, "w"), indent=1, ensure_ascii=False)
    print(json.dumps(summary, indent=1))
    print("wrote", path)


if __name__ == "__main__":
    main()
