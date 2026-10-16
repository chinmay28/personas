"""Phase 5: router accuracy and confusion matrix on eval/questions.jsonl (run only after the owner has reviewed the set).

  .venv/bin/python scripts/eval_router.py [--model qwen3:14b] [--include-extras]
Writes eval/results/router_<ts>.json and prints the confusion matrix. The error to minimise is persona-when-oracle-was-possible.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections import Counter, defaultdict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import modes  # noqa: E402
import vklib  # noqa: E402

LABELS = ["scholar", "oracle", "persona", "mixed", "decline"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=modes.DEV_MODEL)
    ap.add_argument("--include-extras", action="store_true", help="also score the 20 decline/mixed extras")
    ap.add_argument("--no-expand", action="store_true")
    ap.add_argument("--extras-only", action="store_true")
    args = ap.parse_args()
    con = vklib.connect()
    qs = [json.loads(l) for l in open(os.path.join(vklib.ROOT, "eval", "questions.jsonl"))]
    if args.extras_only:
        qs = [q for q in qs if q.get("extra")]
    elif not args.include_extras:
        qs = [q for q in qs if not q.get("extra")]
    conf = defaultdict(Counter)
    rows = []
    t0 = time.time()
    for i, q in enumerate(qs, 1):
        ctx = modes.retrieve(con, q["question"], 8, vklib.DEFAULT_MODEL, None if args.no_expand else args.model)
        r = modes.route(con, ctx, args.model)
        conf[q["expected_mode"]][r["mode"]] += 1
        rows.append({"id": q["id"], "question": q["question"], "expected": q["expected_mode"], "predicted": r["mode"], "reason": r.get("reason", "")})
        if i % 20 == 0:
            print(f"{i}/{len(qs)} {time.time() - t0:.0f}s", flush=True)
    n = len(qs)
    correct = sum(conf[l][l] for l in LABELS)
    print(f"\naccuracy {correct}/{n} = {correct / n:.3f}  (model {args.model})")
    print("expected \\ predicted " + " ".join(f"{l:>8s}" for l in LABELS))
    for e in LABELS:
        if sum(conf[e].values()):
            print(f"{e:20s} " + " ".join(f"{conf[e][p]:8d}" for p in LABELS))
    pwo = conf["oracle"]["persona"]
    print(f"persona-when-oracle-expected: {pwo} ({pwo / max(sum(conf['oracle'].values()), 1):.1%} of oracle questions)")
    out = {"model": args.model, "n": n, "accuracy": round(correct / n, 4), "confusion": {e: dict(c) for e, c in conf.items()}, "rows": rows}
    path = os.path.join(vklib.ROOT, "eval", "results", f"router_{time.strftime('%Y%m%d_%H%M%S')}.json")
    json.dump(out, open(path, "w"), indent=1, ensure_ascii=False)
    print("wrote", path)


if __name__ == "__main__":
    main()
