"""Convert a datamill `pairs` export (question-for-passage, -answer-source chunk) into data/qfp/questions.jsonl format.

  datamill export -project data/qfp/project -format pairs -answer-source chunk -out data/qfp/pairs.jsonl
  .venv/bin/python scripts/import_datamill_pairs.py data/qfp/pairs.jsonl [--out data/qfp/questions_full.jsonl]
Records: id, para_id (from ref), section_id (from meta), genre, kind ("datamill"), question, model.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import vklib  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("pairs")
    ap.add_argument("--out", default=os.path.join(vklib.ROOT, "data", "qfp", "questions_full.jsonl"))
    ap.add_argument("--model", default="qwen3:14b")
    args = ap.parse_args()
    n = skipped = 0
    seen = {}
    with open(args.pairs) as f, open(args.out, "w") as out:
        for line in f:
            d = json.loads(line)
            ref, meta = d.get("ref"), d.get("meta") or {}
            q = (d.get("question") or "").strip()
            if not ref or not q or not meta.get("section_id"):
                skipped += 1
                continue
            k = seen.get(ref, 0) + 1
            seen[ref] = k
            out.write(json.dumps({"id": f"{ref}-d{k}", "para_id": ref, "section_id": meta["section_id"], "genre": meta.get("genre"),
                                  "kind": "datamill", "question": q, "model": args.model}, ensure_ascii=False) + "\n")
            n += 1
    print(f"wrote {n} questions for {len(seen)} paragraphs to {args.out} (skipped {skipped})")


if __name__ == "__main__":
    main()
