"""Hybrid search CLI.  .venv/bin/python scripts/search.py "what is karma yoga" [--level para|section] [-k 10] [--json] [--model M]"""
from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import vklib  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("query")
    ap.add_argument("--level", choices=["para", "section"], default="para")
    ap.add_argument("-k", type=int, default=8)
    ap.add_argument("--model", default=vklib.DEFAULT_MODEL, choices=list(vklib.EMBED_MODELS))
    ap.add_argument("--json", action="store_true")
    ap.add_argument("--width", type=int, default=300)
    args = ap.parse_args()
    con = vklib.connect()
    out = []
    if args.level == "para":
        for h in vklib.search_paragraphs(con, args.query, args.model, args.k):
            p = vklib.get_paragraph(con, h.id)
            s = vklib.get_section(con, p["section_id"])
            out.append({"id": h.id, "section_id": s["id"], "score": round(h.score, 5), "dense_rank": h.dense_rank, "bm25_rank": h.bm25_rank,
                        "citation": vklib.citation(s), "genre": s["genre"], "text": p["text"]})
    else:
        for h in vklib.search_sections(con, args.query, args.model, args.k):
            s = vklib.get_section(con, h.id)
            out.append({"id": h.id, "score": round(h.score, 5), "dense_rank": h.dense_rank, "bm25_rank": h.bm25_rank,
                        "citation": vklib.citation(s), "genre": s["genre"], "char_count": s["char_count"], "text": s["text"]})
    if args.json:
        print(json.dumps(out, ensure_ascii=False, indent=1))
        return
    for i, r in enumerate(out, 1):
        ranks = f"dense#{r['dense_rank'] or '-'} bm25#{r['bm25_rank'] or '-'}"
        print(f"{i}. {r['citation']}  ({r['id']}, {r['genre']}, {ranks})")
        t = r["text"].replace("\n", " ")
        print("   " + (t[:args.width] + ("..." if len(t) > args.width else "")))
        print()


if __name__ == "__main__":
    main()
