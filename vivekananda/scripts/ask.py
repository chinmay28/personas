"""ask CLI (phase 9): streams the answer with the mode badge and numbered citations.

  .venv/bin/python scripts/ask.py "What is Karma-Yoga?" [--mode auto|scholar|oracle|persona|mixed] [--backend prompted|unified]
                                  [--model M] [--show 3] [--json] [--api http://host:8765] [-k 8] [--no-expand]

--backend unified (default) uses the fine-tuned single model (vivekananda:8b-run2) with the training-time prompt; prompted is the phase 3 cascade.
--show N prints the full section behind citation N after the answer.  --api sends the question to a running server (scripts/server.py)
instead of answering in-process; --json then prints the server's JSON, which is identical to the in-process result.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import requests

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import engine  # noqa: E402
import vklib  # noqa: E402

BADGE = {"scholar": "SCHOLAR (recounting my words, cited)", "oracle": "ORACLE (my own words, verbatim)",
         "persona": "PERSONA (reasoning beyond my record)", "mixed": "MIXED (what I said / what I would say)"}


def via_api(api: str, args) -> dict:
    body = {"question": args.question, "mode": args.mode, "stream": False, "k": args.k}
    if args.backend:
        body["backend"] = args.backend
    if args.model:
        body["model"] = args.model
    r = requests.post(api.rstrip("/") + "/ask", json=body, timeout=900)
    r.raise_for_status()
    return r.json()


def section_text(args, sid: str) -> dict:
    if args.api:
        r = requests.get(args.api.rstrip("/") + f"/section/{sid}", timeout=60)
        r.raise_for_status()
        return r.json()
    return engine.section_payload(vklib.connect(), sid)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("question")
    ap.add_argument("--mode", default="auto", choices=list(engine.MODES))
    ap.add_argument("--backend", default=None, choices=list(engine.BACKENDS), help=f"default: VK_BACKEND or {engine.DEFAULT_BACKEND}")
    ap.add_argument("--model", default=None, help="Ollama model (default: per backend)")
    ap.add_argument("--embed-model", default=vklib.DEFAULT_MODEL)
    ap.add_argument("-k", type=int, default=8)
    ap.add_argument("--json", action="store_true")
    ap.add_argument("--show", type=int, default=None, metavar="N", help="print the full section behind citation N")
    ap.add_argument("--api", default=os.environ.get("VK_API"), help="answer via a running server, e.g. http://127.0.0.1:8765")
    ap.add_argument("--no-expand", action="store_true", help="prompted backend: skip LLM query expansion")
    args = ap.parse_args()

    printed = {"mode": False}
    if args.api:
        r = via_api(args.api, args)
    else:
        backend = args.backend or engine.DEFAULT_BACKEND
        con = vklib.connect()

        def on_mode(m, info):
            if args.json:
                return
            reason = (info.get("router") or {}).get("reason") or ""
            cascade = info.get("cascade")
            print(f"[{BADGE.get(m, m)}]" + (f"  router: {reason}" if reason else "") + (f"  ({cascade})" if cascade else ""), flush=True)
            printed["mode"] = True

        def on_token(t):
            if not args.json:
                sys.stdout.write(t)
                sys.stdout.flush()

        r = engine.slim(engine.answer(con, args.question, args.mode, backend, args.model, args.k, expand=not args.no_expand,
                                      on_mode=on_mode, on_token=on_token))
        if not args.json and backend == "unified" and r["mode"] == "persona" and not r.get("cascade"):
            print("\n")  # tokens were streamed; the guarded text follows
    if args.json:
        print(json.dumps(r, ensure_ascii=False, indent=1))
        return
    if args.api or not printed["mode"]:
        print(f"[{BADGE.get(r['mode'], r['mode'])}]" + (f"  router: {r['router']['reason']}" if r.get("router") else "")
              + (f"  ({r.get('cascade')})" if r.get("cascade") else ""))
    print()
    print(r["text"])
    if r.get("citations"):
        print("\nCitations")
        for c in r["citations"]:
            print(f"  [{c['n']}] {c['citation']}  ({c['para_id']})")
    if r.get("see_also"):
        print("See also: " + "; ".join(s["citation"] for s in r["see_also"]))
    g = r.get("quote_guard") or []
    if g:
        print(f"\nQuote guard: {sum(1 for x in g if x['verified'])}/{len(g)} quoted spans verified verbatim")
    if r.get("stats"):
        s = r["stats"]
        print(f"Citation coverage: {s['sentences_cited']}/{s['sentences']} sentences cited, {s['invalid_tags']} invalid tags removed")
    if len(r.get("queries", [])) > 1:
        print(f"Search queries: {r['queries']}")
    print(f"\n({r.get('backend')}, {r['model']}, {r['seconds']}s)")
    if args.show is not None:
        cites = r.get("citations") or []
        c = next((c for c in cites if c["n"] == args.show), None)
        if c is None:
            print(f"\n--show {args.show}: no such citation (have 1..{len(cites)})")
            return
        sec = section_text(args, c["section_id"])
        print(f"\n===== [{c['n']}] {sec['citation']}  ({sec['id']}, {sec['genre']}, {len(sec['paragraphs'])} paragraphs) =====")
        for p in sec["paragraphs"]:
            mark = ">> " if p["id"] == c["para_id"] else "   "
            print(mark + p["text"].replace("\n", "\n   ") + "\n")


if __name__ == "__main__":
    main()
