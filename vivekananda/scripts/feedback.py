"""Review the thumbs-up / thumbs-down log the PWA posts to the server (eval/feedback.jsonl, one JSON event per tap).

  .venv/bin/python scripts/feedback.py                 # summary: latest rating per answer, counts by mode and backend, every thumbs-down
  .venv/bin/python scripts/feedback.py --since 2026-10-20 --down-only --json
  .venv/bin/python scripts/feedback.py --export eval/feedback_review.md   # the same as a Markdown file to annotate

The latest event per entry_id wins ("clear" withdraws a rating). Nothing here changes the model: the point is to read the
thumbs-downs every so often, decide whether each is a retrieval miss, a routing miss, a voice problem or a bad question, and turn the
real ones into eval questions (eval/questions.jsonl) or dataset fixes for the next run.
"""
from __future__ import annotations

import argparse
import collections
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import vklib  # noqa: E402

PATH = os.environ.get("VK_FEEDBACK", os.path.join(vklib.ROOT, "eval", "feedback.jsonl"))


def load(path: str, since: str | None) -> list[dict]:
    out = []
    if not os.path.exists(path):
        return out
    for l in open(path, encoding="utf-8"):
        try:
            e = json.loads(l)
        except json.JSONDecodeError:
            continue
        if since and e.get("ts", "")[:10] < since:
            continue
        out.append(e)
    return out


def latest_per_entry(events: list[dict]) -> list[dict]:
    d: dict[str, dict] = {}
    for e in events:
        d[e.get("entry_id") or e["ts"]] = e
    return [e for e in d.values() if e.get("rating") in ("up", "down")]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--path", default=PATH)
    ap.add_argument("--since", help="YYYY-MM-DD")
    ap.add_argument("--down-only", action="store_true")
    ap.add_argument("--json", action="store_true")
    ap.add_argument("--export", help="write the review as Markdown to this path")
    args = ap.parse_args()
    events = load(args.path, args.since)
    rated = latest_per_entry(events)
    by_rating = collections.Counter(e["rating"] for e in rated)
    by_mode = collections.defaultdict(collections.Counter)
    by_backend = collections.defaultdict(collections.Counter)
    for e in rated:
        by_mode[e.get("mode") or "?"][e["rating"]] += 1
        by_backend[f"{e.get('backend') or '?'} / {e.get('model') or '?'}"][e["rating"]] += 1
    shown = [e for e in rated if e["rating"] == "down"] if args.down_only else rated
    if args.json:
        print(json.dumps({"events": len(events), "rated": len(rated), "counts": dict(by_rating), "by_mode": {k: dict(v) for k, v in by_mode.items()},
                          "by_backend": {k: dict(v) for k, v in by_backend.items()}, "entries": shown}, ensure_ascii=False, indent=1))
        return
    lines = [f"# Feedback review ({args.path})", "", f"events {len(events)}, answers with a current rating {len(rated)}: "
             f"up {by_rating.get('up', 0)}, down {by_rating.get('down', 0)}" + (f", since {args.since}" if args.since else ""), "",
             "| mode | up | down |", "|---|---|---|"]
    lines += [f"| {m} | {c.get('up', 0)} | {c.get('down', 0)} |" for m, c in sorted(by_mode.items())]
    lines += ["", "| backend / model | up | down |", "|---|---|---|"]
    lines += [f"| {b} | {c.get('up', 0)} | {c.get('down', 0)} |" for b, c in sorted(by_backend.items())]
    lines += ["", "## Thumbs down" if args.down_only else "## Rated answers (newest first)", ""]
    for e in sorted(shown, key=lambda x: x.get("ts", ""), reverse=True):
        mark = "👍" if e["rating"] == "up" else "👎"
        lines.append(f"### {mark} {e.get('ts', '')[:16]} · {e.get('mode') or '?'} · {e.get('backend') or '?'} / {e.get('model') or '?'}"
                     + (f" · {e['seconds']} s" if e.get("seconds") is not None else ""))
        lines.append(f"**Q:** {e.get('question', '')}" + (f"  _(requested mode {e['req_mode']})_" if e.get("req_mode") and e["req_mode"] != "auto" else ""))
        if e.get("comment"):
            lines.append(f"**Comment:** {e['comment']}")
        if e.get("queries"):
            lines.append(f"_queries: {' | '.join(e['queries'])}_")
        txt = (e.get("text") or "").strip()
        lines.append("")
        lines.append("> " + txt[:700].replace("\n", "\n> ") + ("…" if len(txt) > 700 else ""))
        for c in e.get("citations") or []:
            lines.append(f"- [{c.get('n')}] {c.get('citation')} ({c.get('para_id')})")
        lines.append("")
    out = "\n".join(lines)
    if args.export:
        with open(args.export, "w", encoding="utf-8") as f:
            f.write(out + "\n")
        print(f"wrote {args.export}")
    else:
        print(out)


if __name__ == "__main__":
    main()
