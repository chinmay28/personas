"""Phase 6 assembly: merge data/tiers/tier*.jsonl -> data/train.jsonl + data/holdout.jsonl + data/dataset_stats.md

  .venv/bin/python scripts/assemble_dataset.py [--holdout 0.05] [--seed 1] [--max-chars 30000] [--persona-verbatim]
Run2 default (2026-10-16): tiers 2 and 3 come from tier2_o/tier3_o.jsonl (oracle plans over the dialogue answer / letter paragraph), tier 1
persona prose is dropped, tier 6 is uncapped (persona = marked synthetic answers only), tier 3o is capped (--tier3-cap 300) and tier 4
(--tier4-cap 2300) so oracle does not swamp the mix, tier 9 (query expansion, no mode tag) joins like tier 8. --persona-verbatim restores the
run1 mix (tier2/tier3 verbatim persona, tier 6 capped at 25% of persona).
Checks on every record (PLAN.md 7, phase 6): oracle plans reference only block paragraphs with valid sentence ranges and <= 250 quoted words;
scholar sentences cite only block paragraphs; persona tags point into the block; every span in quotation marks verifies against the corpus
(quote guard). Records that fail a structural check are dropped and listed; verbatim tiers (1-3) only report guard misses (their text IS corpus
text). Caps: tier 6 <= 25% of persona records, tier 5 no_record <= 5% of tier 5, tier 8 = 10-15% of the total. Eval questions are excluded.
Holdout: 5% of each tier, stratified, seeded. Near-duplicate user turns (same normalized question) are collapsed (tier 9 only against itself: its questions come from the other tiers).
"""
from __future__ import annotations

import argparse
import collections
import glob
import json
import os
import random
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import modes  # noqa: E402
import traindata as td  # noqa: E402
import vklib  # noqa: E402

TAG_LINE = re.compile(r"^<mode>(\w+)</mode>\n?", re.S)


def split_assistant(rec: dict) -> tuple[str, str]:
    a = rec["messages"][2]["content"]
    m = TAG_LINE.match(a)
    return (m.group(1), a[m.end():]) if m else ("", a)


def question_of(rec: dict) -> str:
    u = rec["messages"][1]["content"]
    return u.split("QUESTION: ")[-1] if "QUESTION: " in u else u


def check(con, rec: dict, sent_counts: dict) -> tuple[bool, list[str]]:
    """Returns (keep, problems). Problems that do not drop the record start with 'warn:'."""
    mode, body = split_assistant(rec)
    probs: list[str] = []
    block = set(rec.get("block_ids", []))
    if rec["tier"] in (8, 9):
        return True, probs
    if mode != rec["mode"]:
        return False, ["mode tag mismatch"]
    if mode == "oracle":
        try:
            plan = json.loads(body)
        except json.JSONDecodeError:
            return False, ["oracle plan not JSON"]
        words = 0
        for q in plan.get("quotes", []):
            pid = str(q.get("para"))
            if pid not in block:
                return False, [f"oracle quote para {pid} not in block"]
            n = sent_counts.get(pid) or con.execute("SELECT json_array_length(sentences) FROM paragraphs WHERE id = ?", (pid,)).fetchone()[0]
            sent_counts[pid] = n
            a, b = int(q.get("from", 1)), int(q.get("to", 1))
            if not (1 <= a <= b <= max(1, n)):
                return False, [f"oracle range {a}-{b} invalid for {pid} ({n} sentences)"]
            words += len(" ".join(modes.split_sentences(vklib.get_paragraph(con, pid)["text"])[a - 1:b]).split()) if n else 0
        if words > modes.QUOTE_WORD_CAP + 30:
            probs.append(f"warn: quoted words {words}")
        for s in plan.get("cite_more", []):
            if s not in set(rec.get("section_ids", [])) | {p for p in block}:
                r = con.execute("SELECT 1 FROM sections WHERE id = ?", (s,)).fetchone()
                if not r:
                    return False, [f"cite_more unknown section {s}"]
        for b_ in plan.get("bridges", []):
            if '"' in b_ or "“" in b_:
                return False, ["bridge contains quotation marks"]
    elif mode == "scholar":
        try:
            plan = json.loads(body)
        except json.JSONDecodeError:
            return False, ["scholar plan not JSON"]
        for s in plan.get("sentences", []):
            ids = s.get("ids") or []
            if not ids or any(i not in block for i in ids):
                return False, [f"scholar sentence cites outside block: {ids}"]
            _, rep = modes.quote_guard(con, s["text"], None)
            if any(not r["verified"] for r in rep):
                return False, ["scholar quoted span not verbatim"]
    elif mode == "persona":
        bad = [t for t in modes.TAG_RE.findall(body) if t not in block]
        if bad:
            return False, [f"persona tags outside block: {bad[:3]}"]
        _, rep = modes.quote_guard(con, body, None)
        misses = [r for r in rep if not r["verified"]]
        if misses:
            if rec["tier"] in (1, 2, 3):
                probs.append(f"warn: guard miss on verbatim text ({len(misses)}): {misses[0]['quote'][:60]!r}")
            else:
                return False, [f"persona quoted span not verbatim: {misses[0]['quote'][:60]!r}"]
    else:
        return False, [f"unknown mode {mode}"]
    return True, probs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--holdout", type=float, default=0.05)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--max-chars", type=int, default=30000, help="drop records whose messages exceed this many characters (~8K tokens)")
    ap.add_argument("--tier8-share", type=float, default=0.12)
    ap.add_argument("--persona-verbatim", action="store_true", help="run1 mix: tiers 2/3 as verbatim persona, tier 6 capped")
    ap.add_argument("--tier3-cap", type=int, default=300)
    ap.add_argument("--tier4-cap", type=int, default=2300)
    args = ap.parse_args()
    con = vklib.connect()
    rng = random.Random(args.seed)
    evalq = {td.norm(q["question"]) for q in td.load_jsonl(os.path.join(vklib.ROOT, "eval", "questions.jsonl"))}

    names = ["tier1", "tier2", "tier3", "tier4", "tier5", "tier6", "tier8"] if args.persona_verbatim else ["tier1", "tier2_o", "tier3_o", "tier4", "tier5", "tier6", "tier8", "tier9"]
    recs = []
    for name in names:
        path = os.path.join(td.TIER_DIR, name + ".jsonl")
        if not os.path.exists(path):
            td.log(f"assemble: {name}.jsonl missing, skipped")
            continue
        recs += td.load_jsonl(path)
    td.log(f"assemble: {len(recs)} records from {td.TIER_DIR} ({', '.join(names)})")

    dropped: list[tuple[str, str]] = []
    warns: list[tuple[str, str]] = []
    seen_q: set[str] = set()
    seen_q9: set[str] = set()   # tier 9 reuses the other tiers' questions on purpose (expansion is a different task); dedup only within tier 9
    kept: list[dict] = []
    sent_counts: dict = {}
    for r in recs:
        q = td.norm(question_of(r))
        if q in evalq:
            dropped.append((r["id"], "eval question")); continue
        if r["tier"] == 9:
            if q in seen_q9:
                dropped.append((r["id"], "duplicate question")); continue
        elif r["tier"] != 8 and q in seen_q:
            dropped.append((r["id"], "duplicate question")); continue
        total = sum(len(m["content"]) for m in r["messages"])
        if total > args.max_chars:
            dropped.append((r["id"], f"too long ({total} chars)")); continue
        if r["tier"] not in (8, 9):
            r["messages"][0]["content"] = td.UNIFIED_SYSTEM   # stamp the current fixed system prompt (tier files may predate prompt edits)
        if not args.persona_verbatim and r["tier"] == 1 and r["mode"] == "persona":
            dropped.append((r["id"], "tier1 verbatim persona (run2)")); continue
        ok, probs = check(con, r, sent_counts)
        warns += [(r["id"], p) for p in probs if p.startswith("warn:")]
        if not ok:
            dropped.append((r["id"], probs[0] if probs else "check failed")); continue
        (seen_q9 if r["tier"] == 9 else seen_q).add(q)
        kept.append(r)
    td.log(f"assemble: kept {len(kept)}, dropped {len(dropped)}, warnings {len(warns)}")

    # caps
    by_tier = collections.defaultdict(list)
    for r in kept:
        by_tier[r["tier"]].append(r)
    for t in by_tier:
        rng.shuffle(by_tier[t])
    if args.persona_verbatim:
        persona_base = sum(1 for t in (1, 2, 3) for r in by_tier[t] if r["mode"] == "persona")
        cap6 = int(persona_base * 0.25 / 0.75) if persona_base else 0
        if len(by_tier[6]) > cap6:
            dropped += [(r["id"], "tier6 cap") for r in by_tier[6][cap6:]]
            by_tier[6] = by_tier[6][:cap6]
    else:
        for t, cap in ((3, args.tier3_cap), (4, args.tier4_cap)):
            if len(by_tier[t]) > cap:
                dropped += [(r["id"], f"tier{t} cap") for r in by_tier[t][cap:]]
                by_tier[t] = by_tier[t][:cap]
    nr = [r for r in by_tier[5] if r["meta"].get("no_record")]
    cap_nr = int(0.05 * len(by_tier[5]))
    if len(nr) > cap_nr:
        drop_ids = {r["id"] for r in nr[cap_nr:]}
        dropped += [(i, "tier5 no_record cap") for i in drop_ids]
        by_tier[5] = [r for r in by_tier[5] if r["id"] not in drop_ids]
    core = sum(len(v) for t, v in by_tier.items() if t not in (8, 9))
    cap8 = int(core * args.tier8_share / (1 - args.tier8_share))
    if len(by_tier[8]) > cap8:
        dropped += [(r["id"], "tier8 share cap") for r in by_tier[8][cap8:]]
        by_tier[8] = by_tier[8][:cap8]

    # stratified holdout
    train, hold = [], []
    for t in sorted(by_tier):
        rs = by_tier[t]
        k = max(1, int(round(len(rs) * args.holdout))) if rs else 0
        hold += rs[:k]
        train += rs[k:]
    rng.shuffle(train)
    rng.shuffle(hold)
    for name, rs in (("train.jsonl", train), ("holdout.jsonl", hold)):
        with open(os.path.join(vklib.ROOT, "data", name), "w") as f:
            for r in rs:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
    with open(os.path.join(td.TIER_DIR, "assemble_dropped.jsonl"), "w") as f:
        for i, why in dropped:
            f.write(json.dumps({"id": i, "why": why}) + "\n")

    # stats
    def tok(r):
        return sum(len(m["content"]) for m in r["messages"]) / 3.8
    lines = ["# Dataset stats", "", f"records: train {len(train)}, holdout {len(hold)}, dropped {len(dropped)}, warnings {len(warns)}", "",
             "| tier | mode | train | holdout | mean est. tokens | max est. tokens |", "|---|---|---|---|---|---|"]
    allr = train + hold
    for t in sorted(by_tier):
        for mode in sorted({r["mode"] for r in by_tier[t]}):
            tr = [r for r in train if r["tier"] == t and r["mode"] == mode]
            ho = [r for r in hold if r["tier"] == t and r["mode"] == mode]
            toks = [tok(r) for r in tr + ho] or [0]
            lines.append(f"| {t} | {mode or 'chat'} | {len(tr)} | {len(ho)} | {int(sum(toks) / len(toks))} | {int(max(toks))} |")
    lines += ["", "| mode | total |", "|---|---|"]
    for mode, n in collections.Counter(r["mode"] or "chat" for r in allr).most_common():
        lines.append(f"| {mode} | {n} |")
    lines += ["", f"gold paragraph injected into the block (oracle tiers): {sum(1 for r in allr if r.get('gold_injected'))}",
              f"persona records with the extrapolation marker (tier 6): {sum(1 for r in allr if r['tier'] == 6 and r['meta'].get('marker'))} / {sum(1 for r in allr if r['tier'] == 6)}",
              f"records over 8K est. tokens: {sum(1 for r in allr if tok(r) > 8192)}", "", "## Drop reasons", ""]
    for why, n in collections.Counter(re.sub(r"\d+", "N", w) for _, w in dropped).most_common():
        lines.append(f"- {n}: {why}")
    lines += ["", "## Warnings (first 20)", ""] + [f"- {i}: {w}" for i, w in warns[:20]]
    with open(os.path.join(vklib.ROOT, "data", "dataset_stats.md"), "w") as f:
        f.write("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
