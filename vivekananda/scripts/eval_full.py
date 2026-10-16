"""Phase 8 harness: run eval/questions.jsonl and eval/position_probes.jsonl through a backend with the real retrieval pipeline,
score with automatic checks plus an LLM judge, and compare runs.

  .venv/bin/python scripts/eval_full.py run --tag prompted14b --backend prompted --model qwen3:14b [--judge qwen3:14b]
                                            [--limit N] [--include-extras] [--no-judge] [--workers 2] [--only questions|probes]
  .venv/bin/python scripts/eval_full.py compare prompted14b unified8b

Answers are cached per question in eval/results/full_<tag>/answers.jsonl (resumable: rerun the same tag to continue); judge verdicts in
judged.jsonl; the summary is written to eval/results/full_<tag>.json and .md.

Metrics (PLAN.md section 8): router/mode accuracy + confusion; oracle verbatim rate (must be 1.0), fallthrough rate, quoted words vs cap,
judged relevance; scholar citation coverage, judged support rate, no-record rate; persona marker rate, 8-gram overlap with the corpus
outside quotation marks (memorization); all modes: quote-guard violations (spans that failed verification and were unquoted),
third-person self-references, judged first-person consistency and anachronisms; position probes: judged agreement with the documented stance.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import engine  # noqa: E402
import modes  # noqa: E402
import traindata  # noqa: E402
import vklib  # noqa: E402

RESULTS = os.path.join(vklib.ROOT, "eval", "results")
QFILE = os.path.join(vklib.ROOT, "eval", "questions.jsonl")
PFILE = os.path.join(vklib.ROOT, "eval", "position_probes.jsonl")
LABELS = ["scholar", "oracle", "persona", "mixed"]
THIRD_PERSON_RE = re.compile(r"\b(Swami Vivekananda|Vivekananda|Swamiji|the Swami|Narendra(nath)?)\b")
MARKER_RE = re.compile(r"(did not (speak|address|write|treat|take up)|never (spoke|addressed|wrote|treated)|not (a )?(question|matter|thing) of my (day|time|age)"
                       r"|was not (a question|known|before me)|beyond (my|what I)|I cannot claim|had no occasion|in my day there was no|I did not live to)", re.I)
QUOTED_RE = re.compile(r"[“\"][^”\"]+[”\"]")

JUDGE_VOICE = """You are grading an answer written as if spoken by Swami Vivekananda (1863-1902) in the first person.
Reply with JSON only: {"first_person": true|false, "anachronism": true|false, "generic": true|false, "notes": "<short>"}
first_person: the answer consistently speaks as him (no "Vivekananda said", no "the Swami", no assistant framing).
anachronism: the answer asserts, as his own knowledge, events, people, technology or facts after 1902 (reasoning about a modern situation the
user raised is NOT an anachronism; claiming he witnessed or knew it is).
generic: the answer is generic inspirational prose that could come from anyone and does not rest on his specific ideas."""

JUDGE_SUPPORT = """You check whether cited passages support sentences. For each numbered sentence, say whether the passages listed for it
state or clearly imply its claim (paraphrase is fine; the sentence may attribute the claim to the speaker himself, e.g. "I wrote that...").
Reply with JSON only: {"results": [{"n": 1, "supported": true|false}, ...]}"""

JUDGE_RELEVANCE = """Does the quoted material answer the question (fully or in substantial part)? The quotes are verbatim passages chosen to
answer it. Reply with JSON only: {"relevant": true|false, "notes": "<short>"}"""

JUDGE_STANCE = """You compare an answer with a documented position. Reply with JSON only:
{"agreement": "agree"|"disagree"|"unclear", "notes": "<short>"}
agree: the answer takes the documented position (or is consistent with it); disagree: it contradicts it; unclear: it does not take a position."""


def judge_json(system: str, user: str, model: str) -> dict | None:
    for t in range(3):
        try:
            raw = modes.chat([{"role": "system", "content": system}, {"role": "user", "content": user}], model, json_mode=True, temperature=0.0, max_tokens=400)
            j = modes.parse_json(raw)
            if j is not None:
                return j
        except Exception as e:  # noqa: BLE001
            if t == 2:
                print(f"judge failed: {e!r}", flush=True)
            time.sleep(1)
    return None


# ------------------------------------------------------------------ automatic checks
CITATION_RE = re.compile(r"\[(?:Vol \d+|Front matter)[^\]]*\]")


def unquoted(text: str) -> str:
    """Prose outside quotation marks and outside rendered citations (a cited title such as 'With the Swami Vivekananda at Madura' is not
    a third-person self-reference)."""
    return CITATION_RE.sub(" ", QUOTED_RE.sub(" ", text))


def ngram_overlap(con, text: str, n: int = 8, stride: int = 4) -> tuple[int, int]:
    """(hits, windows): n-word windows of the unquoted text that exist verbatim (as an FTS phrase) somewhere in the corpus."""
    words = re.findall(r"[A-Za-z][A-Za-z']*", unquoted(re.sub(r"\[\d+\]", " ", text)))
    hits = windows = 0
    for i in range(0, max(0, len(words) - n + 1), stride):
        phrase = " ".join(w.replace("'", "''") for w in words[i:i + n])
        windows += 1
        try:
            if con.execute('SELECT 1 FROM fts_para WHERE fts_para MATCH ? LIMIT 1', (f'"{phrase}"',)).fetchone():
                hits += 1
        except Exception:  # noqa: BLE001
            pass
    return hits, windows


def check_answer(con, r: dict, expected_mode: str | None) -> dict:
    parts = r.get("parts") or [r]
    guard = [g for p in parts for g in (p.get("quote_guard") or [])]
    text = r.get("text", "")
    stats = r.get("stats") or {}
    row = {"mode": r.get("mode"), "mode_tag": r.get("mode_tag"), "expected": expected_mode, "cascade": r.get("cascade"), "seconds": r.get("seconds"),
           "quotes": len(guard), "quotes_verified": sum(1 for g in guard if g["verified"]), "guard_violations": sum(1 for g in guard if not g["verified"]),
           "third_person": len(THIRD_PERSON_RE.findall(unquoted(text))), "citations": len(r.get("citations") or []), "chars": len(text)}
    if r.get("mode") == "oracle":
        row.update(quoted_words=r.get("quoted_words"), over_cap=(r.get("quoted_words") or 0) > modes.QUOTE_WORD_CAP, see_also=len(r.get("see_also") or []))
    if r.get("mode") == "scholar":
        row.update(sentences=stats.get("sentences"), cited=stats.get("sentences_cited"), dropped_uncited=stats.get("dropped_uncited"),
                   model_sentences=stats.get("model_sentences"), no_record="no record" in text.lower())
    if r.get("mode") in ("persona", "scholar", "mixed"):
        h, w = ngram_overlap(con, text)
        row.update(ngram_hits=h, ngram_windows=w)
    if r.get("mode") == "persona":
        head = " ".join(modes.split_sentences(text)[:2])
        row.update(marker=bool(MARKER_RE.search(head)), cited=stats.get("sentences_cited"), sentences=stats.get("sentences"))
    return row


# ------------------------------------------------------------------ judging
def judge_answer(con, item: dict, r: dict, judge: str) -> dict:
    out = {}
    text = r.get("text", "")
    v = judge_json(JUDGE_VOICE, f"QUESTION: {item['question']}\n\nANSWER:\n{text}", judge)
    if v:
        out["voice"] = {k: v.get(k) for k in ("first_person", "anachronism", "generic", "notes")}
    if r.get("mode") == "oracle":
        q = judge_json(JUDGE_RELEVANCE, f"QUESTION: {item['question']}\n\nQUOTED ANSWER:\n{text}", judge)
        if q:
            out["relevant"] = bool(q.get("relevant"))
    if r.get("mode") == "scholar" and r.get("citations"):
        sents = [s for s in modes.split_sentences(text) if re.search(r"\[\d+\]", s)]
        cmap = {c["n"]: c for c in r["citations"]}
        lines, ptexts = [], {}
        for i, s in enumerate(sents, 1):
            ns = [int(x) for x in re.findall(r"\[(\d+)\]", s)]
            for n in ns:
                if n in cmap and n not in ptexts:
                    ptexts[n] = vklib.get_paragraph(con, cmap[n]["para_id"])["text"]
            lines.append(f"({i}) {re.sub(r'\[\d+\]', '', s).strip()}  -> passages {ns}")
        passages = "\n\n".join(f"[{n}] {t}" for n, t in sorted(ptexts.items()))
        j = judge_json(JUDGE_SUPPORT, f"SENTENCES\n" + "\n".join(lines) + f"\n\nPASSAGES\n{passages}", judge)
        if j and isinstance(j.get("results"), list):
            res = [bool(x.get("supported")) for x in j["results"] if isinstance(x, dict)][:len(sents)]
            out["support"] = {"sentences": len(sents), "supported": sum(res), "judged": len(res)}
    if item.get("stance"):
        j = judge_json(JUDGE_STANCE, f"DOCUMENTED POSITION: {item['stance']}\n(supporting citation: {item['support']['citation']})\n\nQUESTION: {item['question']}\n\nANSWER:\n{text}", judge)
        if j:
            out["agreement"] = str(j.get("agreement", "unclear")).lower()
            out["agreement_notes"] = j.get("notes", "")
    return out


# ------------------------------------------------------------------ run
def load_items(include_extras: bool, only: str | None) -> list[dict]:
    items = []
    if only in (None, "questions"):
        for q in traindata.load_jsonl(QFILE):
            if q.get("extra") and not include_extras:
                continue
            items.append({"set": "questions", **q})
    if only in (None, "probes"):
        for p in traindata.load_jsonl(PFILE):
            items.append({"set": "probes", **p})
    return items


def run(args):
    d = os.path.join(RESULTS, f"full_{args.tag}")
    os.makedirs(d, exist_ok=True)
    apath, jpath = os.path.join(d, "answers.jsonl"), os.path.join(d, "judged.jsonl")
    answers = {r["id"]: r for r in traindata.load_jsonl(apath)} if os.path.exists(apath) else {}
    judged = {r["id"]: r for r in traindata.load_jsonl(jpath)} if os.path.exists(jpath) else {}
    items = load_items(args.include_extras, args.only)
    if args.limit:
        items = items[:args.limit]
    model = args.model or engine.default_model(args.backend)
    todo = [it for it in items if it["id"] not in answers]
    print(f"{args.tag}: {len(items)} items, {len(answers)} cached, {len(todo)} to answer with {args.backend}/{model}", flush=True)
    t0 = time.time()

    def do_answer(it):
        r = engine.answer(traindata.con(), it["question"], "auto", args.backend, model, 8, expand=not args.no_expand)
        return {"id": it["id"], "set": it["set"], "question": it["question"], "expected": it.get("expected_mode"), "result": engine.slim(r) | {"raw": r.get("raw", "")}}

    with ThreadPoolExecutor(max_workers=args.workers) as ex, open(apath, "a") as f:
        for i, rec in enumerate(ex.map(lambda it: _safe(do_answer, it), todo), 1):
            if rec:
                answers[rec["id"]] = rec
                f.write(json.dumps(rec, ensure_ascii=False) + "\n"); f.flush()
            if i % 10 == 0 or i == len(todo):
                print(f"answered {i}/{len(todo)} ({time.time() - t0:.0f}s)", flush=True)

    if not args.no_judge:
        jtodo = [it for it in items if it["id"] in answers and it["id"] not in judged]
        print(f"judging {len(jtodo)} answers with {args.judge}", flush=True)

        def do_judge(it):
            return {"id": it["id"], **judge_answer(traindata.con(), it, answers[it["id"]]["result"], args.judge)}

        with ThreadPoolExecutor(max_workers=args.workers) as ex, open(jpath, "a") as f:
            for i, rec in enumerate(ex.map(lambda it: _safe(do_judge, it), jtodo), 1):
                if rec:
                    judged[rec["id"]] = rec
                    f.write(json.dumps(rec, ensure_ascii=False) + "\n"); f.flush()
                if i % 10 == 0 or i == len(jtodo):
                    print(f"judged {i}/{len(jtodo)} ({time.time() - t0:.0f}s)", flush=True)

    summary = summarize(args.tag, args.backend, model, items, answers, judged)
    json.dump(summary, open(os.path.join(RESULTS, f"full_{args.tag}.json"), "w"), indent=1, ensure_ascii=False)
    md = to_markdown([summary])
    open(os.path.join(RESULTS, f"full_{args.tag}.md"), "w").write(md)
    print(md)
    print("wrote", os.path.join(RESULTS, f"full_{args.tag}.json"))


def _safe(fn, it):
    try:
        return fn(it)
    except Exception as e:  # noqa: BLE001
        print(f"{it.get('id')}: {e!r}", flush=True)
        return None


# ------------------------------------------------------------------ summary
def summarize(tag: str, backend: str, model: str, items: list[dict], answers: dict, judged: dict) -> dict:
    con = traindata.con()
    rows = []
    for it in items:
        a = answers.get(it["id"])
        if not a:
            continue
        row = {"id": it["id"], "set": it["set"], **check_answer(con, a["result"], it.get("expected_mode"))}
        row["judge"] = judged.get(it["id"], {})
        rows.append(row)
    qrows = [r for r in rows if r["set"] == "questions"]
    prows = [r for r in rows if r["set"] == "probes"]
    conf = defaultdict(Counter)
    for r in qrows:
        conf[r["expected"]][r["mode"]] += 1
    correct = sum(conf[l][l] for l in LABELS)
    by_mode = {m: [r for r in rows if r["mode"] == m] for m in LABELS}

    def rate(num, den):
        return round(num / den, 3) if den else None

    o, s, p = by_mode["oracle"], by_mode["scholar"], by_mode["persona"]
    vj = [r["judge"].get("voice") for r in rows if r["judge"].get("voice")]
    sup = [r["judge"]["support"] for r in s if r["judge"].get("support")]
    rel = [r["judge"]["relevant"] for r in o if "relevant" in r["judge"]]
    agr = Counter(r["judge"].get("agreement", "n/a") for r in prows)
    m = {
        "tag": tag, "backend": backend, "model": model, "n": len(rows), "n_questions": len(qrows), "n_probes": len(prows),
        "mode_accuracy": rate(correct, len(qrows)), "persona_when_oracle_expected": conf["oracle"]["persona"],
        "confusion": {e: dict(c) for e, c in conf.items()}, "modes_served": dict(Counter(r["mode"] for r in rows)),
        "cascade_rate": rate(sum(1 for r in rows if r["cascade"]), len(rows)),
        "oracle_n": len(o), "oracle_verbatim_rate": rate(sum(r["quotes_verified"] for r in o), sum(r["quotes"] for r in o)),
        "oracle_mean_quoted_words": rate(sum(r.get("quoted_words") or 0 for r in o), len(o)), "oracle_over_cap": sum(1 for r in o if r.get("over_cap")),
        "oracle_relevant_rate": rate(sum(rel), len(rel)),
        "scholar_n": len(s), "scholar_citation_coverage": rate(sum(r.get("cited") or 0 for r in s), sum(r.get("sentences") or 0 for r in s)),
        "scholar_dropped_uncited": sum(r.get("dropped_uncited") or 0 for r in s), "scholar_no_record_rate": rate(sum(1 for r in s if r.get("no_record")), len(s)),
        "scholar_support_rate": rate(sum(x["supported"] for x in sup), sum(x["judged"] for x in sup)),
        "persona_n": len(p), "persona_marker_rate": rate(sum(1 for r in p if r.get("marker")), len(p)),
        "persona_cited_sentence_rate": rate(sum(r.get("cited") or 0 for r in p), sum(r.get("sentences") or 0 for r in p)),
        "persona_8gram_overlap": rate(sum(r.get("ngram_hits") or 0 for r in p), sum(r.get("ngram_windows") or 0 for r in p)),
        "guard_violations": sum(r["guard_violations"] for r in rows), "third_person_refs": sum(r["third_person"] for r in rows),
        "judged_first_person_rate": rate(sum(1 for v in vj if v.get("first_person")), len(vj)),
        "judged_anachronism_rate": rate(sum(1 for v in vj if v.get("anachronism")), len(vj)),
        "judged_generic_rate": rate(sum(1 for v in vj if v.get("generic")), len(vj)),
        "probe_agree_rate": rate(agr["agree"], len(prows)), "probe_agreement": dict(agr),
        "mean_seconds": rate(sum(r["seconds"] or 0 for r in rows), len(rows)),
        "failures": [{"id": r["id"], "why": why} for r in rows for why in failure_reasons(r)],
        "rows": rows,
    }
    return m


def failure_reasons(r: dict) -> list[str]:
    why = []
    if r["set"] == "questions" and r["expected"] and r["mode"] != r["expected"]:
        why.append(f"mode {r['mode']} != expected {r['expected']}")
    if r["guard_violations"]:
        why.append(f"{r['guard_violations']} quoted span(s) failed verification")
    if r["third_person"]:
        why.append(f"{r['third_person']} third-person self-reference(s)")
    if r["mode"] == "oracle" and r.get("over_cap"):
        why.append("quoted words over cap")
    if r["mode"] == "scholar" and (r.get("dropped_uncited") or 0) > 3:
        why.append(f"{r['dropped_uncited']} uncited sentences dropped")
    if r["mode"] == "persona" and not r.get("marker"):
        why.append("persona answer without an extrapolation marker")
    j = r.get("judge") or {}
    v = j.get("voice") or {}
    if v and not v.get("first_person"):
        why.append("judge: not consistently first person")
    if v.get("anachronism"):
        why.append("judge: anachronism")
    if v.get("generic"):
        why.append("judge: generic prose")
    if j.get("relevant") is False:
        why.append("judge: oracle quotes not relevant")
    sp = j.get("support")
    if sp and sp["judged"] and sp["supported"] < sp["judged"]:
        why.append(f"judge: {sp['judged'] - sp['supported']}/{sp['judged']} scholar sentences unsupported")
    if j.get("agreement") == "disagree":
        why.append("judge: contradicts documented stance")
    return why


KEYS = ["n", "mode_accuracy", "persona_when_oracle_expected", "cascade_rate", "oracle_n", "oracle_verbatim_rate", "oracle_mean_quoted_words", "oracle_over_cap",
        "oracle_relevant_rate", "scholar_n", "scholar_citation_coverage", "scholar_dropped_uncited", "scholar_no_record_rate", "scholar_support_rate",
        "persona_n", "persona_marker_rate", "persona_cited_sentence_rate", "persona_8gram_overlap", "guard_violations", "third_person_refs",
        "judged_first_person_rate", "judged_anachronism_rate", "judged_generic_rate", "probe_agree_rate", "mean_seconds"]


def to_markdown(summaries: list[dict]) -> str:
    head = "| metric | " + " | ".join(f"{s['tag']} ({s['backend']}, {s['model']})" for s in summaries) + " |"
    lines = [head, "|---|" + "---|" * len(summaries)]
    for k in KEYS:
        lines.append(f"| {k} | " + " | ".join(str(s.get(k)) for s in summaries) + " |")
    for s in summaries:
        lines.append(f"\n**{s['tag']} confusion (expected -> predicted):** " + "; ".join(f"{e}: {dict(c)}" for e, c in s["confusion"].items()))
        cats = Counter(re.sub(r"\d+", "N", f["why"]) for f in s["failures"])
        lines.append(f"\n**{s['tag']} failure categories:** " + "; ".join(f"{k} ({v})" for k, v in cats.most_common()))
    return "\n".join(lines) + "\n"


def compare(args):
    sums = [json.load(open(os.path.join(RESULTS, f"full_{t}.json"))) for t in args.tags]
    md = to_markdown(sums)
    print(md)
    path = os.path.join(RESULTS, "full_compare_" + "_vs_".join(args.tags) + ".md")
    open(path, "w").write(md)
    print("wrote", path)


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--tag", required=True)
    r.add_argument("--backend", default=engine.DEFAULT_BACKEND, choices=engine.BACKENDS)
    r.add_argument("--model", default=None)
    r.add_argument("--judge", default=os.environ.get("VK_JUDGE", "qwen3:14b"))
    r.add_argument("--limit", type=int, default=0)
    r.add_argument("--include-extras", action="store_true")
    r.add_argument("--only", choices=["questions", "probes"], default=None)
    r.add_argument("--no-judge", action="store_true")
    r.add_argument("--no-expand", action="store_true")
    r.add_argument("--workers", type=int, default=2)
    r.set_defaults(fn=run)
    c = sub.add_parser("compare")
    c.add_argument("tags", nargs="+")
    c.set_defaults(fn=compare)
    args = ap.parse_args()
    args.fn(args)


if __name__ == "__main__":
    main()
