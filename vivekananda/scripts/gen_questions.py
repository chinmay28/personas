"""Phase 2b: question-for-passage generation with Ollama (qwen3:14b, thinking off), resumable.

  .venv/bin/python scripts/gen_questions.py --sample 1000            # eval sample (seeded)
  .venv/bin/python scripts/gen_questions.py --all                    # full run (hours; confirm with owner)
Output: data/qfp/questions.jsonl  (one line per question: id, para_id, section_id, kind, question, model)
"""
from __future__ import annotations

import argparse
import json
import os
import random
import re
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import requests

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import vklib  # noqa: E402

OUT = os.path.join(vklib.ROOT, "data", "qfp", "questions.jsonl")
LOG = os.path.join(vklib.ROOT, "data", "qfp", "gen.log")
DEFAULT_GENRES = "lecture,book_chapter,essay,notes,qa,interview,dialogue,letter,sayings"

SYSTEM = """You write search-evaluation questions for passages by Swami Vivekananda (1863-1902).
Given one passage, write exactly two questions that THIS passage answers:
1. "specific": a precise question that uses the passage's own key terms or names, answerable only from this passage.
2. "general": the natural question a modern reader might type into a search box that this passage would answer, in plain words, without rare proper nouns.
Rules: do not copy sentences from the passage; do not mention "the passage", "the text" or "the author"; questions may address him directly ("you") or not; one sentence each, ending with a question mark.
Reply with JSON only: {"specific": "...", "general": "..."}"""


def chat_json(prompt: str, model: str, timeout: int = 180) -> dict:
    r = requests.post(f"{vklib.OLLAMA}/v1/chat/completions", json={
        "model": model, "messages": [{"role": "system", "content": SYSTEM}, {"role": "user", "content": prompt}],
        "temperature": 0.7, "max_tokens": 300, "reasoning_effort": "none", "response_format": {"type": "json_object"},
    }, timeout=timeout)
    r.raise_for_status()
    txt = r.json()["choices"][0]["message"]["content"]
    txt = re.sub(r"^```(?:json)?|```$", "", txt.strip(), flags=re.M).strip()
    return json.loads(txt)


def main():
    ap = argparse.ArgumentParser()
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--sample", type=int)
    g.add_argument("--all", action="store_true")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--model", default="qwen3:14b")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--genres", default=DEFAULT_GENRES)
    ap.add_argument("--min-chars", type=int, default=300)
    ap.add_argument("--max-chars", type=int, default=4000)
    args = ap.parse_args()

    con = vklib.connect()
    genres = tuple(args.genres.split(","))
    rows = con.execute(
        f"SELECT p.id, p.section_id, p.text, s.genre FROM paragraphs p JOIN sections s ON s.id = p.section_id "
        f"WHERE s.kind = 'section' AND s.genre IN ({','.join('?' * len(genres))}) AND length(p.text) BETWEEN ? AND ? ORDER BY p.id",
        (*genres, args.min_chars, args.max_chars)).fetchall()
    pool = [dict(r) for r in rows]
    if args.sample:
        rng = random.Random(args.seed)
        pool = rng.sample(pool, min(args.sample, len(pool)))
    done = set()
    if os.path.exists(OUT):
        with open(OUT) as f:
            for line in f:
                done.add(json.loads(line)["para_id"])
    todo = [p for p in pool if p["id"] not in done]
    print(f"population={len(rows)} selected={len(pool)} done={len(done)} todo={len(todo)} model={args.model}", flush=True)
    lock = threading.Lock()
    out = open(OUT, "a")
    log = open(LOG, "a")
    t0 = time.time()
    n_ok = n_err = 0

    def work(p):
        prompt = f"Passage (genre: {p['genre']}):\n\n{p['text']}"
        for attempt in range(3):
            try:
                j = chat_json(prompt, args.model)
                qs = [(k, str(j[k]).strip()) for k in ("specific", "general") if j.get(k)]
                if not qs:
                    raise ValueError("no questions in reply")
                return p, qs, None
            except Exception as e:  # noqa: BLE001
                err = f"{type(e).__name__}: {e}"
                time.sleep(2 * (attempt + 1))
        return p, [], err

    with ThreadPoolExecutor(args.workers) as ex:
        futs = [ex.submit(work, p) for p in todo]
        for i, fut in enumerate(as_completed(futs), 1):
            p, qs, err = fut.result()
            with lock:
                if err:
                    n_err += 1
                    log.write(f"{time.strftime('%H:%M:%S')} ERR {p['id']} {err}\n"); log.flush()
                else:
                    n_ok += 1
                    for kind, q in qs:
                        out.write(json.dumps({"id": f"{p['id']}-{kind[0]}", "para_id": p["id"], "section_id": p["section_id"],
                                              "genre": p["genre"], "kind": kind, "question": q, "model": args.model}, ensure_ascii=False) + "\n")
                    out.flush()
                if i % 25 == 0 or i == len(todo):
                    el = time.time() - t0
                    msg = f"{i}/{len(todo)} ok={n_ok} err={n_err} {i / el:.2f} para/s eta {(len(todo) - i) / max(i / el, 1e-6) / 60:.0f} min"
                    print(msg, flush=True); log.write(msg + "\n"); log.flush()


if __name__ == "__main__":
    main()
