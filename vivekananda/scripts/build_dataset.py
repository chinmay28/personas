"""Phase 6: build the training tiers (PLAN.md 6.2) into data/tiers/tierN[_part].jsonl. Resumable: existing record ids are skipped.

  .venv/bin/python scripts/build_dataset.py tier1              # Q&A sessions (oracle) + interviews (LLM-extracted, verbatim-verified)
  .venv/bin/python scripts/build_dataset.py tier2 [--limit N]  # dialogues: direct speech parsed; narrative ones converted by LLM + verified
  .venv/bin/python scripts/build_dataset.py tier3 [--limit N]  # letters: LLM-synthesized incoming letter -> his letter (persona)
  .venv/bin/python scripts/build_dataset.py tier4 [--limit N]  # reverse-prompted paragraphs -> oracle selection plans
  .venv/bin/python scripts/build_dataset.py tier5 [--limit N]  # scholar: templated + generated "about him" questions -> cited JSON sentences
  .venv/bin/python scripts/build_dataset.py tier6 [--limit N]  # persona: modern questions -> grounded first-person answers (capped later)
  .venv/bin/python scripts/build_dataset.py tier8 [--n 1200]   # general chat mix (Dolly 15k), generic system prompt, no mode tag
  .venv/bin/python scripts/build_dataset.py oracle23           # run2: tiers 2 and 3 re-rendered as oracle plans -> tier2_o.jsonl, tier3_o.jsonl
  .venv/bin/python scripts/build_dataset.py tier9 [--n 1000]   # run2: query-expansion records (question -> JSON search queries), no mode tag
tier6 generates questions for any topic in TOPICS that has none yet (run2 added TOPICS_RUN2, 14 questions each) and answers the new ones.
Models: VK_GEN_MODEL (default qwen3:14b) for synthesis/extraction, VK_PERSONA_MODEL for tier 6 answers. Log: data/tiers/build.log
"""
from __future__ import annotations

import argparse
import json
import os
import random
import re
import sys
import urllib.request

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import modes  # noqa: E402
import traindata as td  # noqa: E402
import vklib  # noqa: E402

EVAL_Q = os.path.join(vklib.ROOT, "eval", "questions.jsonl")


def eval_question_set() -> set[str]:
    try:
        return {td.norm(q["question"]) for q in td.load_jsonl(EVAL_Q)}
    except FileNotFoundError:
        return set()


def section_paragraphs(c, sid: str) -> list[dict]:
    rows = c.execute("SELECT id, para_idx, text, sentences FROM paragraphs WHERE section_id = ? ORDER BY para_idx", (sid,)).fetchall()
    return [{"id": r["id"], "para_idx": r["para_idx"], "text": r["text"], "sentences": json.loads(r["sentences"]) if isinstance(r["sentences"], str) else r["sentences"]} for r in rows]


def sections_of(c, genre: str, min_chars: int = 0) -> list[dict]:
    rows = c.execute("SELECT * FROM sections WHERE genre = ? AND text != '' AND char_count >= ? ORDER BY \"order\"", (genre, min_chars)).fetchall()
    return [dict(r) for r in rows]


# ================================================================== tier 1: Q&A sessions and interviews
Q_RE = re.compile(r"^Q\.?\s*[—–-]*\s*")
A_RE = re.compile(r"^A\.?\s*[—–-]*\s*")

INTERVIEW_EXTRACT = """You extract question-and-answer pairs from a newspaper interview with Swami Vivekananda (1890s). The text mixes the
reporter's narration with his quoted speech. Return JSON only: {"pairs": [{"question": "...", "answer": "..."}]}.
- "question": the interviewer's question as asked, or a one-sentence reconstruction of what was put to him, addressed to him in the second person.
- "answer": his words COPIED EXACTLY, character for character, from inside ONE paragraph of the text (no reporter words, no changes, no ellipses, at least 25 words).
Skip exchanges whose answer is not his own quoted speech. Return at most 12 pairs."""


def tier1(args):
    c = td.con()
    w = td.TierWriter(1)
    n_qa = 0
    for sec in sections_of(c, "qa"):
        paras = section_paragraphs(c, sec["id"])
        i = 0
        while i < len(paras):
            t = paras[i]["text"].strip()
            if Q_RE.match(t) and i + 1 < len(paras) and A_RE.match(paras[i + 1]["text"].strip()):
                question = Q_RE.sub("", t).strip()
                ans = paras[i + 1]
                if len(question) >= 10 and len(ans["text"]) >= 25:
                    rid = f"t1-qa-{ans['id']}"
                    if rid not in w.done:
                        ctx, info = td.build_context(question, gold_pids=[ans["id"]])
                        plan, rendered = td.oracle_plan(ctx, ans["id"], question)
                        rec = td.make_record(rid, 1, "oracle", question, ctx, info, td.plan_json(plan),
                                             {"section_id": sec["id"], "para_ids": [ans["id"]], "kind": "qa"},
                                             {"rendered": rendered.get("text", ""), "quoted_words": rendered.get("quoted_words", 0), "citation": vklib.citation(sec)})
                        w.write(rec)
                    n_qa += 1
                i += 2
            else:
                i += 1
    td.log(f"tier1 qa: {n_qa} pairs ({w.n_new} new)")

    # interviews: LLM extraction, every answer verified verbatim inside one paragraph
    secs = sections_of(c, "interview")
    def do(sec):
        paras = section_paragraphs(td.con(), sec["id"])
        j = td.llm_json(INTERVIEW_EXTRACT, sec["text"][:14000], max_tokens=2500, temperature=0.2)
        out = []
        for k, pair in enumerate((j or {}).get("pairs", [])):
            q, a = str(pair.get("question", "")).strip(), str(pair.get("answer", "")).strip()
            if len(q) < 10 or td.words(a) < 20:
                continue
            na = td.norm(a)
            hit = next((p for p in paras if na and na in td.norm(p["text"])), None)
            if hit is None:
                continue
            sents = [hit["text"][s:e] for s, e in hit["sentences"]] or [hit["text"]]
            # whole-sentence match -> oracle plan over that range; otherwise persona prose with his verbatim words
            rng = None
            for i in range(len(sents)):
                for jj in range(i, len(sents)):
                    if td.norm(" ".join(sents[i:jj + 1])) == na:
                        rng = (i + 1, jj + 1)
                        break
                if rng:
                    break
            rid = f"t1-int-{hit['id']}-{k}"
            if rid in w.done:
                continue
            ctx, info = td.build_context(q, gold_pids=[hit["id"]])
            if rng and td.words(" ".join(sents[rng[0] - 1:rng[1]])) <= 250:
                plan = {"quotes": [{"para": hit["id"], "from": rng[0], "to": rng[1]}], "bridges": [td.bridge_for(sec, hit["id"])], "cite_more": []}
                rendered = modes.render_oracle(td.con(), ctx, plan)
                rec = td.make_record(rid, 1, "oracle", q, ctx, info, td.plan_json(plan), {"section_id": sec["id"], "para_ids": [hit["id"]], "kind": "interview"},
                                     {"rendered": rendered.get("text", ""), "quoted_words": rendered.get("quoted_words", 0), "citation": vklib.citation(sec)})
            else:
                rec = td.make_record(rid, 1, "persona", q, ctx, info, a + f" [{hit['id']}]", {"section_id": sec["id"], "para_ids": [hit["id"]], "kind": "interview"},
                                     {"verbatim": True, "citation": vklib.citation(sec)})
            out.append(rec)
        return out
    for recs in td.pmap(do, secs, label="tier1 interviews"):
        for r in recs or []:
            w.write(r)
    td.log(f"tier1 total written: {len(w.done)} ({w.n_new} new this run)")


# ================================================================== tier 2: conversations and dialogues
SPEAKER_RE = re.compile(r"^((?:Mrs?\.|Miss|Dr\.|Sister|Swami|Sri|Shri)?\s?[A-Z][a-z]+(?: [A-Z][a-z]+){0,3}):\s+")
HIS = {"swamiji", "swami", "swami vivekananda", "the swami", "a", "vivekananda"}

NARRATIVE_CONVERT = """You convert a third-person reminiscence of a conversation with Swami Vivekananda into first-person exchanges, for an archive.
Return JSON only: {"exchanges": [{"user": "...", "answer": "..."}]}.
- "user": what was asked of him or the situation put to him, in the second person, one to three sentences, as the questioner.
- "answer": what he said, in the first person as himself. Where the text quotes his words, copy them EXACTLY; where it reports them indirectly,
  render them in the first person without adding any idea that is not in the text. Do not describe him from outside ("he said", "Swamiji smiled").
  No quotation marks except around text copied exactly. 40 to 250 words each.
Return at most 6 exchanges, only those where the text records what he actually said."""


def tier2(args):
    c = td.con()
    w = td.TierWriter(2)
    secs = sections_of(c, "dialogue")
    n_direct = 0
    narrative = []
    for sec in secs:
        paras = section_paragraphs(c, sec["id"])
        marked = [(p, SPEAKER_RE.match(p["text"].strip())) for p in paras]
        if sum(1 for _, m in marked if m) < 4:
            if sec["char_count"] >= 1200 and not sec["title"].upper().startswith("PREFACE"):
                narrative.append(sec)
            continue
        i = 0
        while i < len(paras):
            p, m = marked[i]
            if m and m.group(1).strip().lower() not in HIS:
                question = p["text"].strip()[m.end():].strip()
                j = i + 1
                answer_paras = []
                while j < len(paras):
                    p2, m2 = marked[j]
                    if m2 and m2.group(1).strip().lower() in HIS:
                        answer_paras.append((p2, p2["text"].strip()[m2.end():].strip()))
                    elif m2:
                        break
                    elif answer_paras and len(p2["text"]) > 40 and not p2["text"].startswith(("(", "[")):
                        answer_paras.append((p2, p2["text"].strip()))   # continuation of his speech
                    else:
                        break
                    j += 1
                if answer_paras and len(question) >= 12:
                    answer = "\n\n".join(t for _, t in answer_paras)
                    if 150 <= len(answer) <= 3500:
                        rid = f"t2-direct-{answer_paras[0][0]['id']}"
                        if rid not in w.done and args.limit and len(w.done) >= args.limit:
                            break
                        if rid not in w.done:
                            ctx, info = td.build_context(question, exclude_section=sec["id"])
                            rec = td.make_record(rid, 2, "persona", question, ctx, info, answer,
                                                 {"section_id": sec["id"], "para_ids": [pp["id"] for pp, _ in answer_paras], "kind": "dialogue_direct"},
                                                 {"verbatim": True, "citation": vklib.citation(sec), "questioner": m.group(1).strip()})
                            w.write(rec)
                        n_direct += 1
                i = max(j, i + 1)
            else:
                i += 1
    td.log(f"tier2 direct-speech pairs: {n_direct} ({w.n_new} new)")

    def convert(sec):
        if args.limit and len(w.done) >= args.limit:
            return []
        j = td.llm_json(NARRATIVE_CONVERT, sec["text"][:12000], max_tokens=2500, temperature=0.3)
        out = []
        src_words = td.content_words(sec["text"])
        for k, ex in enumerate((j or {}).get("exchanges", [])):
            u, a = str(ex.get("user", "")).strip(), str(ex.get("answer", "")).strip()
            if len(u) < 10 or not (40 <= td.words(a) <= 300):
                continue
            if re.search(r"\b(Swamiji|the Swami)\b", a) or not re.search(r"\b(I|me|my|mine|we|our|us)\b", a):
                continue
            aw = td.content_words(a)
            overlap = len(aw & src_words) / max(1, len(aw))
            if overlap < 0.6:
                continue
            rid = f"t2-narr-{sec['id']}-{k}"
            if rid in w.done:
                continue
            ctx, info = td.build_context(u, exclude_section=sec["id"])
            guarded, qreport = modes.quote_guard(td.con(), a, None)
            out.append(td.make_record(rid, 2, "persona", u, ctx, info, guarded, {"section_id": sec["id"], "para_ids": [], "kind": "dialogue_converted"},
                                      {"verbatim": False, "citation": vklib.citation(sec), "source_overlap": round(overlap, 2), "quote_guard": qreport,
                                       "audit_source_excerpt": sec["text"][:1500]}))
        return out
    for recs in td.pmap(convert, narrative, label="tier2 narrative"):
        for r in recs or []:
            w.write(r)
    td.log(f"tier2 total written: {len(w.done)} ({w.n_new} new this run)")


# ================================================================== tier 3: letters
LETTER_SYNTH = """You write the letter that a correspondent sent to Swami Vivekananda BEFORE he wrote the letter supplied; his letter is the reply
to yours. You are the correspondent, writing in the first person, in the register of the 1890s, 60 to 150 words. You have not yet received anything
from him, so you cannot thank him for advice, mention his letter, his words or his reply, or agree with what he says. Instead raise the matters his
reply answers: ask the questions he responds to, report the news he reacts to, make the requests he grants or refuses, confess the doubts he addresses.
Mention no person, place or fact that his reply does not imply. No salutation line and no signature: just the body.
Return JSON only: {"letter": "..."}"""
REPLY_LEAK_RE = re.compile(r"\b(your (letter|words|reply|advice|teachings|instructions|counsel|observations|remarks|wisdom)|as you (said|advised|instructed|wrote|suggest)|"
                           r"you have (said|written|advised)|thank you for (your|the))\b", re.I)


def tier3(args):
    c = td.con()
    w = td.TierWriter(3)
    secs = [s for s in sections_of(c, "letter") if 300 <= s["char_count"] <= 9000]
    seen_bodies = set()
    jobs = []
    for sec in secs:
        paras = section_paragraphs(c, sec["id"])
        body, info = td.letter_body(paras)
        text = "\n\n".join(p["text"].strip() for p in body)
        if len(text) < 200:
            continue
        key = td.norm(text[:300])
        if key in seen_bodies:
            continue
        seen_bodies.add(key)
        rid = f"t3-{sec['id']}"
        if rid in w.done:
            continue
        jobs.append((rid, sec, body, text, info))
    if args.limit:
        jobs = jobs[:max(0, args.limit - len(w.done))]
    td.log(f"tier3: {len(secs)} letters, {len(jobs)} to synthesize")

    def do(job):
        rid, sec, body, text, info = job
        to = vklib.title_case(sec.get("recipient_full") or sec.get("recipient") or sec["title"])
        prompt = f"Recipient: {to}\nDate of his reply: {sec.get('date') or 'unknown'}\nWritten from: {sec.get('place') or 'unknown'}\n\nHIS REPLY:\n{text}"
        j = td.llm_json(LETTER_SYNTH, prompt, max_tokens=400, temperature=0.6)
        letter = str((j or {}).get("letter", "")).strip()
        if not (30 <= td.words(letter) <= 220):
            return None
        if REPLY_LEAK_RE.search(letter):
            return None
        ctx, cinfo = td.build_context(letter, exclude_section=sec["id"])
        return td.make_record(rid, 3, "persona", letter, ctx, cinfo, text,
                              {"section_id": sec["id"], "para_ids": [p["id"] for p in body], "kind": "letter"},
                              {"verbatim": True, "citation": vklib.citation(sec), "recipient": to, "date": sec.get("date"), "body_info": info})
    for rec in td.pmap(do, jobs, label="tier3 letters"):
        if rec:
            w.write(rec)
    td.log(f"tier3 total written: {len(w.done)} ({w.n_new} new this run)")


# ================================================================== tier 4: reverse-prompted paragraphs -> oracle plans
THIRD_PERSON_RE = re.compile(r"\b(Swamiji|the Swami|Swami Vivekananda|Vivekananda)\b")
QUOTA = {"lecture": 1200, "book_chapter": 600, "notes": 300, "essay": 250, "letter": 350, "dialogue": 200, "sayings": 50, "interview": 30, "qa": 20}
REWRITE = """Rewrite each question so that it addresses Swami Vivekananda directly in the second person ("you", "your"), as if the user were
speaking to him. Keep the meaning, keep the names of other people and works, remove phrases such as "according to Vivekananda", "the author",
"the speaker". Fix grammar after the change ("does Vivekananda say" -> "do you say"). Return JSON only: {"questions": ["..."]} with exactly the
same number of questions, in the same order."""


def regex_second_person(q: str) -> str:
    q = re.sub(r"\b(According to|In the view of|For) (Swami )?Vivekananda,?\s*", "In your view, ", q)
    q = re.sub(r"\bdoes (Swami )?Vivekananda\b", "do you", q)
    q = re.sub(r"\b(Swami )?Vivekananda's\b", "your", q)
    q = re.sub(r"\b(Swami )?Vivekananda\b", "you", q)
    q = re.sub(r"\bthe speaker's\b", "your", q)
    q = re.sub(r"\bthe speaker\b", "you", q)
    q = re.sub(r"\byou (says|considers|suggests|describes|believes|argues|thinks|explains|views|sees|means|claims|states|uses|defines|recommends)\b",
               lambda m: "you " + m.group(1)[:-1], q)
    return q[:1].upper() + q[1:]


def tier4(args):
    c = td.con()
    w = td.TierWriter(4)
    qs = [q for q in td.load_jsonl(os.path.join(vklib.ROOT, "data", "qfp", "questions_full.jsonl")) if q["id"].endswith("-d1")]
    rng = random.Random(42)
    rng.shuffle(qs)
    seen_text, picked = set(), []
    counts = {g: 0 for g in QUOTA}
    for q in qs:
        g = q["genre"]
        if g not in QUOTA or counts[g] >= QUOTA[g]:
            continue
        p = vklib.get_paragraph(c, q["para_id"])
        key = td.norm(p["text"][:300])
        if key in seen_text or td.words(p["text"]) < 40 or THIRD_PERSON_RE.search(p["text"]):
            continue
        seen_text.add(key)
        counts[g] += 1
        picked.append(q)
    picked = [q for q in picked if f"t4-{q['para_id']}" not in w.done]
    if args.limit:
        picked = picked[:max(0, args.limit - len(w.done))]
    td.log(f"tier4: {len(picked)} paragraphs to build; quota fill {counts}")

    # second-person rewrite in batches of 20
    batches = [picked[i:i + 20] for i in range(0, len(picked), 20)]
    def rewrite(batch):
        j = td.llm_json(REWRITE, json.dumps([q["question"] for q in batch], ensure_ascii=False), max_tokens=1200, temperature=0.1)
        out = (j or {}).get("questions") or []
        if len(out) != len(batch) or not all(isinstance(x, str) and len(x) > 8 for x in out):
            return [regex_second_person(q["question"]) for q in batch]
        return [str(x).strip() for x in out]
    rewritten = td.pmap(rewrite, batches, label="tier4 rewrite")
    jobs = [(q, r) for batch, rs in zip(batches, rewritten) for q, r in zip(batch, rs or [regex_second_person(x["question"]) for x in batch])]

    def build(job):
        q, question = job
        ctx, info = td.build_context(question, gold_pids=[q["para_id"]])
        plan, rendered = td.oracle_plan(ctx, q["para_id"], question)
        if rendered.get("fallthrough"):
            return None
        return td.make_record(f"t4-{q['para_id']}", 4, "oracle", question, ctx, info, td.plan_json(plan),
                              {"section_id": q["section_id"], "para_ids": [q["para_id"]], "kind": "qfp", "qfp_id": q["id"]},
                              {"rendered": rendered.get("text", ""), "quoted_words": rendered.get("quoted_words", 0), "original_question": q["question"], "genre": q["genre"]})
    for rec in td.pmap(build, jobs, workers=4, label="tier4 build"):
        if rec:
            w.write(rec)
    td.log(f"tier4 total written: {len(w.done)} ({w.n_new} new this run)")


# ================================================================== tier 5: scholar
SCHOLAR_Q = """Write questions a reader might put to Swami Vivekananda about what HE said, wrote or did in the text supplied: questions about him
and his statements on that occasion, addressed to him in the second person ("What did you tell the audience at ... about ...?", "What did you
write to ... about ...?", "What was your opinion of ...?", "Why did you say that ...?"). Each must be answerable from the text, name the concrete
topic, and must not be a question about the idea in the abstract. Return JSON only: {"questions": ["...", "..."]} with 2 questions."""


def month_year(date: str | None) -> str:
    if not date:
        return ""
    m = re.match(r"(\d{4})-(\d{2})", date)
    if not m:
        return date[:4]
    months = ["January", "February", "March", "April", "May", "June", "July", "August", "September", "October", "November", "December"]
    return f"{months[int(m.group(2)) - 1]} {m.group(1)}"


def scholar_questions(c, rng: random.Random) -> list[dict]:
    out = []
    for sec in sections_of(c, "lecture", 1500):
        title, work = td._clean_title(sec), td._work(sec)
        if title == work or sec.get("depth", 2) < 2:
            q = f"What did you say in {work}?"
        else:
            q = rng.choice([f"What did you say in your lecture {title}?", f"What did you speak about in {title}?",
                            f"What were the main points of your lecture {title}?"])
        out.append({"q": q, "sec": sec, "kind": "template"})
    for sec in sections_of(c, "letter", 900):
        to = vklib.title_case(sec.get("recipient_full") or sec.get("recipient") or sec["title"])
        my = month_year(sec.get("date"))
        place = vklib.title_case(sec["place"].split(",")[0].strip()) if sec.get("place") else ""
        opts = [f"What did you write to {to} in {my}?" if my else f"What did you write to {to}?"]
        if place:
            opts.append(f"What did you tell {to} in your letter from {place}?")
        out.append({"q": rng.choice(opts), "sec": sec, "kind": "template"})
    for sec in sections_of(c, "book_chapter", 1500):
        title, work = td._clean_title(sec), td._work(sec)
        out.append({"q": f"What did you write in {work}, {title}?" if title != work else f"What did you write in {work}?", "sec": sec, "kind": "template"})
    for sec in sections_of(c, "essay", 1500):
        out.append({"q": f"What did you write in your essay {td._clean_title(sec)}?", "sec": sec, "kind": "template"})
    return out


def tier5(args):
    c = td.con()
    w = td.TierWriter(5)
    rng = random.Random(5)
    evalq = eval_question_set()
    jobs = [j for j in scholar_questions(c, rng) if td.norm(j["q"]) not in evalq]
    # generated topical questions for the larger sections
    big = [s for g in ("lecture", "book_chapter", "essay", "notes", "letter") for s in sections_of(c, g, 2000)]
    rng.shuffle(big)
    big = big[:args.gen_sections]
    td.log(f"tier5: {len(jobs)} templated questions; generating topical questions for {len(big)} sections")

    def gen(sec):
        j = td.llm_json(SCHOLAR_Q, f"{vklib.citation(sec)}\n\n{sec['text'][:3500]}", max_tokens=300, temperature=0.5)
        return [{"q": str(q).strip(), "sec": sec, "kind": "generated"} for q in (j or {}).get("questions", []) if isinstance(q, str) and len(q) > 15][:2]
    for lst in td.pmap(gen, big, label="tier5 questions"):
        jobs += [j for j in (lst or []) if td.norm(j["q"]) not in evalq]
    jobs = [j for j in jobs if f"t5-{td.stable_hash(j['q'])}" not in w.done]
    rng.shuffle(jobs)
    if args.limit:
        jobs = jobs[:max(0, args.limit - len(w.done))]
    td.log(f"tier5: {len(jobs)} scholar answers to generate with {td.GEN_MODEL}")

    def answer(job):
        q, sec = job["q"], job["sec"]
        ctx, info = td.build_context(q)
        block_ids = set(info["block_ids"])
        raw = modes.chat([{"role": "system", "content": modes.SCHOLAR_SYSTEM}, {"role": "user", "content": td.user_content(ctx, q)}],
                         td.GEN_MODEL, json_mode=True, max_tokens=1200)
        try:
            plan = json.loads(re.sub(r"^```(?:json)?|```$", "", raw.strip(), flags=re.M))
        except json.JSONDecodeError:
            return None
        kept, dropped, guard = [], 0, []
        for it in plan.get("sentences") or []:
            if not isinstance(it, dict):
                continue
            text = str(it.get("text", "")).strip()
            ids = [str(x) for x in (it.get("ids") or []) if str(x) in block_ids]
            if not text or not ids:
                dropped += 1
                continue
            text, rep = modes.quote_guard(td.con(), text, ctx)
            guard += rep
            kept.append({"text": text, "ids": ids})
        no_record = bool(plan.get("no_record")) or not kept
        if no_record:
            body = json.dumps({"sentences": [], "no_record": True}, ensure_ascii=False, separators=(",", ":"))
        else:
            if len(kept) < 3:
                return None
            body = json.dumps({"sentences": kept}, ensure_ascii=False, separators=(",", ":"))
        return td.make_record(f"t5-{td.stable_hash(q)}", 5, "scholar", q, ctx, info, body,
                              {"section_id": sec["id"], "para_ids": [], "kind": job["kind"]},
                              {"no_record": no_record, "n_sentences": len(kept), "dropped": dropped, "quote_guard": guard,
                               "rendered": " ".join(f"{s['text']} [{','.join(s['ids'])}]" for s in kept), "citation": vklib.citation(sec)})
    for rec in td.pmap(answer, jobs, label="tier5 answers"):
        if rec:
            w.write(rec)
    td.log(f"tier5 total written: {len(w.done)} ({w.n_new} new this run)")


# ================================================================== tier 6: persona, modern questions
TOPICS = ["burnout and overwork in a modern office job", "choosing between a safe career and a risky passion", "social media and the craving for approval",
          "loneliness in a big city", "anxiety about the future", "a failing marriage", "raising children who are glued to screens", "caring for ageing parents",
          "losing faith after a tragedy", "being bullied at school or work", "procrastination and lack of discipline", "money worries and debt",
          "jealousy of a friend's success", "guilt over past mistakes", "addiction to the phone, games or drink", "fear of death of a loved one",
          "whether to emigrate for a better life", "feeling useless after retirement", "a scientist doubting religion", "climate change and despair about the planet",
          "artificial intelligence and what makes humans special", "working for a corporation whose values one dislikes", "online outrage and arguments",
          "how to meditate with a restless mind in a noisy flat", "starting a charity or volunteer work", "dealing with a cruel boss", "body image and self-hatred",
          "the pressure of exams and grades", "being an immigrant between two cultures", "a friend who has become fanatical", "finding purpose in a boring job",
          "whether to become a monk or nun today", "interfaith marriage", "forgiving an abusive parent", "the fear of public speaking", "creative block for an artist",
          "wealth and whether to give it away", "grief after a miscarriage", "vegetarianism and modern food ethics", "whether prayer works",
          "doubt about one's own abilities", "political polarization among friends", "how to serve the poor without becoming cynical", "insomnia and a racing mind"]
TOPICS_RUN2 = [
    "being replaced by automation or AI at work", "a start-up founder facing failure", "working from home and never switching off", "a nurse or doctor exhausted by patients' suffering",
    "a teacher losing faith in the school system", "a soldier or police officer troubled by what the job demands", "a lawyer asked to defend something unjust",
    "a journalist facing pressure to distort the truth", "an athlete after a career-ending injury", "a musician or actor craving fame", "an engineer building weapons",
    "a civil servant surrounded by corruption", "a farmer ruined by debt and drought", "a gig worker with no security", "unemployment and shame in front of the family",
    "dating apps and finding a partner", "staying celibate in a culture obsessed with sex", "pornography and compulsive habits", "a long-distance relationship",
    "deciding whether to have children", "infertility and the longing for a child", "divorce and shared custody of children", "a spouse who has lost interest in spiritual life",
    "an arranged marriage one does not want", "coming out as gay or transgender to a religious family", "a daughter forbidden to study or work by her family",
    "sexual harassment at the workplace", "domestic violence and whether to leave", "a teenager rebelling against parents", "adopting a child",
    "being the only vegetarian or believer among friends", "losing one's native language and roots abroad", "racism faced in a Western country",
    "caste discrimination faced today", "religious conversion pressure on the poor", "communal riots between Hindus and Muslims", "nationalism and hatred of other countries",
    "whether to vote or stay out of politics", "a politician who wants to stay honest", "protests, strikes and civil disobedience", "censorship and free speech online",
    "terrorism done in the name of religion", "refugees and whether to take them in", "the gap between the very rich and the very poor today", "consumerism and endless shopping",
    "cryptocurrency, gambling and quick money", "advertising and manipulation of desire", "fast fashion and the workers who make it", "eating meat and factory farming",
    "nuclear weapons and the fear of war", "space travel and whether other worlds matter", "genetic engineering and designer babies", "surveillance by governments and companies",
    "video games and virtual worlds", "a child who wants to be an influencer", "cancel culture and public shaming", "conspiracy theories in the family",
    "evolution, physics and whether God is needed", "near-death experiences and life after death", "psychedelic drugs and spiritual experience", "mindfulness apps and paid meditation courses",
    "fake gurus and spiritual scams", "yoga as a fitness industry", "the Ramakrishna Mission and monastic orders today", "Hindu temples and rituals that seem empty to the young",
    "converting to Hinduism as a Westerner", "atheist parents raising a child who wants to pray", "a Muslim or Christian asking whether their path is enough",
    "Buddhism, mindfulness and whether to abandon Hinduism", "astrology, horoscopes and auspicious dates", "superstition in the village and modern medicine",
    "depression and whether medication is weakness", "panic attacks and the body's fear", "an eating disorder", "self-harm and thoughts of suicide", "therapy versus spiritual practice",
    "chronic illness and pain", "a terminal diagnosis", "caring for a parent with dementia", "euthanasia and the right to die", "organ donation and the body after death",
    "cremation versus burial far from home", "a child's death", "a pet's death", "surviving an accident others did not", "trauma from war or disaster",
    "a prisoner seeking redemption", "a recovering alcoholic", "lying to protect someone", "whistleblowing on one's employer", "paying bribes to get things done",
    "cheating in exams because everyone does", "plagiarism and taking credit", "revenge against someone who wronged me", "road rage and anger in traffic",
    "a neighbour from a hated community", "beggars and whether to give", "tipping, charity and feeling used", "inherited wealth and guilt", "minimalism and giving up possessions",
    "learning Sanskrit or the scriptures as an adult", "which translation of the Gita or Upanishads to read", "a daily practice for a busy householder", "a vow one cannot keep",
    "fasting and festivals in a modern schedule", "pilgrimage versus a holiday", "teaching children the Gita without preaching", "running a meditation group at work",
    "being asked to give a speech about India abroad", "the English language versus the mother tongue", "studying abroad and whether to return to India", "a village that wants a school and a temple",
    "writing a novel or making a film about religion", "a programmer who wants to build something good", "science students who mock spirituality", "physical weakness and bodybuilding culture",
    "old age alone in a flat", "a widow or widower starting again", "retirement savings versus giving", "writing a will and leaving things to children who fight",
    "what you would say to India today", "what you would make of the world after your death", "women leading nations and companies", "whether machines can be conscious",
]
TOPICS += TOPICS_RUN2
Q_PER_TOPIC_RUN2 = 8
MODERN_Q = """Write {n} distinct questions a person today might ask Swami Vivekananda for personal advice about the topic given. Each is in the first
person, concrete, 15 to 40 words, with a real-life detail (a job, a family, a habit, a fear). Vary the asker (age, sex, country, faith). Return JSON only: {{"questions": ["..."]}}"""
MARKER_RE = re.compile(r"(did not speak|not a question of my day|never (addressed|spoke)|was not before me|not (something|a matter) I (spoke|addressed)|"
                       r"you ask (me )?of (a )?(thing|matter|world)|I did not know|not of my day|had no (occasion|cause) to|in my day)", re.I)


def tier6(args):
    w = td.TierWriter(6)
    rng = random.Random(6)
    qpath = os.path.join(td.TIER_DIR, "tier6_questions.jsonl")
    questions = td.load_jsonl(qpath) if os.path.exists(qpath) else []
    have = {q["topic"] for q in questions}
    todo = [t for t in TOPICS if t not in have]
    if todo:
        n = 10 if not questions else Q_PER_TOPIC_RUN2
        lists = td.pmap(lambda t: [(t, str(q).strip()) for q in ((td.llm_json(MODERN_Q.format(n=n), f"Topic: {t}", max_tokens=1200, temperature=0.8) or {}).get("questions") or [])],
                        todo, label="tier6 questions")
        seen = {td.norm(q["question"]) for q in questions}
        new_qs = []
        for lst in lists:
            for t, q in lst or []:
                k = td.norm(q)
                if len(q) < 20 or k in seen:
                    continue
                seen.add(k)
                new_qs.append({"id": f"t6-{td.stable_hash(q)}", "topic": t, "question": q})
        with open(qpath, "a") as f:
            for q in new_qs:
                f.write(json.dumps(q, ensure_ascii=False) + "\n")
        td.log(f"tier6: {len(new_qs)} new questions for {len(todo)} topics")
        questions += new_qs
    rng.shuffle(questions)
    jobs = [q for q in questions if q["id"] not in w.done]
    if args.limit:
        jobs = jobs[:max(0, args.limit - len(w.done))]
    td.log(f"tier6: {len(questions)} questions, {len(jobs)} answers to generate with {td.PERSONA_MODEL}")

    def answer(q):
        ctx, info = td.build_context(q["question"])
        raw = modes.chat([{"role": "system", "content": modes.PERSONA_SYSTEM}, {"role": "user", "content": td.user_content(ctx, q["question"])}],
                         td.PERSONA_MODEL, temperature=0.6, max_tokens=700)
        if not raw:
            return None
        text, guard = modes.quote_guard(td.con(), raw.strip(), ctx)
        block_ids = set(info["block_ids"])
        bad = [t for t in modes.TAG_RE.findall(text) if t not in block_ids]
        text = modes.TAG_RE.sub(lambda m: m.group(0) if m.group(1) in block_ids else "", text)
        text = re.sub(r" +([.,;:])", r"\1", text)
        if not (80 <= td.words(text) <= 320) or re.search(r"\b(as an ai|language model)\b", text, re.I):
            return None
        return td.make_record(q["id"], 6, "persona", q["question"], ctx, info, text, {"section_id": None, "para_ids": [], "kind": "modern"},
                              {"topic": q["topic"], "marker": bool(MARKER_RE.search(text[:400])), "quote_guard": guard, "invalid_tags": bad,
                               "n_tags": len(modes.TAG_RE.findall(text)), "model": td.PERSONA_MODEL})
    for rec in td.pmap(answer, jobs, label="tier6 answers"):
        if rec:
            w.write(rec)
    td.log(f"tier6 total written: {len(w.done)} ({w.n_new} new this run)")


# ================================================================== tier 8: general chat mix
DOLLY_URL = "https://huggingface.co/datasets/databricks/databricks-dolly-15k/resolve/main/databricks-dolly-15k.jsonl"


def tier8(args):
    w = td.TierWriter(8)
    path = os.path.join(td.TIER_DIR, "dolly-15k.jsonl")
    if not os.path.exists(path):
        td.log(f"tier8: downloading {DOLLY_URL}")
        urllib.request.urlretrieve(DOLLY_URL, path)
    rows = td.load_jsonl(path)
    rng = random.Random(8)
    rng.shuffle(rows)
    cats = {"open_qa": 0.3, "general_qa": 0.2, "brainstorming": 0.15, "creative_writing": 0.1, "classification": 0.1, "closed_qa": 0.1, "summarization": 0.05}
    want = {k: int(args.n * v) for k, v in cats.items()}
    n = 0
    for r in rows:
        c = r.get("category")
        if want.get(c, 0) <= 0 or not r.get("response") or len(r["response"]) > 2500 or len(r.get("context", "")) > 3000:
            continue
        user = r["instruction"] + (f"\n\n{r['context']}" if r.get("context") else "")
        rid = f"t8-{td.stable_hash(user)}"
        rec = {"id": rid, "tier": 8, "mode": "", "source": {"section_id": None, "para_ids": [], "kind": c}, "block_ids": [], "section_ids": [], "gold_injected": False,
               "messages": [{"role": "system", "content": td.GENERIC_SYSTEM}, {"role": "user", "content": user}, {"role": "assistant", "content": r["response"].strip()}],
               "meta": {"dataset": "databricks-dolly-15k", "license": "CC BY-SA 3.0"}}
        w.write(rec)
        want[c] -= 1
        n += 1
    td.log(f"tier8 total written: {len(w.done)} ({w.n_new} new this run)")


# ================================================================== run2: tiers 2 and 3 as oracle plans
def _pick_gold(c, para_ids: list[str], question: str) -> dict | None:
    """The answer paragraph with the most question overlap (ties: earlier wins); paragraphs under 15 words are skipped."""
    q = td.content_words(question)
    best, best_score = None, -1.0
    for i, pid in enumerate(para_ids):
        p = vklib.get_paragraph(c, pid)
        if td.words(p["text"]) < 15:
            continue
        score = len(q & td.content_words(p["text"])) - 0.01 * i
        if score > best_score:
            best, best_score = p, score
    return best


def oracle23(args):
    """Re-render the verbatim persona tiers (dialogue answers, letters) as oracle records: same user turn, the block now includes the source
    paragraph (injected if retrieval missed it), the target is a selection plan quoting the best window of the answer paragraph with a
    metadata bridge. Writes tier2_o.jsonl / tier3_o.jsonl; the originals are left in place (assemble_dataset.py --persona-verbatim uses them)."""
    c = td.con()
    for t in (2, 3):
        src = os.path.join(td.TIER_DIR, f"tier{t}.jsonl")
        if not os.path.exists(src):
            td.log(f"oracle23: {src} missing, skipped")
            continue
        w = td.TierWriter(t, "o")
        recs = td.load_jsonl(src)
        jobs = []
        for r in recs:
            rid = r["id"].replace(f"t{t}-", f"t{t}o-", 1)
            if rid in w.done or not r["source"].get("para_ids"):
                continue
            jobs.append((rid, r))
        if args.limit:
            jobs = jobs[:max(0, args.limit - len(w.done))]
        td.log(f"oracle23: tier {t}: {len(recs)} source records, {len(jobs)} to render")

        def do(job):
            rid, r = job
            question = r["messages"][1]["content"].split("QUESTION: ")[-1]
            gold = _pick_gold(td.con(), r["source"]["para_ids"], question)
            if gold is None:
                return None
            ctx, info = td.build_context(question, gold_pids=[gold["id"]])
            plan, rendered = td.oracle_plan(ctx, gold["id"], question)
            if rendered.get("fallthrough") or not rendered.get("text"):
                return None
            meta = {"citation": r["meta"].get("citation"), "from_persona_record": r["id"], "rendered": rendered["text"],
                    "quoted_words": rendered.get("quoted_words")}
            for k in ("questioner", "recipient", "date"):
                if k in r["meta"]:
                    meta[k] = r["meta"][k]
            source = dict(r["source"], kind=r["source"]["kind"] + "_oracle", gold_para=gold["id"])
            return td.make_record(rid, t, "oracle", question, ctx, info, td.plan_json(plan), source, meta)
        for rec in td.pmap(do, jobs, label=f"oracle23 tier{t}"):
            if rec:
                w.write(rec)
        td.log(f"oracle23: tier{t}o total written: {len(w.done)} ({w.n_new} new this run)")


# ================================================================== run2: tier 9, query expansion (the 8B expands its own retrieval queries)
def tier9(args):
    """question -> {"queries": [...]} under modes.EXPAND_SYSTEM, written by the gen model. Questions are sampled from the other tiers' user
    turns (short ones only) plus the tier 6 question pool, never from the eval set."""
    w = td.TierWriter(9)
    rng = random.Random(9)
    evalq = eval_question_set()
    pool: dict[str, str] = {}
    for t in (1, 4, 5, 6, "2_o"):
        path = os.path.join(td.TIER_DIR, f"tier{t}.jsonl")
        if not os.path.exists(path):
            continue
        for r in td.load_jsonl(path):
            q = r["messages"][1]["content"].split("QUESTION: ")[-1].strip()
            if 15 <= len(q) <= 300 and td.norm(q) not in evalq:
                pool.setdefault(td.norm(q), q)
    qpath = os.path.join(td.TIER_DIR, "tier6_questions.jsonl")
    if os.path.exists(qpath):
        for r in td.load_jsonl(qpath):
            if td.norm(r["question"]) not in evalq:
                pool.setdefault(td.norm(r["question"]), r["question"])
    qs = list(pool.values())
    rng.shuffle(qs)
    jobs = [q for q in qs if f"t9-{td.stable_hash(q)}" not in w.done][:max(0, args.n - len(w.done))]
    td.log(f"tier9: pool {len(qs)} questions, {len(jobs)} expansions to generate with {td.GEN_MODEL}")

    def do(q):
        queries = modes.expand_queries(q, td.GEN_MODEL)
        queries = [x.strip() for x in queries if 2 <= len(x.split()) <= 10 and not re.search(r"\b(Vivekananda|Swami|Swamiji)\b", x)]
        if len(queries) < 2:
            return None
        body = json.dumps({"queries": queries[:3]}, ensure_ascii=False, separators=(",", ":"))
        return {"id": f"t9-{td.stable_hash(q)}", "tier": 9, "mode": "expand", "source": {"section_id": None, "para_ids": [], "kind": "expand"},
                "block_ids": [], "section_ids": [], "gold_injected": False,
                "messages": [{"role": "system", "content": modes.EXPAND_SYSTEM}, {"role": "user", "content": q}, {"role": "assistant", "content": body}],
                "meta": {"model": td.GEN_MODEL}}
    for rec in td.pmap(do, jobs, label="tier9 expansions"):
        if rec:
            w.write(rec)
    td.log(f"tier9 total written: {len(w.done)} ({w.n_new} new this run)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("tier", choices=["tier1", "tier2", "tier3", "tier4", "tier5", "tier6", "tier8", "oracle23", "tier9"])
    ap.add_argument("--limit", type=int, default=0, help="stop once this many records exist in the tier file (sample runs)")
    ap.add_argument("--n", type=int, default=1200, help="tier8 size (tier9: default 1000)")
    ap.add_argument("--gen-sections", type=int, default=400, help="tier5: sections to generate topical questions for")
    args = ap.parse_args()
    if args.tier == "tier9" and args.n == 1200:
        args.n = 1000
    td.log(f"=== {args.tier} start (gen model {td.GEN_MODEL}, persona model {td.PERSONA_MODEL})")
    globals()[args.tier](args)


if __name__ == "__main__":
    main()
