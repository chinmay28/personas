"""Phase 3: prompted oracle / scholar / persona on top of the hybrid index, plus the quote guard and router.

All modes answer in the first person as Vivekananda. Mode = grounding level:
  scholar  recounts what he said/wrote, every claim tagged with a paragraph id -> rendered as a citation
  oracle   selection plan over numbered sentences of retrieved paragraphs -> verbatim quotes by construction
  persona  grounded extrapolation, explicitly marked, quote-guarded
"""
from __future__ import annotations

import difflib
import json
import os
import re
import sys
import time
from dataclasses import dataclass, field

import requests

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import vklib  # noqa: E402

DEV_MODEL = os.environ.get("VK_MODEL", "qwen3:14b")  # qwen3.8:27b was removed 2026-10-15 (offloaded to CPU, 5x slower)
QUOTE_WORD_CAP = 250
GUARD_THRESHOLD = 0.95

PERSONA_PREAMBLE = (
    "You are Swami Vivekananda, speaking in the first person, in your own voice and period register. "
    "You never speak of yourself in the third person.")

SCHOLAR_SYSTEM = PERSONA_PREAMBLE + """
Task: recount what you actually said, wrote or did, using ONLY the numbered passages supplied. Paraphrase in the first person
("As I told the audience at Chicago...", "I wrote to Alasinga from Yokohama that..."). Reply with JSON only:
{"sentences": [{"text": "one first-person sentence", "ids": ["p-000161"]}, ...]}
Every sentence needs at least one id of a passage that supports it; a sentence you cannot support must not be written: no framing, no summary,
no claims about your life or travels beyond what the passages state. Use the passage headers (work, letter recipient, place, date) as facts
you may state. Do not use quotation marks unless the words are copied exactly from a passage. If the passages do not cover the question,
reply {"sentences": [], "no_record": true}. Write 4 to 9 sentences. Name the occasion or recipient once, at the start; do not repeat it in
every sentence, and vary the openings ("I wrote...", "There I argued...", "My point was...") instead of "I told X that" each time."""

ORACLE_SYSTEM = PERSONA_PREAMBLE + """
Task: choose which of your own recorded sentences answer the question. You will see numbered passages, each with numbered sentences.
Reply with JSON only, of the form:
{"quotes":[{"para":"p-000161","from":1,"to":3}], "bridges":["One short first-person sentence introducing the quote, no quotation marks."],
 "cite_more":["s-0022"]}
Rules: pick 1 or 2 contiguous sentence ranges that best answer the question, at most about 200 words in total; give one bridge sentence per
quote (same order), plain first person, never containing quotation marks or new claims; put further relevant section ids in cite_more.
If no passage actually answers the question, reply exactly {"fallthrough": true}."""

PERSONA_SYSTEM = PERSONA_PREAMBLE + """
Task: the question is one you never addressed directly. Reason from your recorded principles, grounded in the passages supplied, and answer in
the first person. Open with an explicit marker such as "I did not speak to this directly, but..." or "This was not a question of my day, yet...".
Do not put anything inside quotation marks unless it is copied exactly from a passage. Cite a passage id in square brackets, e.g. [p-000161],
after sentences that rest on it. Length: 120-250 words."""

ROUTER_SYSTEM = """Classify a user question addressed to Swami Vivekananda into exactly one label.

oracle   - a question about a teaching, concept, practice or idea that his works answer directly, whatever the phrasing:
           "What is Maya?", "What are the three Gunas?", "How should one meditate?", "Why is man divine?", "What is the ideal of
           womanhood?". Served by quoting him. Prefer oracle over scholar whenever the question is about the idea rather than about him.
scholar  - a question about HIM: what he said or wrote on an occasion or to a person, how his views developed, his life, travels,
           opinions of people and movements: "What did you say at Chicago?", "What did you write to Alasinga?", "What was your
           opinion of the Theosophists?", "What did you say about vegetarianism?". Served by recounting and citing him.
persona  - personal advice, a modern situation, later events, politics, or anything else he never addressed in his works
           (burnout, a job choice, social media, a family dilemma, a later leader or movement).
mixed    - clearly two parts: a scholar/oracle part about his views AND a personal-advice part ("What did you say about X, and how should I...").

You also see retrieval evidence (how strongly the corpus matches). Prefer oracle over persona whenever a matching passage exists.
Reply with JSON only: {"mode": "<label>", "reason": "<one short sentence>"}"""


NUM_CTX = int(os.environ.get("VK_NUM_CTX", "8192"))  # Ollama's default slot is 4096 tokens and silently truncates longer prompts


def chat(messages: list[dict], model: str = DEV_MODEL, json_mode: bool = False, think: bool = False, temperature: float = 0.3, max_tokens: int = 900) -> str:
    """Native Ollama chat endpoint (the OpenAI-compatible one cannot set num_ctx per request)."""
    body = {"model": model, "messages": messages, "stream": False, "think": think,
            "options": {"temperature": temperature, "num_predict": max_tokens, "num_ctx": NUM_CTX}}
    if json_mode:
        body["format"] = "json"
    r = requests.post(f"{vklib.OLLAMA}/api/chat", json=body, timeout=600)
    r.raise_for_status()
    return r.json()["message"]["content"]


# ------------------------------------------------------------------ retrieval block
@dataclass
class Passage:
    para: dict
    section: dict
    citation: str
    sentences: list[str]


@dataclass
class Context:
    question: str
    passages: list[Passage]
    sections: list[dict]            # section-level hits for "see also"
    hits: list[vklib.Hit] = field(default_factory=list)
    queries: list[str] = field(default_factory=list)

    def by_id(self, pid: str) -> Passage | None:
        return next((p for p in self.passages if p.para["id"] == pid), None)


EXPAND_SYSTEM = """You turn a question addressed to Swami Vivekananda into search queries over his Complete Works.
Reply with JSON only: {"queries": ["...", "..."]} with 2 or 3 short queries (3-8 words each) using the concrete terms the matching passages
would contain: topic words, Sanskrit terms, names of other people, places, lectures or works. Never include his own name ("Vivekananda",
"Swami", "Swamiji"): the passages are his own words and do not name him. Do not include words like "say", "think", "view"."""


def expand_queries(question: str, model: str = DEV_MODEL) -> list[str]:
    try:
        raw = chat([{"role": "system", "content": EXPAND_SYSTEM}, {"role": "user", "content": question}], model, json_mode=True, temperature=0.0, max_tokens=120)
        qs = [str(q) for q in json.loads(raw).get("queries", []) if str(q).strip()]
        return qs[:3]
    except Exception:  # noqa: BLE001
        return []


def retrieve(con, question: str, k: int = 8, model: str = vklib.DEFAULT_MODEL, expand_model: str | None = None, section_openings: int = 2) -> Context:
    """Hybrid retrieval; with expand_model set, 2-3 LLM-written keyword queries are fused in with the question itself."""
    queries = [question] + (expand_queries(question, expand_model) if expand_model else [])
    qvecs = vklib.embed(queries, model, "query")
    qvec = qvecs[0]
    if len(queries) == 1:
        hits = vklib.search_paragraphs(con, question, model, k=k, qvec=qvec)
    else:
        lists = [[h.id for h in vklib.search_paragraphs(con, q, model, k=k * 3, qvec=v)] for q, v in zip(queries, qvecs)]
        hits = vklib.rrf(lists)[:k]
    secs = [vklib.get_section(con, h.id) for h in vklib.search_sections(con, question, model, k=6, qvec=qvec)]
    # add the opening paragraph of the top section hits: whole-lecture questions are answered by lecture openings
    ids = [h.id for h in hits]
    for sec in secs[:section_openings]:
        row = con.execute("SELECT id FROM paragraphs WHERE section_id = ? AND length(text) >= 200 ORDER BY para_idx LIMIT 1", (sec["id"],)).fetchone()
        if row and row[0] not in ids:
            ids.append(row[0])
    passages = []
    for pid in ids:
        p = vklib.get_paragraph(con, pid)
        s = vklib.get_section(con, p["section_id"])
        sents = [p["text"][a:b] for a, b in p["sentences"]] or [p["text"]]
        passages.append(Passage(p, s, vklib.citation(s), sents))
    ctx = Context(question, passages, secs, hits)
    ctx.queries = queries
    return ctx


def format_block(ctx: Context, numbered_sentences: bool) -> str:
    out = []
    for p in ctx.passages:
        head = f"[{p.para['id']}] {p.citation} ({p.section['genre']})"
        if numbered_sentences:
            body = " ".join(f"({i}) {s}" for i, s in enumerate(p.sentences, 1))
        else:
            body = p.para["text"]
        out.append(head + "\n" + body)
    see = "\n".join(f"[{s['id']}] {vklib.citation(s)}" for s in ctx.sections)
    return "PASSAGES\n\n" + "\n\n".join(out) + "\n\nRELATED SECTIONS (for cite_more / see also)\n" + see


# ------------------------------------------------------------------ quote guard
def _norm(s: str) -> str:
    s = s.lower().replace("’", "'").replace("‘", "'").replace("“", '"').replace("”", '"').replace("—", "-").replace("–", "-")
    s = re.sub(r"[^a-z0-9' ]+", " ", s)
    return re.sub(r"\s+", " ", s).strip()


QUOTE_RE = re.compile(r"[“\"](\S[^”\"]{10,}?\S)[”\"]")  # span must start/end on a non-space: keeps the little "I" and the great "I" from pairing up


def verify_quote(con, span: str, ctx: Context | None) -> tuple[bool, float, str | None]:
    """Does `span` exist (near-)verbatim in the corpus? Checks retrieved passages first, then an FTS lookup."""
    target = _norm(span)
    if not target:
        return False, 0.0, None
    cands: list[tuple[str, str]] = []  # (normalized text, para id)
    paras = [p.para for p in ctx.passages] if ctx else []
    if len(target) >= 10:
        rows = con.execute("SELECT id, text FROM fts_para WHERE fts_para MATCH ? ORDER BY bm25(fts_para, 0.0, 0.0, 1.0) LIMIT 15", (vklib.fts_query(span),)).fetchall()
        paras += [{"id": r[0], "text": r[1]} for r in rows]
    best, best_pid = 0.0, None
    for p in paras:
        nt = _norm(p["text"])
        if target in nt:
            return True, 1.0, p["id"]
        # sliding windows of the paragraph roughly the span's length
        L = len(target)
        step = max(8, L // 4)
        for i in range(0, max(1, len(nt) - L + 1), step):
            r = difflib.SequenceMatcher(None, target, nt[i:i + L + 8]).ratio()
            if r > best:
                best, best_pid = r, p["id"]
                if best >= 0.995:
                    return True, best, best_pid
    return best >= GUARD_THRESHOLD, best, best_pid


def quote_guard(con, text: str, ctx: Context | None) -> tuple[str, list[dict]]:
    """Every span inside quotation marks must be verbatim corpus text; otherwise the quotation marks are removed (paraphrase)."""
    report = []

    def fix(m):
        span = m.group(1)
        ok, score, pid = verify_quote(con, span, ctx)
        report.append({"quote": span, "verified": ok, "score": round(score, 3), "para": pid})
        return m.group(0) if ok else span

    return QUOTE_RE.sub(fix, text), report


# ------------------------------------------------------------------ citations in prose
TAG_RE = re.compile(r"\[(p-\d{6})\]")


NUM_CITE_RE = re.compile(r"\[\d+\]")


def split_sentences(text: str) -> list[str]:
    """Abbreviation-aware sentence split (extract_corpus.sentence_spans) that also knows rendered citation markers: "...untrue.[1] They"
    ends a sentence at the marker. Each [n] is collapsed to one ")" for splitting and the spans are mapped back to the original text."""
    from extract_corpus import sentence_spans  # abbreviation-aware (Mr., Mrs., St., ...)
    mod, back = [], []
    i = 0
    for m in NUM_CITE_RE.finditer(text):
        mod.extend(text[i:m.start()]); back.extend(range(i, m.start()))
        mod.append(")"); back.append(m.start())
        i = m.end()
    mod.extend(text[i:]); back.extend(range(i, len(text)))
    back.append(len(text))
    return [text[back[a]:back[b]] for a, b in sentence_spans("".join(mod))]


NO_RECORD_RE = re.compile(r"no record|did not (speak|write|address)|cannot speak to", re.I)


def render_citations(text: str, ctx: Context, strict: bool = False) -> tuple[str, list[dict], dict]:
    """Replace [p-xxxxxx] tags with numbered citations; drop tags that point outside the retrieved set.
    strict=True (scholar): sentences without a surviving citation are dropped, except a 'no record' sentence."""
    cites: list[dict] = []
    index: dict[str, int] = {}
    bad = 0

    def sub(m):
        nonlocal bad
        pid = m.group(1)
        p = ctx.by_id(pid)
        if p is None:
            bad += 1
            return ""
        if pid not in index:
            index[pid] = len(cites) + 1
            cites.append({"n": index[pid], "para_id": pid, "section_id": p.section["id"], "citation": p.citation})
        return f"[{index[pid]}]"

    out = TAG_RE.sub(sub, text)
    out = re.sub(r"(\[\d+\])(\[\d+\])+", lambda m: "".join(sorted(set(re.findall(r"\[\d+\]", m.group(0))))), out)
    out = re.sub(r" +([.,;:])", r"\1", out)
    sentences = [s.strip() for s in split_sentences(out) if s.strip()]
    dropped = []
    if strict:
        kept = []
        for s in sentences:
            if re.search(r"\[\d+\]", s) or NO_RECORD_RE.search(s):
                kept.append(s)
            else:
                dropped.append(s)
        sentences = kept
        out = " ".join(sentences)
    stats = {"sentences": len(sentences), "sentences_cited": sum(1 for s in sentences if re.search(r"\[\d+\]", s)), "invalid_tags": bad,
             "dropped_uncited": len(dropped), "dropped_text": dropped}
    return out, cites, stats


# ------------------------------------------------------------------ modes
def parse_json(raw: str):
    """Lenient JSON parse of a model reply (strips ``` fences); None when unparseable."""
    try:
        j = json.loads(re.sub(r"^```(?:json)?|```$", "", raw.strip(), flags=re.M))
        return j if isinstance(j, dict) else None
    except (json.JSONDecodeError, AttributeError):
        return None


def scholar_from_plan(con, ctx: Context, plan: dict | None, raw: str = "") -> dict:
    """Render a scholar JSON plan ({"sentences": [{"text", "ids"}], "no_record"}): tagged prose -> quote guard -> strict citations."""
    plan = plan or {}
    items = plan.get("sentences") or []
    # assemble tagged prose; sentences whose ids all fall outside the retrieved set are dropped by render_citations(strict)
    tagged = []
    for it in items:
        if not isinstance(it, dict) or not str(it.get("text", "")).strip():
            continue
        ids = [str(x) for x in (it.get("ids") or []) if TAG_RE.fullmatch(f"[{x}]")]
        tagged.append(str(it["text"]).strip() + "".join(f"[{x}]" for x in ids))
    guarded, qreport = quote_guard(con, " ".join(tagged), ctx)
    text, cites, stats = render_citations(guarded, ctx, strict=True)
    stats["model_sentences"] = len(items)
    if not text.strip() or plan.get("no_record"):
        text = "I have no record of speaking to this."
    return {"mode": "scholar", "text": text.strip(), "citations": cites, "quote_guard": qreport, "stats": stats, "raw": raw}


def run_scholar(con, ctx: Context, model: str = DEV_MODEL) -> dict:
    msgs = [{"role": "system", "content": SCHOLAR_SYSTEM},
            {"role": "user", "content": format_block(ctx, False) + f"\n\nQUESTION: {ctx.question}"}]
    raw = chat(msgs, model, json_mode=True, max_tokens=1200)
    return scholar_from_plan(con, ctx, parse_json(raw), raw)


def persona_from_text(con, ctx: Context, raw: str) -> dict:
    """Quote-guard persona prose and turn [p-id] tags into numbered citations."""
    guarded, qreport = quote_guard(con, raw, ctx)
    text, cites, stats = render_citations(guarded, ctx)
    return {"mode": "persona", "text": text.strip(), "citations": cites, "quote_guard": qreport, "stats": stats, "raw": raw}


def run_persona(con, ctx: Context, model: str = DEV_MODEL) -> dict:
    msgs = [{"role": "system", "content": PERSONA_SYSTEM},
            {"role": "user", "content": format_block(ctx, False) + f"\n\nQUESTION: {ctx.question}"}]
    raw = chat(msgs, model, temperature=0.6)
    return persona_from_text(con, ctx, raw)


def oracle_from_plan(con, ctx: Context, plan: dict | None, raw: str = "") -> dict:
    if plan is None:
        return {"mode": "oracle", "fallthrough": True, "error": "unparseable plan", "raw": raw}
    if plan.get("fallthrough") or not plan.get("quotes"):
        return {"mode": "oracle", "fallthrough": True, "raw": raw}
    return render_oracle(con, ctx, plan, raw)


def run_oracle(con, ctx: Context, model: str = DEV_MODEL) -> dict:
    msgs = [{"role": "system", "content": ORACLE_SYSTEM},
            {"role": "user", "content": format_block(ctx, True) + f"\n\nQUESTION: {ctx.question}"}]
    raw = chat(msgs, model, json_mode=True, temperature=0.2, max_tokens=500)
    return oracle_from_plan(con, ctx, parse_json(raw), raw)


def render_oracle(con, ctx: Context, plan: dict, raw: str = "") -> dict:
    parts, cites, quotes = [], [], []
    words = 0
    overflow_secs: list[str] = []
    bridges = [b for b in plan.get("bridges", []) if isinstance(b, str)]
    for i, q in enumerate(plan.get("quotes", [])):
        p = ctx.by_id(str(q.get("para", "")))
        if p is None:
            continue
        n = len(p.sentences)
        a = max(1, int(q.get("from", 1)))
        b = min(n, int(q.get("to", a)))
        if a > b:
            continue
        span = re.sub(r"^[QA]\.\s*[—–-]*\s*", "", " ".join(p.sentences[a - 1:b]))  # Q&A sections carry "A. — " markers in the text
        w = len(span.split())
        if words and words + w > QUOTE_WORD_CAP:
            overflow_secs.append(p.section["id"])
            continue
        if w > QUOTE_WORD_CAP:  # trim a single over-long range sentence by sentence
            while b > a and len(" ".join(p.sentences[a - 1:b]).split()) > QUOTE_WORD_CAP:
                b -= 1
            span = re.sub(r"^[QA]\.\s*[—–-]*\s*", "", " ".join(p.sentences[a - 1:b])); w = len(span.split())
        words += w
        span = span.strip()
        if len(span) > 2 and span[0] in "\"“" and span[-1] in "\"”" and span.count('"') + span.count("“") + span.count("”") == 2:
            span = span[1:-1].strip()  # the corpus paragraph itself was a quotation (interviews, reports)
        bridge = bridges[i].strip() if i < len(bridges) else ""
        bridge = bridge.replace('"', "").replace("“", "").replace("”", "")
        bridge = split_sentences(bridge)[0].strip() if bridge else ""
        cites.append({"n": len(cites) + 1, "para_id": p.para["id"], "section_id": p.section["id"], "citation": p.citation})
        quotes.append({"para_id": p.para["id"], "from": a, "to": b, "text": span, "words": w})
        parts.append((bridge + "\n" if bridge else "") + f"“{span}” [{len(cites)}]")
    if not quotes:
        return {"mode": "oracle", "fallthrough": True, "error": "plan referenced no valid passages", "raw": raw}
    see_ids = [s for s in plan.get("cite_more", []) if isinstance(s, str)] + overflow_secs
    see_also = []
    seen = {c["section_id"] for c in cites}
    for sid in see_ids:
        if sid in seen:
            continue
        row = con.execute("SELECT id FROM sections WHERE id = ?", (sid,)).fetchone()
        if row:
            see_also.append({"section_id": sid, "citation": vklib.citation(vklib.get_section(con, sid))}); seen.add(sid)
    text = "\n\n".join(parts)
    if see_also:
        text += "\n\nI say more on this in " + "; ".join(s["citation"] for s in see_also) + "."
    _, qreport = quote_guard(con, text, ctx)  # by construction every quote verifies; recorded for the eval
    return {"mode": "oracle", "text": text, "citations": cites, "see_also": see_also, "quotes": quotes, "quoted_words": words,
            "quote_guard": qreport, "raw": raw}


def route(con, ctx: Context, model: str = DEV_MODEL) -> dict:
    top = ctx.hits[0].score if ctx.hits else 0
    evidence = f"retrieval evidence: top fused score {top:.4f}; top passages: " + "; ".join(
        f"{p.citation} ({p.section['genre']}): {p.para['text'][:160]!r}" for p in ctx.passages[:3])
    raw = chat([{"role": "system", "content": ROUTER_SYSTEM}, {"role": "user", "content": f"QUESTION: {ctx.question}\n\n{evidence}"}],
               model, json_mode=True, temperature=0.0, max_tokens=120)
    try:
        j = json.loads(raw)
        mode = str(j.get("mode", "persona")).lower()
    except json.JSONDecodeError:
        j, mode = {"raw": raw}, "persona"
    if mode not in {"scholar", "oracle", "persona", "mixed"}:
        mode = "persona"
    return {"mode": mode, "reason": j.get("reason", "")}


def ask(con, question: str, mode: str = "auto", model: str = DEV_MODEL, k: int = 8, embed_model: str = vklib.DEFAULT_MODEL, expand: bool = True,
        on_mode=None) -> dict:
    """on_mode(mode, info) is called as soon as the mode is known (and again on an oracle->persona cascade)."""
    t0 = time.time()
    ctx = retrieve(con, question, k, embed_model, model if expand else None)
    routed = None
    if mode == "auto":
        routed = route(con, ctx, model)
        mode = routed["mode"]
    if on_mode:
        on_mode(mode, {"router": routed})
    result: dict
    if mode == "mixed":
        a = run_scholar(con, ctx, model)
        b = run_persona(con, ctx, model)
        result = {"mode": "mixed", "parts": [a, b], "text": "WHAT I SAID\n" + a["text"] + "\n\nWHAT I WOULD SAY\n" + b["text"], "citations": a["citations"] + b["citations"]}
    elif mode == "oracle":
        result = run_oracle(con, ctx, model)
        if result.get("fallthrough"):
            if on_mode:
                on_mode("persona", {"cascade": "oracle fell through to persona"})
            result = run_persona(con, ctx, model)
            result["cascade"] = "oracle fell through to persona"
    elif mode == "scholar":
        result = run_scholar(con, ctx, model)
    else:
        result = run_persona(con, ctx, model)
    result["question"] = question
    result["router"] = routed
    result["model"] = model
    result["queries"] = ctx.queries
    result["retrieved"] = [{"para_id": p.para["id"], "section_id": p.section["id"], "citation": p.citation} for p in ctx.passages]
    result["seconds"] = round(time.time() - t0, 1)
    return result
