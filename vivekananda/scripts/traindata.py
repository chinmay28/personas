"""Shared pieces for the phase 6 training dataset (PLAN.md section 6).

Record format (6.3): OpenAI-style messages; system is fixed (UNIFIED_SYSTEM); user = retrieval block + question; assistant =
"<mode>X</mode>\n" + mode output (JSON plan for oracle, JSON sentences for scholar, prose for persona).
The retrieval block is modes.format_block(ctx, numbered_sentences=True) for every mode, because the single model must choose the mode
after seeing the block, so the block cannot depend on the mode. Retrieval uses no LLM query expansion (deterministic, reproducible).
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import modes  # noqa: E402
import vklib  # noqa: E402

TIER_DIR = os.path.join(vklib.ROOT, "data", "tiers")
LOG_PATH = os.path.join(TIER_DIR, "build.log")
GEN_MODEL = os.environ.get("VK_GEN_MODEL", "qwen3:14b")          # user-turn synthesis, extraction, rewriting
PERSONA_MODEL = os.environ.get("VK_PERSONA_MODEL", "qwen3:14b")  # tier 6 answers (27b is 5x slower; set VK_PERSONA_MODEL=qwen3.8:27b)
WORKERS = int(os.environ.get("VK_WORKERS", "4"))
K = 8

UNIFIED_SYSTEM = modes.PERSONA_PREAMBLE + """
You receive numbered passages from your Complete Works (each sentence numbered) and a question. Output a mode tag on its own first line,
then the answer in that mode's format.
<mode>oracle</mode>  the passages answer the question in your own recorded words. Output a JSON selection plan only:
{"quotes":[{"para":"p-000161","from":1,"to":3}],"bridges":["one plain first-person sentence introducing the quote, no quotation marks"],"cite_more":["s-0022"]}
The system renders the chosen sentences verbatim; keep to one or two contiguous ranges of about 200 words in all.
<mode>scholar</mode>  the question is about what you said, wrote or did. Output JSON only:
{"sentences":[{"text":"one first-person sentence recounting it","ids":["p-000161"]}]}
Every sentence must be supported by the passages it cites; nothing is quoted unless copied exactly. If the passages do not cover it:
{"sentences":[],"no_record":true}
<mode>persona</mode>  personal advice or a situation you never addressed directly. Answer in first-person prose grounded in your recorded
principles, opening with a marker that you did not speak to this directly, citing a passage as [p-000161] after a sentence that rests on it,
with nothing inside quotation marks unless copied exactly from a passage."""

GENERIC_SYSTEM = "You are a helpful assistant."

_tls = threading.local()


def con():
    """One sqlite connection per thread."""
    c = getattr(_tls, "con", None)
    if c is None:
        c = _tls.con = vklib.connect()
    return c


def log(msg: str):
    os.makedirs(TIER_DIR, exist_ok=True)
    line = f"{time.strftime('%Y-%m-%d %H:%M:%S')} {msg}"
    print(line, flush=True)
    with open(LOG_PATH, "a") as f:
        f.write(line + "\n")


# ------------------------------------------------------------------ text helpers
def norm(s: str) -> str:
    return modes._norm(s)


def words(s: str) -> int:
    return len(s.split())


def stable_hash(s: str) -> str:
    return hashlib.sha1(s.encode("utf-8")).hexdigest()[:12]


STOP = set("the a an and or of to in on for with is are was were be been by as at that this these those it its he his him she her they "
           "them their we our you your i my me what which who whom whose how why when where does do did say says said about from into "
           "than then there here also not no yes if but so such very can could would should may might must shall will one ones all any "
           "some each every both either neither more most much many few less least own same other another over under again further".split())


def content_words(s: str) -> set[str]:
    return {w for w in re.findall(r"[a-z']+", s.lower()) if len(w) >= 3 and w not in STOP}


# ------------------------------------------------------------------ retrieval block
def build_context(question: str, gold_pids: list[str] | None = None, exclude_section: str | None = None, k: int = K) -> tuple[modes.Context, dict]:
    """Deterministic hybrid retrieval (no LLM expansion). Gold paragraphs that were not retrieved are injected in place of the lowest hits
    (flagged); passages from `exclude_section` are removed (persona tiers must not see their own answer)."""
    c = con()
    ctx = modes.retrieve(c, question, k, vklib.DEFAULT_MODEL, None)
    info = {"gold_injected": False, "excluded": 0}
    if exclude_section:
        before = len(ctx.passages)
        ctx.passages = [p for p in ctx.passages if p.section["id"] != exclude_section]
        ctx.sections = [s for s in ctx.sections if s["id"] != exclude_section]
        info["excluded"] = before - len(ctx.passages)
    for pid in gold_pids or []:
        if ctx.by_id(pid) is None:
            p = vklib.get_paragraph(c, pid)
            s = vklib.get_section(c, p["section_id"])
            sents = [p["text"][a:b] for a, b in p["sentences"]] or [p["text"]]
            passage = modes.Passage(p, s, vklib.citation(s), sents)
            # drop the last non-gold passage to keep the block size stable
            for i in range(len(ctx.passages) - 1, -1, -1):
                if ctx.passages[i].para["id"] not in (gold_pids or []):
                    del ctx.passages[i]
                    break
            ctx.passages.append(passage)
            info["gold_injected"] = True
    info["block_ids"] = [p.para["id"] for p in ctx.passages]
    info["section_ids"] = [s["id"] for s in ctx.sections]
    return ctx, info


def user_content(ctx: modes.Context, question: str) -> str:
    return modes.format_block(ctx, True) + f"\n\nQUESTION: {question}"


def make_record(rid: str, tier: int, mode: str, question: str, ctx: modes.Context, info: dict, assistant_body: str,
                source: dict, meta: dict | None = None, system: str = UNIFIED_SYSTEM) -> dict:
    assistant = f"<mode>{mode}</mode>\n{assistant_body}" if mode else assistant_body
    return {"id": rid, "tier": tier, "mode": mode, "source": source,
            "block_ids": info.get("block_ids", []), "section_ids": info.get("section_ids", []), "gold_injected": info.get("gold_injected", False),
            "messages": [{"role": "system", "content": system}, {"role": "user", "content": user_content(ctx, question)},
                         {"role": "assistant", "content": assistant}],
            "meta": meta or {}}


# ------------------------------------------------------------------ oracle plans from a known gold paragraph
def best_window(sentences: list[str], question: str, cap_words: int = 200) -> tuple[int, int]:
    """1-based inclusive sentence range: the whole paragraph if it fits in cap_words, else the window (<= cap) with the most question overlap."""
    n = len(sentences)
    total = sum(words(s) for s in sentences)
    if total <= cap_words:
        return 1, n
    q = content_words(question)
    best, best_score = (1, 1), -1.0
    for i in range(n):
        w = 0
        for j in range(i, n):
            w += words(sentences[j])
            if w > cap_words and j > i:
                break
            if w > cap_words:
                break
            span_words = content_words(" ".join(sentences[i:j + 1]))
            score = len(q & span_words) + 0.001 * w
            if score > best_score:
                best, best_score = (i + 1, j + 1), score
    return best


def _clean_title(sec: dict) -> str:
    return vklib.title_case(re.sub(r"^\d+\s+", "", sec["title"]))


def _work(sec: dict) -> str:
    return vklib.title_case(re.sub(r"\s*[–-]+\s*(FIRST|SECOND|THIRD|FOURTH|FIFTH) SERIES$", "", sec["work"]))


def bridge_for(sec: dict, pid: str) -> str:
    """A plain first-person bridge sentence built from the passage's metadata (no model call, no quotation marks)."""
    g = sec.get("genre")
    title, work = _clean_title(sec), _work(sec)
    if sec.get("depth", 2) < 2 or title == work:
        title = work
    yr = (sec.get("date") or "")[:4]
    h = int(re.sub(r"\D", "", pid) or 0) % 3
    if g == "lecture":
        opts = [f"I spoke to this in my lecture {title}" + (f", in {yr}" if yr else "") + ".",
                f"Here is what I said on this when I spoke on {title}.",
                f"I put it this way in my address {title}."]
    elif g == "book_chapter":
        opts = [f"I wrote of this in {work}, in the chapter {title}." if title != work else f"I wrote of this in {work}.",
                f"This is how I treated it in {work}.",
                f"In {work} I set it down thus."]
    elif g == "letter":
        to = vklib.title_case(sec.get("recipient_full") or sec.get("recipient") or sec["title"])
        place = vklib.title_case(sec["place"].split(",")[0].strip()) if sec.get("place") else ""
        opts = [f"I wrote this to {to}" + (f" from {place}" if place else "") + (f" in {yr}" if yr else "") + ".",
                f"This is what I wrote to {to}" + (f" in {yr}" if yr else "") + ".",
                f"I said as much in a letter to {to}."]
    elif g == "qa":
        opts = [f"This was put to me at {title}, and I answered thus.", f"I was asked this at {title}; here is my answer.",
                f"My answer, when this was asked at {title}, was this."]
    elif g == "dialogue":
        opts = [f"I said this in conversation, as {title} recorded it.", f"This is what I told {title} when we spoke of it.",
                f"I spoke of this in a conversation that {title} wrote down."]
    elif g == "notes":
        opts = [f"I said this in a discourse, {title}, as my hearers noted it down.", f"This is from the notes taken of my talk {title}.",
                f"My hearers recorded me saying this in {title}."]
    elif g == "interview":
        opts = [f"I said this to a correspondent, as printed under the heading {title}.", f"This is what I told an interviewer; it was printed as {title}.",
                f"I answered a reporter in these words."]
    elif g == "sayings":
        opts = ["This is one of my recorded sayings.", "I have said this, and it was written down.", "Here is a saying of mine that was recorded."]
    elif g == "newspaper_report":
        opts = [f"The newspapers reported me saying this under the heading {title}.", f"I said this, as the press report {title} has it.",
                f"This is what I was reported to have said in {title}."]
    else:
        opts = [f"I wrote this in {title}.", f"This is from my {title}.", f"I set this down in {title}."]
    return opts[h].replace('"', "").replace("“", "").replace("”", "")


def oracle_plan(ctx: modes.Context, gold_pid: str, question: str, cap_words: int = 200) -> tuple[dict, dict]:
    """Plan quoting the best window of the gold paragraph, plus a rendered form for audit. Returns (plan, rendered_result)."""
    p = ctx.by_id(gold_pid)
    a, b = best_window(p.sentences, question, cap_words)
    gold_sec = p.section
    cite_more = [s["id"] for s in ctx.sections if s["id"] != gold_sec["id"] and s.get("work") == gold_sec.get("work")][:2]
    plan = {"quotes": [{"para": gold_pid, "from": a, "to": b}], "bridges": [bridge_for(gold_sec, gold_pid)], "cite_more": cite_more}
    rendered = modes.render_oracle(con(), ctx, plan)
    return plan, rendered


def plan_json(plan: dict) -> str:
    return json.dumps(plan, ensure_ascii=False, separators=(",", ":"))


# ------------------------------------------------------------------ letters
SIG_RE = re.compile(r"^(yours|ever yours|your(s)? (ever|affectionate|loving|sincerely|truly)|with (all )?(love|blessings|best)|affectionately|"
                    r"sincerely|faithfully|vivekananda|v\.?$|\(?vivekananda\)?|yours in|in haste|love and blessings|blessings and love|"
                    r"with (my )?(kind|best)|yours etc|ever your|your friend|your brother|your son|your servant|your loving)", re.I)
HEADER_RE = re.compile(r"^(dear|my dear|dearest|blessed and beloved|beloved|sister|brother|mother|sir|madam|to |c/o|\d{1,2}(st|nd|rd|th)? |"
                       r"[A-Z][A-Z .,'-]{2,}$)", re.I)


def letter_body(paragraphs: list[dict]) -> tuple[list[dict], dict]:
    """Strip place/date header, salutation and signature; keep the body and any PS. Returns (kept paragraphs, info)."""
    texts = [p["text"].strip() for p in paragraphs]
    start = 0
    for i, t in enumerate(texts):
        is_header = (len(t) < 90 and (HEADER_RE.match(t) or re.fullmatch(r"[A-Z0-9 .,'’\-()/&]+", t) or re.search(r"\b1[89]\d\d\b", t)
                                      or t.startswith("(") or re.match(r"^\d", t) or re.fullmatch(r"[A-Z][a-z]+day[.,]?", t)))
        if is_header:
            start = i + 1
            continue
        break
    kept = []
    for p, t in zip(paragraphs[start:], texts[start:]):
        if len(t) < 90 and SIG_RE.match(t):
            continue
        if re.fullmatch(r"[A-Z .,'’\-()]+", t) and len(t) < 40:
            continue
        kept.append(p)
    return kept, {"dropped_header": start, "kept": len(kept), "total": len(paragraphs)}


# ------------------------------------------------------------------ LLM helpers
def llm_json(system: str, user: str, model: str = GEN_MODEL, max_tokens: int = 700, temperature: float = 0.4, tries: int = 3):
    for t in range(tries):
        try:
            raw = modes.chat([{"role": "system", "content": system}, {"role": "user", "content": user}], model, json_mode=True,
                             temperature=temperature, max_tokens=max_tokens)
            raw = re.sub(r"^```(?:json)?|```$", "", raw.strip(), flags=re.M)
            return json.loads(raw)
        except Exception as e:  # noqa: BLE001
            if t == tries - 1:
                log(f"llm_json failed: {e!r}")
                return None
            time.sleep(1.0)


def llm_text(system: str, user: str, model: str = GEN_MODEL, max_tokens: int = 900, temperature: float = 0.6) -> str:
    try:
        return modes.chat([{"role": "system", "content": system}, {"role": "user", "content": user}], model, temperature=temperature, max_tokens=max_tokens)
    except Exception as e:  # noqa: BLE001
        log(f"llm_text failed: {e!r}")
        return ""


# ------------------------------------------------------------------ resumable tier files
class TierWriter:
    def __init__(self, tier: int, name: str | None = None):
        os.makedirs(TIER_DIR, exist_ok=True)
        self.path = os.path.join(TIER_DIR, f"tier{tier}{('_' + name) if name else ''}.jsonl")
        self.done: set[str] = set()
        if os.path.exists(self.path):
            with open(self.path) as f:
                for line in f:
                    try:
                        self.done.add(json.loads(line)["id"])
                    except (json.JSONDecodeError, KeyError):
                        pass
        self.lock = threading.Lock()
        self.n_new = 0

    def write(self, rec: dict):
        with self.lock:
            if rec["id"] in self.done:
                return
            with open(self.path, "a") as f:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            self.done.add(rec["id"])
            self.n_new += 1


def pmap(fn, items, workers: int = WORKERS, label: str = ""):
    """Parallel map with progress logging, yielding results as they complete (callers write each record at once, so runs are resumable);
    exceptions are logged and yield None."""
    t0 = time.time()

    def safe(x):
        try:
            return fn(x)
        except Exception as e:  # noqa: BLE001
            log(f"{label} item failed: {e!r}")
            return None

    with ThreadPoolExecutor(max_workers=workers) as ex:
        for i, r in enumerate(ex.map(safe, items), 1):
            if i % 50 == 0 or i == len(items):
                log(f"{label}: {i}/{len(items)} ({time.time() - t0:.0f}s)")
            yield r


def load_jsonl(path: str) -> list[dict]:
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]
