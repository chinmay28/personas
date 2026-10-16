"""Phase 1: extract the Complete Works epub into corpus/sections.jsonl and corpus/paragraphs.jsonl.

Run:  .venv/bin/python scripts/extract_corpus.py [--epub PATH] [--out corpus/]
Writes sections.jsonl, paragraphs.jsonl, extraction_report.md, extraction_report.json,
genre_spotcheck.tsv (50 random sections for manual genre review).
"""
from __future__ import annotations

import argparse
import json
import os
import random
import re
import sys
from collections import Counter, defaultdict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from epub_split import split_epub, TocEntry  # noqa: E402

# The epub is not distributed with the repo (corpus/ holds the extracted public-domain text). Set VK_EPUB to re-extract.
DEFAULT_EPUB = os.path.expanduser(os.environ.get(
    "VK_EPUB", "~/Documents/Books/Complete Works of Swami Vivekananda - Swami Vivekananda/"
    "Complete Works of Swami Vivekananda - Swami Vivekananda.epub"))

# ---------------------------------------------------------------- genre rules
# genre values: lecture | book_chapter | qa | dialogue | letter | interview | poem | notes |
#               translation | front_matter | essay | newspaper_report | sayings | other
WORK_GENRE = [
    (r"^EPISTLES|^LETTERS", "letter"),
    (r"^QUESTIONS AND ANSWERS", "qa"),
    (r"^CONVERSATIONS AND DIALOGUES", "dialogue"),
    (r"^CONVERSATIONS AND INTERVIEWS", "dialogue"),
    (r"^INTERVIEWS", "interview"),
    (r"^INSPIRED TALKS", "notes"),
    (r"^NOTES (OF|FROM) ", "notes"),
    (r"^SAYINGS AND UTTERANCES", "sayings"),
    (r"NEWSPAPER", "newspaper_report"),
    (r"^EXCERPTS FROM SISTER NIVEDITA", "other"),
    (r"POEMS$", "poem"),
    (r"^WRITINGS: PROSE AND POEMS", "essay_or_poem"),
    (r"^WRITINGS: PROSE$", "essay"),
    (r"^TRANSLATION", "translation"),
    (r"^(KARMA|RAJA|JNANA|BHAKTI)-YOGA$|^PARA-BHAKTI", "book_chapter"),
    (r"^ADDRESSES|^LECTURES|^WORK AND ITS SECRET|^THE POWERS OF THE MIND|^HINTS ON PRACTICAL|"
     r"^BHAKTI OR DEVOTION|^PRACTICAL VEDANTA|^BUDDHISTIC INDIA", "lecture"),
    (r"^UNPUBLISHED", "other"),
]
MONTHS = "january|february|march|april|may|june|july|august|september|october|november|december"
MON_ABBR = {m[:3]: i + 1 for i, m in enumerate(MONTHS.split("|"))}
DATE_RES = [
    # 5th January, 1890 / 5 Jan. 1890 / 5th of January 1890
    re.compile(r"\b(\d{1,2})(?:st|nd|rd|th)?\.?\s+(?:of\s+)?(" + MONTHS + r"|jan|feb|mar|apr|jun|jul|aug|sept?|oct|nov|dec)\.?,?\s+(\d{4})\b", re.I),
    # January 5, 1890 / Monday, July 3, 1895
    re.compile(r"\b(" + MONTHS + r"|jan|feb|mar|apr|jun|jul|aug|sept?|oct|nov|dec)\.?\s+(\d{1,2})(?:st|nd|rd|th)?,?\s+(\d{4})\b", re.I),
    # January 1890 / Jan., 1890
    re.compile(r"\b(" + MONTHS + r"|jan|feb|mar|apr|jun|jul|aug|sept?|oct|nov|dec)\.?,?\s+(\d{4})\b", re.I),
    # 4th June '95 / June '95
    re.compile(r"\b(\d{1,2})(?:st|nd|rd|th)?\.?\s+(" + MONTHS + r"|jan|feb|mar|apr|jun|jul|aug|sept?|oct|nov|dec)\.?,?\s+[’']\s?(\d{2})\b", re.I),
    re.compile(r"\b(" + MONTHS + r"|jan|feb|mar|apr|jun|jul|aug|sept?|oct|nov|dec)\.?,?\s+[’']\s?(\d{2})\b", re.I),
    # bare year
    re.compile(r"\b(18[5-9]\d|190[0-2])\b"),
]
NOTE_REF_RE = re.compile(r"\[(\d{1,3})\]")
SALUTATION_RE = re.compile(r"^(?:MY\s+)?(?:DEAR|DEAREST|BLESSED|BELOVED|REVERED|RESPECTED|HONOURED|DEAR\s+SIR|MADAM|SIR)\b[^\n]*", re.I)
ABBREV = {"mr", "mrs", "dr", "st", "prof", "rev", "no", "vol", "ch", "chap", "sri", "shri", "swami", "mt", "viz", "etc", "e.g", "i.e", "cf", "pp", "p", "ms", "messrs", "jr", "sr", "capt", "col", "gen", "lt", "hon", "esq", "rs", "re", "bart", "rt"}


def classify_work(work: str) -> str:
    w = work.strip().upper()
    for pat, g in WORK_GENRE:
        if re.search(pat, w):
            return g
    return "unmapped"


def poem_like(paras: list[str]) -> bool:
    if not paras:
        return False
    with_breaks = sum(1 for p in paras if "\n" in p)
    short_lines = sum(1 for p in paras for ln in p.split("\n") if len(ln) < 70)
    total_lines = sum(len(p.split("\n")) for p in paras)
    return with_breaks / len(paras) >= 0.5 and short_lines / max(total_lines, 1) >= 0.6


def iso_date(m, kind) -> str:
    g = m.groups()
    def mon(s):
        return MON_ABBR[s[:3].lower()]
    if kind == 0:
        return f"{int(g[2]):04d}-{mon(g[1]):02d}-{int(g[0]):02d}"
    if kind == 1:
        return f"{int(g[2]):04d}-{mon(g[0]):02d}-{int(g[1]):02d}"
    if kind == 2:
        return f"{int(g[1]):04d}-{mon(g[0]):02d}"
    if kind == 3:
        return f"18{int(g[2]):02d}-{mon(g[1]):02d}-{int(g[0]):02d}"
    if kind == 4:
        return f"18{int(g[1]):02d}-{mon(g[0]):02d}"
    return f"{int(g[0]):04d}"


def find_date(lines: list[str]) -> tuple[str | None, int | None]:
    """Return (iso date, index of line) using the most specific pattern that matches any header line."""
    for kind, rx in enumerate(DATE_RES):
        for i, ln in enumerate(lines):
            m = rx.search(ln)
            if m:
                return iso_date(m, kind), i
    return None, None


def letter_meta(title: str, paras: list[str]) -> dict:
    header = []
    for p in paras[:6]:
        header.extend(p.split("\n"))
    header = [h.strip() for h in header if h.strip()]
    meta = {"recipient": re.sub(r"^\d+\s+", "", title).strip() or None, "recipient_full": None, "date": None, "place": None, "salutation": None, "translated_from": None}
    sal_idx = None
    for i, ln in enumerate(header):
        if SALUTATION_RE.match(ln):
            meta["salutation"] = re.sub(r"\s*\[?\d+\]?\s*$", "", ln).strip()
            pm = re.search(r"\(([^)]+)\)", ln)
            if pm:
                meta["recipient_full"] = pm.group(1).strip().rstrip(".")
            sal_idx = i
            break
    region = header[:sal_idx] if sal_idx is not None else header[:8]
    for ln in region:
        m = re.search(r"\(Translated from ([A-Za-z]+)\)", ln)
        if m:
            meta["translated_from"] = m.group(1)
    date, di = find_date(region)
    meta["date"] = date
    if di is not None:
        place_lines = [ln for ln in region[:di] if not ln.startswith("(") and not re.match(r"^\d+\s", ln)
                       and len(ln) < 80 and ln.upper() != title.upper()]
        # the date line itself may hold a trailing/leading place, e.g. "ALLAHABAD, 5th January 1890"
        dl = region[di]
        head = DATE_RES[0].split(dl)[0] if DATE_RES[0].search(dl) else ""
        head = re.sub(r"[,\s]+$", "", head).strip()
        if head and len(head) < 60 and not re.search(r"\d", head):
            place_lines.append(head)
        if place_lines:
            cleaned = [re.sub(r"^[\[\(]+|[\]\)]+$", "", ln).strip(" ,.;") for ln in place_lines]
            cleaned = [c for c in cleaned if c]
            meta["place"] = ", ".join(cleaned) or None
    return meta


def generic_date(paras: list[str]) -> str | None:
    lines = []
    for p in paras[:3]:
        lines.extend(p.split("\n"))
    date, _ = find_date([ln for ln in lines if len(ln) < 160])
    return date


def sentence_spans(text: str) -> list[list[int]]:
    spans = []
    start = 0
    n = len(text)
    i = 0
    while i < n:
        c = text[i]
        if c in ".!?":
            j = i + 1
            while j < n and text[j] in "”\"'’)]":
                j += 1
            if j >= n or (text[j].isspace() and (j + 1 < n and (text[j + 1].isupper() or text[j + 1] in "“\"‘([" or text[j + 1].isdigit()))):
                # abbreviation guard
                word = re.search(r"([A-Za-z\.]+)$", text[start:i])
                tok = (word.group(1) if word else "").lower().strip(".")
                if c == "." and (tok in ABBREV or len(tok) == 1 or re.fullmatch(r"[a-z](\.[a-z])*", tok)):
                    i = j
                    continue
                end = j
                while end > start and text[end - 1].isspace():
                    end -= 1
                if end > start:
                    spans.append([start, end])
                start = j
                while start < n and text[start].isspace():
                    start += 1
                i = start
                continue
        elif c == "\n":
            end = i
            while end > start and text[end - 1].isspace():
                end -= 1
            if end > start:
                spans.append([start, end])
            start = i + 1
        i += 1
    if start < n:
        tail = text[start:].rstrip()
        if tail:
            spans.append([start, start + len(tail)])
    return spans


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epub", default=DEFAULT_EPUB)
    ap.add_argument("--out", default=os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "corpus"))
    ap.add_argument("--expected-chars", type=float, default=8.3e6)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    entries, dropped = split_epub(args.epub)
    # volume by document order
    vol_markers = sorted([(e.doc_seq, int(m.group(1))) for e in entries
                          if (m := re.match(r"^VOLUME (\d+)$", e.label)) and e.doc_seq is not None])

    def volume_of(e: TocEntry):
        v = None
        for seq, vol in vol_markers:
            if e.doc_seq is not None and seq <= e.doc_seq:
                v = vol
        return v

    sections, paragraphs = [], []
    report = defaultdict(list)
    genre_counts, kind_counts = Counter(), Counter()
    pid = 0
    letters_total = letters_dated = letters_recip = letters_place = 0
    endnotes: list[dict] = []
    for e in entries:
        # editorial endnotes block ("FOOTNOTES" heading followed by [1] ... [n]) lives at the very end of the book
        for k, ptxt in enumerate(e.paragraphs):
            if ptxt.strip().upper() == "FOOTNOTES" and k + 1 < len(e.paragraphs) and e.paragraphs[k + 1].startswith("[1]"):
                for q in e.paragraphs[k + 1:]:
                    m = NOTE_REF_RE.match(q)
                    if m:
                        endnotes.append({"id": f"n-{int(m.group(1)):04d}", "number": int(m.group(1)),
                                         "text": NOTE_REF_RE.sub("", q, count=1).strip(), "split_from": f"s-{e.order:04d}"})
                    elif endnotes:  # continuation paragraph of the previous note
                        endnotes[-1]["text"] = (endnotes[-1]["text"] + "\n\n" + q).strip()
                e.paragraphs = e.paragraphs[:k]
                report["endnotes_split_from"].append(f"{e.order}\t{' / '.join(e.path)}\t{len(endnotes)} notes")
                break
        # inline footnote markers [n] -> note_refs, removed from text
        note_refs_per_para: list[list[int]] = []
        cleaned_paras = []
        for ptxt in e.paragraphs:
            refs = [int(x) for x in NOTE_REF_RE.findall(ptxt)]
            if refs:
                ptxt = NOTE_REF_RE.sub("", ptxt)
                ptxt = re.sub(r" +([,.;:!?])", r"\1", ptxt)
                ptxt = re.sub(r"[ ]{2,}", " ", ptxt).strip()
            note_refs_per_para.append(refs)
            cleaned_paras.append(ptxt)
        e.paragraphs = cleaned_paras
        work = e.path[1] if len(e.path) >= 2 else e.path[0]
        top = e.path[0]
        is_volume_child = bool(re.match(r"^VOLUME \d+$", top))
        if not is_volume_child and top != "UNPUBLISHED":
            genre = "front_matter"
        elif len(e.path) == 1:
            genre = "front_matter" if is_volume_child else classify_work(work)
        else:
            genre = classify_work(work)
            if genre == "essay_or_poem":
                genre = "poem" if poem_like(e.paragraphs) else "essay"
                report["essay_or_poem_decided"].append(f"{e.order}\t{genre}\t{' / '.join(e.path)}")
            if genre == "letter" and e.label.strip().upper() in {"NOTE", "PREFACE", "INTRODUCTION"}:
                genre = "front_matter"
            if genre == "dialogue" and e.depth >= 2 and ("INTERVIEW" in e.label.upper() or "interview" in " ".join(e.paragraphs[:2])[:400].lower()):
                genre = "interview"
        if genre == "unmapped":
            report["unmapped_works"].append(" / ".join(e.path))
            genre = "other"
        text = "\n\n".join(e.paragraphs)
        kind = "section" if len(text) >= 60 else ("stub" if text else "container")
        kind_counts[kind] += 1
        if kind == "section":
            genre_counts[genre] += 1
        meta = {"date": None, "recipient": None, "place": None}
        if genre == "letter" and kind == "section":
            lm = letter_meta(e.label, e.paragraphs)
            meta.update({k: lm[k] for k in ("date", "recipient", "place", "recipient_full")})
            meta["salutation"] = lm["salutation"]
            meta["translated_from"] = lm["translated_from"]
            letters_total += 1
            letters_dated += bool(lm["date"]); letters_recip += bool(lm["recipient"]); letters_place += bool(lm["place"])
            if not lm["date"]:
                report["letters_without_date"].append(f"{e.order}\t{' / '.join(e.path)}\t{(e.paragraphs[0] if e.paragraphs else '')[:100]!r}")
        elif kind == "section":
            meta["date"] = generic_date(e.paragraphs)
        sid = f"s-{e.order:04d}"
        sec = {
            "id": sid, "volume": volume_of(e), "work": work, "title": e.label, "toc_path": e.path,
            "genre": genre, "kind": kind, "date": meta.get("date"), "recipient": meta.get("recipient"),
            "place": meta.get("place"), "order": e.order, "doc_seq": e.doc_seq, "depth": e.depth,
            "text": text, "char_count": len(text), "para_count": len(e.paragraphs),
            "images_dropped": e.images, "source_file": e.first_file,
        }
        if genre == "letter" and kind == "section":
            sec["salutation"] = meta.get("salutation"); sec["translated_from"] = meta.get("translated_from"); sec["recipient_full"] = meta.get("recipient_full")
        sections.append(sec)
        for k, p in enumerate(e.paragraphs):
            pid += 1
            paragraphs.append({"id": f"p-{pid:06d}", "section_id": sid, "para_idx": k, "text": p,
                               "sentences": sentence_spans(p), "note_refs": note_refs_per_para[k]})
        if kind == "section" and not e.heading_blocks and e.depth == 2:
            report["sections_without_heading_block"].append(f"{e.order}\t{' / '.join(e.path)}\tfirst={e.paragraphs[0][:80]!r}")
        if len(text) > 120_000:
            report["very_long_sections"].append(f"{e.order}\t{len(text)}\t{' / '.join(e.path)}")
        if kind == "container" and e.depth == 2:
            report["empty_leaf_sections"].append(f"{e.order}\t{' / '.join(e.path)}")

    total_chars = sum(s["char_count"] for s in sections)
    para_texts = Counter(p["text"] for p in paragraphs if len(p["text"]) > 200)
    dup_paras = sum(c - 1 for c in para_texts.values() if c > 1)

    with open(os.path.join(args.out, "sections.jsonl"), "w") as f:
        for s in sections:
            f.write(json.dumps(s, ensure_ascii=False) + "\n")
    with open(os.path.join(args.out, "paragraphs.jsonl"), "w") as f:
        for p in paragraphs:
            f.write(json.dumps(p, ensure_ascii=False) + "\n")
    with open(os.path.join(args.out, "endnotes.jsonl"), "w") as f:
        for n in endnotes:
            f.write(json.dumps(n, ensure_ascii=False) + "\n")

    rng = random.Random(42)
    sample = rng.sample([s for s in sections if s["kind"] == "section"], 50)
    with open(os.path.join(args.out, "genre_spotcheck.tsv"), "w") as f:
        f.write("id\tgenre\tvolume\twork\ttitle\tdate\tfirst_120_chars\n")
        for s in sorted(sample, key=lambda x: x["order"]):
            f.write(f"{s['id']}\t{s['genre']}\t{s['volume']}\t{s['work']}\t{s['title']}\t{s['date']}\t{s['text'][:120].replace(chr(10), ' ')!r}\n")

    summary = {
        "toc_entries": len(entries), "sections_with_text": kind_counts["section"], "containers": kind_counts["container"],
        "paragraphs": len(paragraphs), "sentences": sum(len(p["sentences"]) for p in paragraphs),
        "total_chars": total_chars, "expected_chars": args.expected_chars, "chars_within_5pct": abs(total_chars - args.expected_chars) / args.expected_chars <= 0.05,
        "dropped": dropped, "images_dropped": sum(e.images for e in entries),
        "genre_counts": dict(genre_counts.most_common()),
        "chars_by_genre": {g: sum(s["char_count"] for s in sections if s["genre"] == g) for g in genre_counts},
        "sections_by_volume": dict(sorted(Counter(str(s["volume"]) for s in sections if s["kind"] == "section").items())),
        "letters": {"total": letters_total, "with_date": letters_dated, "with_recipient": letters_recip, "with_place": letters_place,
                    "date_rate": round(letters_dated / max(letters_total, 1), 3)},
        "duplicate_paragraphs_gt200chars": dup_paras,
        "endnotes": len(endnotes), "paragraphs_with_note_refs": sum(1 for p in paragraphs if p["note_refs"]),
        "stubs": kind_counts["stub"],
        "issues": {k: len(v) for k, v in report.items()},
    }
    with open(os.path.join(args.out, "extraction_report.json"), "w") as f:
        json.dump({"summary": summary, "issues": report}, f, indent=1, ensure_ascii=False)
    with open(os.path.join(args.out, "extraction_report.md"), "w") as f:
        f.write("# Extraction report\n\n```json\n" + json.dumps(summary, indent=1, ensure_ascii=False) + "\n```\n")
        for k, v in report.items():
            f.write(f"\n## {k} ({len(v)})\n\n")
            for line in v[:300]:
                f.write(f"- {line}\n")
            if len(v) > 300:
                f.write(f"- ... {len(v) - 300} more\n")
    print(json.dumps(summary, indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main()
