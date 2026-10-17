"""Flag editorial paratext in the corpus so the oracle never quotes it as Vivekananda's words.

  .venv/bin/python scripts/flag_editorial.py            # writes `editorial` into corpus/paragraphs.jsonl and index/corpus.db, lists hits
  .venv/bin/python scripts/flag_editorial.py --dry-run  # list only

Paragraph ids never change: this is a post-pass over the extracted corpus (2026-10-17, after the oracle quoted the translator's
preface to the Conversations and Dialogues as his own words on "hi swamiji"). Rules, deliberately narrow (false positives would
hide his text; false negatives are what we have today):
  1. the whole paragraph is one parenthesised note and it contains editorial vocabulary (published, translated, reprinted, Vol. V, ...);
  2. the paragraph opens as a report/translation/address-of-welcome notice ("The following is a report of a lecture ...");
  3. the paragraph names him in the third person as "Swami Vivekananda" / "the Swami" inside a parenthesised note.
Review file: corpus/editorial_paragraphs.tsv. Retrieval (vklib.search_paragraphs, modes.retrieve section openings) skips flagged ids.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import vklib  # noqa: E402

EDIT_WORDS = re.compile(
    r"\b(translat(ed|ion)|the disciple|editor|editorial|Complete Works|published|reprinted|reproduced|Bengali (book|article|magazine|paper)|"
    r"this series|these (conversations|dialogues|notes|letters)|appeared in|Vol\. [IVX]+|the above (address|lecture|is)|"
    r"report(ed|s)? (of|in) (this|the) (lecture|address|speech)|from the notes|notes (taken|of)|delivered (before|at|on)|"
    r"was presented to|Memorial Volume|Centenary|unpublished|manuscript|the Swami|Swami Vivekananda)\b", re.I)
OPENERS = re.compile(r"^(The following (is|are) (a |an |the )?(report|translation|summary|address|account|text|notes?|extracts?|letter|version)|"
                     r"This (lecture|address|speech|conversation|article|essay|letter|paper) (was|is|appeared|has been)|"
                     r"These (conversations|dialogues|notes|letters|extracts) (were|are|have))\b", re.I)


def is_editorial(text: str) -> str | None:
    t = text.strip()
    if len(t) < 40:
        return None
    paren = t.startswith("(") and t.endswith(")") and t.count("(") == t.count(")")
    if paren and EDIT_WORDS.search(t):
        return "parenthesised note"
    if OPENERS.match(t) and re.search(r"\b(report|editor|editorial|the Swami|Swami Vivekananda|Swamiji|presented to|published|reprinted)\b", t, re.I):
        return "notice opener"
    if paren and re.search(r"\b(the Swami|Swami Vivekananda|Swamiji)\b", t) and not re.search(r"\bI\b", t):
        return "third-person note"
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    path = os.path.join(vklib.CORPUS_DIR, "paragraphs.jsonl")
    paras = [json.loads(l) for l in open(path)]
    secs = {s["id"]: s for s in (json.loads(l) for l in open(os.path.join(vklib.CORPUS_DIR, "sections.jsonl")))}
    flagged = []
    for p in paras:
        why = is_editorial(p["text"])
        p["editorial"] = bool(why)
        if why:
            s = secs.get(p["section_id"], {})
            flagged.append((p["id"], why, s.get("genre", ""), s.get("title", ""), p["text"][:140].replace("\n", " ")))
    print(f"{len(flagged)} of {len(paras)} paragraphs flagged editorial")
    for row in flagged:
        print("  ", "\t".join(row[:4]), "|", row[4][:100])
    if args.dry_run:
        return
    with open(path, "w") as f:
        for p in paras:
            f.write(json.dumps(p, ensure_ascii=False) + "\n")
    with open(os.path.join(vklib.CORPUS_DIR, "editorial_paragraphs.tsv"), "w") as f:
        f.write("id\trule\tgenre\ttitle\ttext\n")
        for row in flagged:
            f.write("\t".join(row) + "\n")
    con = vklib.connect()
    cols = {r[1] for r in con.execute("PRAGMA table_info(paragraphs)")}
    if "editorial" not in cols:
        con.execute("ALTER TABLE paragraphs ADD COLUMN editorial INTEGER NOT NULL DEFAULT 0")
    con.execute("UPDATE paragraphs SET editorial = 0")
    con.executemany("UPDATE paragraphs SET editorial = 1 WHERE id = ?", [(r[0],) for r in flagged])
    con.commit()
    n = con.execute("SELECT count(*) FROM paragraphs WHERE editorial = 1").fetchone()[0]
    print(f"index/corpus.db: {n} paragraphs marked editorial; corpus/paragraphs.jsonl rewritten with the `editorial` field (ids unchanged)")


if __name__ == "__main__":
    main()
