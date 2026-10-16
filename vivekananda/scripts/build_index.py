"""Phase 2a: build index/corpus.db (metadata + FTS5 + sqlite-vec embeddings). Resumable per model.

  .venv/bin/python scripts/build_index.py                 # tables + FTS + nomic-embed-text vectors
  .venv/bin/python scripts/build_index.py --model bge-m3  # add another model's vectors (same DB)
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import vklib  # noqa: E402

SCHEMA = """
CREATE TABLE IF NOT EXISTS sections (
  id TEXT PRIMARY KEY, volume INTEGER, work TEXT, title TEXT, toc_path TEXT, genre TEXT, kind TEXT,
  date TEXT, recipient TEXT, recipient_full TEXT, place TEXT, "order" INTEGER, depth INTEGER,
  text TEXT, char_count INTEGER, para_count INTEGER);
CREATE TABLE IF NOT EXISTS paragraphs (
  id TEXT PRIMARY KEY, section_id TEXT, para_idx INTEGER, text TEXT, sentences TEXT, note_refs TEXT);
CREATE INDEX IF NOT EXISTS paragraphs_section ON paragraphs(section_id);
CREATE TABLE IF NOT EXISTS endnotes (id TEXT PRIMARY KEY, number INTEGER, text TEXT);
CREATE VIRTUAL TABLE IF NOT EXISTS fts_para USING fts5(id UNINDEXED, header, text, tokenize = "unicode61 remove_diacritics 2");
CREATE VIRTUAL TABLE IF NOT EXISTS fts_sec USING fts5(id UNINDEXED, title, text, tokenize = "unicode61 remove_diacritics 2");
CREATE TABLE IF NOT EXISTS meta (key TEXT PRIMARY KEY, value TEXT);
"""


def load_jsonl(path):
    with open(path) as f:
        for line in f:
            yield json.loads(line)


def header_of(s: dict) -> str:
    """Searchable header: volume, work, title, and for letters recipient/place/date. Indexed with every paragraph and section."""
    parts = [f"Volume {s['volume']}" if s.get("volume") else "", s["work"], s["title"]]
    if s.get("genre") == "letter":
        parts += [s.get("recipient_full") or "", s.get("place") or "", s.get("date") or ""]
    return " ".join(p for p in parts if p)


def build_fts(con, secs, paras):
    con.execute("DELETE FROM fts_sec"); con.execute("DELETE FROM fts_para")
    hdr = {s["id"]: header_of(s) for s in secs}
    con.executemany("INSERT INTO fts_sec VALUES (?,?,?)", [(s["id"], hdr[s["id"]], s["text"]) for s in secs if s["kind"] == "section"])
    con.executemany("INSERT INTO fts_para VALUES (?,?,?)", [(p["id"], hdr[p["section_id"]], p["text"]) for p in paras])
    con.commit()


def build_tables(con, rebuild_fts: bool = False):
    con.executescript(SCHEMA)
    if con.execute("SELECT COUNT(*) FROM sections").fetchone()[0]:
        if rebuild_fts:
            secs = list(load_jsonl(os.path.join(vklib.CORPUS_DIR, "sections.jsonl")))
            paras = list(load_jsonl(os.path.join(vklib.CORPUS_DIR, "paragraphs.jsonl")))
            build_fts(con, secs, paras)
            print("FTS tables rebuilt with headers")
        else:
            print("tables already populated; skipping text load")
        return
    secs = list(load_jsonl(os.path.join(vklib.CORPUS_DIR, "sections.jsonl")))
    con.executemany(
        'INSERT INTO sections VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)',
        [(s["id"], s["volume"], s["work"], s["title"], json.dumps(s["toc_path"]), s["genre"], s["kind"], s.get("date"),
          s.get("recipient"), s.get("recipient_full"), s.get("place"), s["order"], s["depth"], s["text"], s["char_count"], s["para_count"]) for s in secs])
    paras = list(load_jsonl(os.path.join(vklib.CORPUS_DIR, "paragraphs.jsonl")))
    con.executemany("INSERT INTO paragraphs VALUES (?,?,?,?,?,?)",
                    [(p["id"], p["section_id"], p["para_idx"], p["text"], json.dumps(p["sentences"]), json.dumps(p["note_refs"])) for p in paras])
    build_fts(con, secs, paras)
    con.executemany("INSERT INTO endnotes VALUES (?,?,?)",
                    [(n["id"], n["number"], n["text"]) for n in load_jsonl(os.path.join(vklib.CORPUS_DIR, "endnotes.jsonl"))])
    con.commit()
    print(f"loaded {len(secs)} sections, {len(paras)} paragraphs")


def build_vectors(con, model: str, batch: int, min_chars: int):
    dims = vklib.EMBED_MODELS[model][0]
    slug = vklib.model_slug(model)
    con.execute(f"CREATE VIRTUAL TABLE IF NOT EXISTS vec_para_{slug} USING vec0(id TEXT PRIMARY KEY, embedding float[{dims}])")
    done = {r[0] for r in con.execute(f"SELECT id FROM vec_para_{slug}")}
    rows = con.execute(
        "SELECT p.id, p.text, s.id, s.volume, s.work, s.title, s.genre, s.recipient_full, s.place, s.date FROM paragraphs p JOIN sections s ON s.id = p.section_id "
        "WHERE s.kind = 'section' AND length(p.text) >= ? ORDER BY p.id", (min_chars,)).fetchall()
    if vklib.with_header(model):
        todo = [(r[0], header_of({"volume": r[3], "work": r[4], "title": r[5], "genre": r[6], "recipient_full": r[7], "place": r[8], "date": r[9]}) + "\n\n" + r[1])
                for r in rows if r[0] not in done]
    else:
        todo = [(r[0], r[1]) for r in rows if r[0] not in done]
    print(f"[{model}] {len(done)} embedded, {len(todo)} to go")
    t0 = time.time()
    for i in range(0, len(todo), batch):
        chunk = todo[i:i + batch]
        embs = vklib.embed([t for _, t in chunk], model, "document")
        con.executemany(f"INSERT INTO vec_para_{slug}(id, embedding) VALUES (?, ?)",
                        [(pid, vklib.pack(e)) for (pid, _), e in zip(chunk, embs)])
        con.commit()
        if (i // batch) % 20 == 0:
            rate = (i + len(chunk)) / max(time.time() - t0, 1e-6)
            print(f"  {i + len(chunk)}/{len(todo)}  {rate:.0f} para/s  eta {(len(todo) - i - len(chunk)) / max(rate, 1e-6):.0f}s", flush=True)
    con.execute("INSERT OR REPLACE INTO meta VALUES (?, ?)", (f"vectors_{slug}", json.dumps({"model": model, "dims": dims, "min_chars": min_chars, "count": len(done) + len(todo)})))
    con.commit()
    print(f"[{model}] done in {time.time() - t0:.0f}s")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=vklib.DEFAULT_MODEL, choices=list(vklib.EMBED_MODELS))
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--min-chars", type=int, default=40, help="skip paragraphs shorter than this (headers, dates)")
    ap.add_argument("--no-vectors", action="store_true")
    ap.add_argument("--rebuild-fts", action="store_true", help="re-create the FTS tables (e.g. after changing the header scheme)")
    args = ap.parse_args()
    os.makedirs(vklib.INDEX_DIR, exist_ok=True)
    con = vklib.connect()
    build_tables(con, args.rebuild_fts)
    if not args.no_vectors:
        build_vectors(con, args.model, args.batch, args.min_chars)


if __name__ == "__main__":
    main()
