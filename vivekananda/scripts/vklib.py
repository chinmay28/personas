"""Shared helpers: corpus DB access, citations, Ollama embeddings, hybrid search."""
from __future__ import annotations

import json
import os
import re
import sqlite3
import struct
from dataclasses import dataclass

import requests
import sqlite_vec

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CORPUS_DIR = os.path.join(ROOT, "corpus")
INDEX_DIR = os.path.join(ROOT, "index")
DB_PATH = os.path.join(INDEX_DIR, "corpus.db")
OLLAMA = os.environ.get("OLLAMA_HOST_URL", "http://127.0.0.1:11434")

# model -> (dims, document prefix, query prefix). A "/h" suffix means the section header is prepended to each
# paragraph before embedding (contextual chunk header); the Ollama model is the part before "/h".
EMBED_MODELS = {
    "nomic-embed-text": (768, "search_document: ", "search_query: "),
    "bge-m3": (1024, "", ""),
    "qwen3-embedding:0.6b": (1024, "", "Instruct: Given a question, retrieve passages that answer it\nQuery: "),
    "nomic-embed-text/h": (768, "search_document: ", "search_query: "),
    "bge-m3/h": (1024, "", ""),
    "qwen3-embedding:0.6b/h": (1024, "", "Instruct: Given a question, retrieve passages that answer it\nQuery: "),
}


def base_model(model: str) -> str:
    return model.split("/")[0]


def with_header(model: str) -> bool:
    return model.endswith("/h")
DEFAULT_MODEL = "bge-m3/h"
# dense lists fused at paragraph level (eval 2026-10-14: bge-m3/h + bge-m3 + bm25, rrf k=10, cand=100 -> para recall@5 0.70)
FUSE_MODELS = ["bge-m3/h", "bge-m3"]
PARA_RRF_K = 10
PARA_CAND = 100
SMALL_WORDS = {"a", "an", "and", "as", "at", "but", "by", "for", "from", "in", "into", "of", "on", "or", "the", "to", "with", "vs"}


def model_slug(model: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", model.lower()).strip("_")


def title_case(s: str) -> str:
    words = s.split()
    out = []
    for i, w in enumerate(words):
        lw = w.lower()
        if i and lw in SMALL_WORDS and not words[i - 1].endswith(":"):
            out.append(lw)
        elif "-" in w:
            out.append("-".join(p[:1].upper() + p[1:].lower() for p in w.split("-")))
        elif re.fullmatch(r"[IVX]+\.?", w):
            out.append(w)
        else:
            out.append(w[:1].upper() + w[1:].lower())
    s2 = " ".join(out)
    s2 = re.sub(r"^Chapter (\w+):", r"Ch. \1:", s2)
    return s2


def citation(sec: dict) -> str:
    vol = f"Vol {sec['volume']}" if sec.get("volume") else "Front matter"
    work = title_case(re.sub(r"\s*[–-]+\s*(FIRST|SECOND|THIRD|FOURTH|FIFTH) SERIES$", "", sec["work"]))
    if sec.get("genre") == "letter":
        to = sec.get("recipient_full") or sec.get("recipient") or sec["title"]
        parts = [vol, "Epistles", f"To {title_case(to)}"]
        if sec.get("place"):
            parts.append(title_case(sec["place"].split(",")[0].strip()))
        if sec.get("date"):
            parts.append(sec["date"])
        return "[" + ", ".join(parts) + "]"
    title = title_case(re.sub(r"^\d+\s+", "", sec["title"]))
    if sec.get("depth", 2) < 2 or title == work:
        return f"[{vol}, {work}]"
    return f"[{vol}, {work}, {title}]"


def connect(path: str = DB_PATH) -> sqlite3.Connection:
    con = sqlite3.connect(path)
    con.enable_load_extension(True)
    sqlite_vec.load(con)
    con.enable_load_extension(False)
    con.row_factory = sqlite3.Row
    return con


def embed(texts: list[str], model: str = DEFAULT_MODEL, kind: str = "document") -> list[list[float]]:
    dims, dpre, qpre = EMBED_MODELS[model]
    pre = dpre if kind == "document" else qpre
    r = requests.post(f"{OLLAMA}/api/embed", json={"model": base_model(model), "input": [pre + t for t in texts], "truncate": True,
                                                   "options": {"num_ctx": 8192}}, timeout=600)
    r.raise_for_status()
    embs = r.json()["embeddings"]
    assert all(len(e) == dims for e in embs), f"unexpected dims from {model}"
    return embs


def pack(vec: list[float]) -> bytes:
    return struct.pack(f"{len(vec)}f", *vec)


@dataclass
class Hit:
    id: str
    score: float
    dense_rank: int | None
    bm25_rank: int | None


def rrf(rank_lists: list[list[str]], k: int = 60) -> list[Hit]:
    scores: dict[str, float] = {}
    ranks: dict[str, list] = {}
    for li, lst in enumerate(rank_lists):
        for r, pid in enumerate(lst, 1):
            scores[pid] = scores.get(pid, 0.0) + 1.0 / (k + r)
            ranks.setdefault(pid, [None] * len(rank_lists))[li] = r
    out = [Hit(pid, s, ranks[pid][0], ranks[pid][1] if len(rank_lists) > 1 else None) for pid, s in scores.items()]
    out.sort(key=lambda h: -h.score)
    return out


def fts_query(q: str) -> str:
    """Turn free text into a safe FTS5 OR-query of tokens (keeps multiword precision via phrase bonus)."""
    toks = re.findall(r"[A-Za-zÀ-ɏ0-9']+", q)
    toks = [t.replace("'", "''") for t in toks if len(t) > 1]
    if not toks:
        return '""'
    return " OR ".join(f'"{t}"' for t in toks)


def search_paragraphs(con, q: str, model: str = DEFAULT_MODEL, k: int = 10, cand: int = PARA_CAND, qvec=None) -> list[Hit]:
    """Hybrid paragraph search: dense lists for `model` (plus the other FUSE_MODELS when model is the default) and BM25 over
    paragraph text only (headers unweighted: they hurt paragraph precision), fused with RRF."""
    models = FUSE_MODELS if model == DEFAULT_MODEL else [model]
    lists = []
    for m in models:
        slug = model_slug(m)
        v = qvec if (qvec is not None and m == model) else embed([q], m, "query")[0]
        lists.append([r[0] for r in con.execute(
            f"SELECT id FROM vec_para_{slug} WHERE embedding MATCH ? ORDER BY distance LIMIT ?", (pack(v), cand))])
    lists.append([r[0] for r in con.execute(
        "SELECT id FROM fts_para WHERE fts_para MATCH ? ORDER BY bm25(fts_para, 0.0, 0.0, 1.0) LIMIT ?", (fts_query(q), cand))])
    hits = rrf(lists, k=PARA_RRF_K)[:k]
    for h in hits:  # report ranks as (first dense list, bm25)
        h.bm25_rank = None
    ranks = {}
    for li, lst in enumerate(lists):
        for r, pid in enumerate(lst, 1):
            ranks.setdefault(pid, [None] * len(lists))[li] = r
    for h in hits:
        h.dense_rank, h.bm25_rank = ranks[h.id][0], ranks[h.id][-1]
    return hits


def search_sections(con, q: str, model: str = DEFAULT_MODEL, k: int = 10, cand: int = 50, qvec=None) -> list[Hit]:
    """Section ranking: BM25 over full section text fused with best-paragraph dense rank per section."""
    slug = model_slug(model)
    if qvec is None:
        qvec = embed([q], model, "query")[0]
    rows = con.execute(
        f"SELECT v.id, p.section_id FROM (SELECT id, distance FROM vec_para_{slug} WHERE embedding MATCH ? ORDER BY distance LIMIT ?) v "
        f"JOIN paragraphs p ON p.id = v.id ORDER BY v.distance", (pack(qvec), cand * 3)).fetchall()
    dense_secs: list[str] = []
    for r in rows:
        if r[1] not in dense_secs:
            dense_secs.append(r[1])
    dense_secs = dense_secs[:cand]
    bm = [r[0] for r in con.execute(
        "SELECT id FROM fts_sec WHERE fts_sec MATCH ? ORDER BY bm25(fts_sec, 0.0, 8.0, 1.0) LIMIT ?", (fts_query(q), cand))]
    return rrf([dense_secs, bm])[:k]


def get_section(con, sid: str) -> dict:
    r = con.execute("SELECT * FROM sections WHERE id = ?", (sid,)).fetchone()
    d = dict(r)
    d["toc_path"] = json.loads(d["toc_path"])
    return d


def get_paragraph(con, pid: str) -> dict:
    r = con.execute("SELECT * FROM paragraphs WHERE id = ?", (pid,)).fetchone()
    d = dict(r)
    d["sentences"] = json.loads(d["sentences"])
    return d
