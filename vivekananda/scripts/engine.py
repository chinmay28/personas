"""One entry point shared by the CLI, the API server and the phase 8 eval: answer(con, question, mode, backend, model).

backend 'prompted'  phase 3 cascade on a stock Ollama model (router + per-mode prompts, LLM query expansion), modes.ask
backend 'unified'   the fine-tuned single model (phase 7), unified.ask: one prompt, model emits the mode tag
Both return the same result dict: mode, text, citations[{n, para_id, section_id, citation}], see_also, quote_guard, stats, router,
cascade, model, backend, queries, retrieved, seconds (+ raw model output, stripped by slim()).
"""
from __future__ import annotations

import os
import sys

import requests

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import modes  # noqa: E402
import unified  # noqa: E402
import vklib  # noqa: E402

BACKENDS = ("prompted", "unified")
MODES = ("auto", "scholar", "oracle", "persona", "mixed")
DEFAULT_BACKEND = os.environ.get("VK_BACKEND", "prompted")


def default_model(backend: str) -> str:
    return unified.UNIFIED_MODEL if backend == "unified" else modes.DEV_MODEL


def answer(con, question: str, mode: str = "auto", backend: str = DEFAULT_BACKEND, model: str | None = None, k: int = 8,
           expand: bool = True, on_mode=None, on_token=None) -> dict:
    if backend not in BACKENDS:
        raise ValueError(f"backend must be one of {BACKENDS}")
    if mode not in MODES:
        raise ValueError(f"mode must be one of {MODES}")
    model = model or default_model(backend)
    if backend == "unified":
        return unified.ask(con, question, mode, model, k, on_token=on_token, on_mode=on_mode, expand=expand)
    r = modes.ask(con, question, mode, model, k, expand=expand, on_mode=on_mode)
    r["backend"] = "prompted"
    return r


def slim(r: dict) -> dict:
    """The public (API / CLI --json) form: raw model output removed, everything else kept."""
    out = {k: v for k, v in r.items() if k not in ("raw", "oracle_raw")}
    if "parts" in out:
        out["parts"] = [slim(p) for p in out["parts"]]
    return out


def ollama_models() -> list[str]:
    r = requests.get(f"{vklib.OLLAMA}/api/tags", timeout=10)
    r.raise_for_status()
    return sorted(m["name"] for m in r.json().get("models", []))


def section_payload(con, sid: str) -> dict:
    s = vklib.get_section(con, sid)
    paras = [{"id": r["id"], "para_idx": r["para_idx"], "text": r["text"]}
             for r in con.execute("SELECT id, para_idx, text FROM paragraphs WHERE section_id = ? ORDER BY para_idx", (sid,))]
    return {"id": s["id"], "citation": vklib.citation(s), "volume": s["volume"], "work": s["work"], "title": s["title"], "genre": s["genre"],
            "date": s["date"], "recipient": s.get("recipient_full") or s.get("recipient"), "place": s["place"], "toc_path": s["toc_path"],
            "char_count": s["char_count"], "paragraphs": paras}


def paragraph_payload(con, pid: str) -> dict:
    p = vklib.get_paragraph(con, pid)
    s = vklib.get_section(con, p["section_id"])
    return {"id": p["id"], "section_id": p["section_id"], "para_idx": p["para_idx"], "text": p["text"], "citation": vklib.citation(s),
            "sentences": [p["text"][a:b] for a, b in p["sentences"]]}


def search_payload(con, q: str, k: int = 10) -> dict:
    paras = []
    for h in vklib.search_paragraphs(con, q, k=k):
        p = vklib.get_paragraph(con, h.id)
        s = vklib.get_section(con, p["section_id"])
        paras.append({"para_id": h.id, "section_id": s["id"], "citation": vklib.citation(s), "genre": s["genre"], "score": round(h.score, 5),
                      "dense_rank": h.dense_rank, "bm25_rank": h.bm25_rank, "text": p["text"]})
    secs = []
    for h in vklib.search_sections(con, q, k=min(k, 10)):
        s = vklib.get_section(con, h.id)
        secs.append({"section_id": h.id, "citation": vklib.citation(s), "genre": s["genre"], "score": round(h.score, 5)})
    return {"q": q, "paragraphs": paras, "sections": secs}
