"""Inference path for the fine-tuned single model (phase 7 output): one prompt, the model emits <mode>X</mode> then the mode output.

The prompt is byte-for-byte the phase 6 training prompt: UNIFIED_SYSTEM, the numbered-sentence retrieval block from deterministic hybrid
retrieval (no LLM query expansion), the question, then the Qwen3 non-thinking assistant header. Ollama's /api/generate is used in raw mode
so the rendered prompt is exactly the Hugging Face chat template (verified 2026-10-15) and so a mode override can be prefilled as
"<mode>X</mode>\\n" (the chat endpoint cannot prefill with the Qwen3 template). The model's output is then rendered by the same
functions as the prompted system: oracle plans -> verbatim quotes, scholar JSON -> strict citations, persona prose -> quote guard.

    .venv/bin/python scripts/unified.py "What is Maya?" [--mode auto|oracle|scholar|persona|mixed] [--model vivekananda:8b]
"""
from __future__ import annotations

import json
import os
import re
import sys
import time

import requests

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import modes  # noqa: E402
import traindata  # noqa: E402
import vklib  # noqa: E402

UNIFIED_MODEL = os.environ.get("VK_UNIFIED_MODEL", "vivekananda:8b")
MODE_RE = re.compile(r"^\s*<mode>(scholar|oracle|persona)</mode>[ \t]*\r?\n?", re.S)
MODES = ("scholar", "oracle", "persona")


def render_prompt(system: str, user: str, prefill: str = "") -> str:
    """Qwen3 chat template with thinking disabled (matches tokenizer.apply_chat_template(..., enable_thinking=False))."""
    return f"<|im_start|>system\n{system}<|im_end|>\n<|im_start|>user\n{user}<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n{prefill}"


def generate(prompt: str, model: str = UNIFIED_MODEL, temperature: float = 0.3, max_tokens: int = 900, on_token=None) -> str:
    """Raw streaming generation; on_token(text) receives each chunk. Returns the full completion."""
    body = {"model": model, "prompt": prompt, "raw": True, "stream": True,
            "options": {"temperature": temperature, "num_predict": max_tokens, "num_ctx": modes.NUM_CTX, "top_p": 0.9, "top_k": 20,
                        "repeat_penalty": 1.0, "stop": ["<|im_end|>", "<|im_start|>", "<|endoftext|>"]}}
    out = []
    with requests.post(f"{vklib.OLLAMA}/api/generate", json=body, stream=True, timeout=600) as r:
        r.raise_for_status()
        for line in r.iter_lines():
            if not line:
                continue
            j = json.loads(line)
            if j.get("error"):
                raise RuntimeError(j["error"])
            t = j.get("response", "")
            if t:
                out.append(t)
                if on_token:
                    on_token(t)
            if j.get("done"):
                break
    return "".join(out)


def split_mode(raw: str) -> tuple[str | None, str]:
    m = MODE_RE.match(raw)
    if not m:
        return None, raw
    return m.group(1), raw[m.end():]


def render(con, ctx: modes.Context, mode: str, body: str, raw: str) -> dict:
    if mode == "oracle":
        return modes.oracle_from_plan(con, ctx, modes.parse_json(body), raw)
    if mode == "scholar":
        return modes.scholar_from_plan(con, ctx, modes.parse_json(body), raw)
    return modes.persona_from_text(con, ctx, body)


def _generate_mode(con, ctx: modes.Context, user: str, mode: str, model: str, on_token=None, on_mode=None) -> tuple[str, dict]:
    """One generation. mode='auto' lets the model choose; otherwise the tag is prefilled. Returns (mode, result)."""
    prefill = f"<mode>{mode}</mode>\n" if mode in MODES else ""
    prompt = render_prompt(traindata.UNIFIED_SYSTEM, user, prefill)
    state = {"mode": mode if mode in MODES else None, "buf": prefill, "announced": mode in MODES}
    if state["announced"] and on_mode:
        on_mode(mode, {"router": None})

    def tok(t: str):
        state["buf"] += t
        if state["mode"] is None:
            m, rest = split_mode(state["buf"])
            if m:
                state["mode"] = m
                if on_mode and not state["announced"]:
                    state["announced"] = True
                    on_mode(m, {"router": {"mode": m, "reason": "mode tag emitted by the model"}})
                if on_token and m == "persona" and rest:
                    on_token(rest)
            elif len(state["buf"]) > 40 and "<mode>" not in state["buf"]:  # no tag coming: treat as persona prose
                state["mode"] = "persona"
                if on_mode and not state["announced"]:
                    state["announced"] = True
                    on_mode("persona", {"router": {"mode": "persona", "reason": "no mode tag; treated as persona"}})
                if on_token:
                    on_token(state["buf"])
        elif on_token and state["mode"] == "persona":
            on_token(t)

    temperature = 0.6 if mode == "persona" else 0.3
    completion = generate(prompt, model, temperature=temperature, max_tokens=1200, on_token=tok)
    raw = prefill + completion
    m, body = split_mode(raw)
    m = m or "persona"
    if state["mode"] is None and on_mode and not state["announced"]:
        on_mode(m, {"router": {"mode": m, "reason": "mode tag emitted by the model"}})
    return m, render(con, ctx, m, body, raw)


def ask(con, question: str, mode: str = "auto", model: str = UNIFIED_MODEL, k: int = 8, on_token=None, on_mode=None) -> dict:
    """Same result shape as modes.ask (plus backend='unified', mode_tag)."""
    t0 = time.time()
    ctx = modes.retrieve(con, question, k, vklib.DEFAULT_MODEL, None)  # deterministic, as in training
    user = traindata.user_content(ctx, question)
    routed = None
    if mode == "mixed":  # not a trained mode: two prefilled generations, same layout as the prompted system
        _, a = _generate_mode(con, ctx, user, "scholar", model, None, on_mode)
        _, b = _generate_mode(con, ctx, user, "persona", model, on_token, None)
        result = {"mode": "mixed", "parts": [a, b], "text": "WHAT I SAID\n" + a["text"] + "\n\nWHAT I WOULD SAY\n" + b["text"],
                  "citations": a["citations"] + b["citations"], "raw": a.get("raw", "") + "\n---\n" + b.get("raw", "")}
    else:
        chosen, result = _generate_mode(con, ctx, user, mode, model, on_token, on_mode)
        if mode == "auto":
            routed = {"mode": chosen, "reason": "mode tag emitted by the model"}
        if result.get("fallthrough"):
            if on_mode:
                on_mode("persona", {"cascade": "oracle fell through to persona"})
            _, result2 = _generate_mode(con, ctx, user, "persona", model, on_token, None)
            result2["cascade"] = "oracle fell through to persona"
            result2["oracle_raw"] = result.get("raw", "")
            result = result2
        result["mode_tag"] = chosen
    result["question"] = question
    result["router"] = routed
    result["model"] = model
    result["backend"] = "unified"
    result["queries"] = [question]
    result["retrieved"] = [{"para_id": p.para["id"], "section_id": p.section["id"], "citation": p.citation} for p in ctx.passages]
    result["seconds"] = round(time.time() - t0, 1)
    return result


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("question")
    ap.add_argument("--mode", default="auto", choices=["auto", "scholar", "oracle", "persona", "mixed"])
    ap.add_argument("--model", default=UNIFIED_MODEL)
    ap.add_argument("--json", action="store_true")
    a = ap.parse_args()
    r = ask(vklib.connect(), a.question, a.mode, a.model)
    if a.json:
        print(json.dumps(r, ensure_ascii=False, indent=1))
    else:
        print(f"[{r['mode']}]  ({r['model']}, {r['seconds']}s)\n\n{r['text']}")
        for c in r.get("citations", []):
            print(f"  [{c['n']}] {c['citation']}  ({c['para_id']})")
