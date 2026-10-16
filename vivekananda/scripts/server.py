"""Phase 9 API server + PWA host (single process, stdlib only).

    .venv/bin/python scripts/server.py [--host 0.0.0.0] [--port 8765] [--backend prompted|unified] [--model M] [-k 8]

Endpoints
  POST /ask        {"question", "mode": auto|scholar|oracle|persona|mixed, "backend"?, "model"?, "k"?, "stream": true}
                   streams Server-Sent Events: status, mode (first), token (persona prose as it is generated), text, citations,
                   meta, done | error.  With "stream": false (or GET /ask?q=...&stream=0) the full result JSON is returned instead,
                   which is exactly what the CLI prints with --json.
  GET  /ask?q=&mode=      same as POST, for EventSource clients
  GET  /search?q=&k=      raw hybrid retrieval (paragraphs + sections)
  GET  /section/{id}      section metadata + all paragraphs (citation drill-down)
  GET  /paragraph/{id}    one paragraph with its sentences and citation
  GET  /health            Ollama reachability, models present, index counts, active backend/model
  GET  /                  the PWA (static files from pwa/)
"""
from __future__ import annotations

import argparse
import json
import mimetypes
import os
import sys
import threading
import time
import traceback
import urllib.parse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import engine  # noqa: E402
import vklib  # noqa: E402

PWA_DIR = os.path.join(vklib.ROOT, "pwa")
VERSION = "0.1"
CFG = {"backend": engine.DEFAULT_BACKEND, "model": None, "k": 8, "expand": True}
_tls = threading.local()
MIME = {".webmanifest": "application/manifest+json", ".js": "text/javascript; charset=utf-8", ".css": "text/css; charset=utf-8",
        ".html": "text/html; charset=utf-8", ".svg": "image/svg+xml", ".png": "image/png", ".json": "application/json"}


def con():
    c = getattr(_tls, "con", None)
    if c is None:
        c = _tls.con = vklib.connect()
    return c


def health() -> dict:
    out = {"ok": True, "version": VERSION, "backend": CFG["backend"], "model": CFG["model"] or engine.default_model(CFG["backend"]), "k": CFG["k"],
           "backends": list(engine.BACKENDS), "modes": list(engine.MODES), "default_models": {b: engine.default_model(b) for b in engine.BACKENDS}}
    try:
        models = engine.ollama_models()
        out["ollama"] = "ok"
        out["models"] = models
        out["model_present"] = out["model"] in models
        out["embed_models_present"] = {m: any(x == vklib.base_model(m) or x.startswith(vklib.base_model(m) + ":") for x in models)
                                      for m in vklib.FUSE_MODELS}
        if not out["model_present"]:
            out["ok"] = False
    except Exception as e:  # noqa: BLE001
        out.update(ollama=f"error: {e}", ok=False, models=[], model_present=False)
    try:
        c = con()
        out["sections"] = c.execute("SELECT count(*) FROM sections").fetchone()[0]
        out["paragraphs"] = c.execute("SELECT count(*) FROM paragraphs").fetchone()[0]
    except Exception as e:  # noqa: BLE001
        out.update(index=f"error: {e}", ok=False)
    return out


class Handler(BaseHTTPRequestHandler):
    server_version = f"vivekananda-api/{VERSION}"
    protocol_version = "HTTP/1.1"

    def log_message(self, fmt, *args):
        sys.stderr.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')} {self.address_string()} {fmt % args}\n")

    # ---------------------------------------------------------------- helpers
    def _cors(self):
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")

    def _json(self, code: int, obj):
        body = json.dumps(obj, ensure_ascii=False).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self._cors()
        self.end_headers()
        self.wfile.write(body)

    def _read_json(self) -> dict:
        n = int(self.headers.get("Content-Length") or 0)
        raw = self.rfile.read(n) if n else b""
        if not raw:
            return {}
        try:
            j = json.loads(raw.decode("utf-8"))
            return j if isinstance(j, dict) else {}
        except json.JSONDecodeError:
            return {}

    def _static(self, path: str):
        rel = urllib.parse.unquote(path.lstrip("/")) or "index.html"
        full = os.path.normpath(os.path.join(PWA_DIR, rel))
        if not full.startswith(PWA_DIR + os.sep) or not os.path.isfile(full):
            self._json(404, {"error": "not found"})
            return
        ext = os.path.splitext(full)[1].lower()
        ctype = MIME.get(ext) or mimetypes.guess_type(full)[0] or "application/octet-stream"
        with open(full, "rb") as f:
            body = f.read()
        self.send_response(200)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-cache")  # the service worker does the caching
        if rel == "sw.js":
            self.send_header("Service-Worker-Allowed", "/")
        self.end_headers()
        self.wfile.write(body)

    # ---------------------------------------------------------------- routes
    def do_OPTIONS(self):
        self.send_response(204)
        self._cors()
        self.send_header("Content-Length", "0")
        self.end_headers()

    def do_GET(self):
        u = urllib.parse.urlsplit(self.path)
        qs = {k: v[0] for k, v in urllib.parse.parse_qs(u.query).items()}
        path = u.path
        try:
            if path == "/health":
                return self._json(200, health())
            if path == "/search":
                q = (qs.get("q") or "").strip()
                if not q:
                    return self._json(400, {"error": "q required"})
                return self._json(200, engine.search_payload(con(), q, int(qs.get("k") or 10)))
            if path.startswith("/section/"):
                sid = path.split("/", 2)[2]
                if not con().execute("SELECT 1 FROM sections WHERE id = ?", (sid,)).fetchone():
                    return self._json(404, {"error": "no such section"})
                return self._json(200, engine.section_payload(con(), sid))
            if path.startswith("/paragraph/"):
                pid = path.split("/", 2)[2]
                if not con().execute("SELECT 1 FROM paragraphs WHERE id = ?", (pid,)).fetchone():
                    return self._json(404, {"error": "no such paragraph"})
                return self._json(200, engine.paragraph_payload(con(), pid))
            if path == "/ask":
                params = {"question": qs.get("q") or qs.get("question") or "", "mode": qs.get("mode", "auto"), "backend": qs.get("backend"),
                          "model": qs.get("model"), "k": qs.get("k"), "stream": qs.get("stream", "1") not in ("0", "false")}
                return self._ask(params)
            if path.startswith("/section") or path.startswith("/paragraph") or path.startswith("/ask"):
                return self._json(404, {"error": "not found"})
            return self._static(path)
        except (BrokenPipeError, ConnectionResetError):
            pass
        except Exception as e:  # noqa: BLE001
            traceback.print_exc()
            try:
                self._json(500, {"error": str(e)})
            except Exception:  # noqa: BLE001
                pass

    def do_POST(self):
        path = urllib.parse.urlsplit(self.path).path
        if path != "/ask":
            return self._json(404, {"error": "not found"})
        body = self._read_json()
        params = {"question": body.get("question") or body.get("q") or "", "mode": body.get("mode") or "auto", "backend": body.get("backend"),
                  "model": body.get("model"), "k": body.get("k"), "stream": body.get("stream", True)}
        try:
            self._ask(params)
        except (BrokenPipeError, ConnectionResetError):
            pass
        except Exception as e:  # noqa: BLE001
            traceback.print_exc()
            try:
                self._json(500, {"error": str(e)})
            except Exception:  # noqa: BLE001
                pass

    # ---------------------------------------------------------------- /ask
    def _ask(self, params: dict):
        question = str(params["question"]).strip()
        mode = str(params["mode"] or "auto").lower()
        backend = str(params.get("backend") or CFG["backend"]).lower()
        model = params.get("model") or CFG["model"] or engine.default_model(backend)
        try:
            k = int(params.get("k") or CFG["k"])
        except (TypeError, ValueError):
            k = CFG["k"]
        if not question:
            return self._json(400, {"error": "question required"})
        if mode not in engine.MODES:
            return self._json(400, {"error": f"mode must be one of {list(engine.MODES)}"})
        if backend not in engine.BACKENDS:
            return self._json(400, {"error": f"backend must be one of {list(engine.BACKENDS)}"})
        if not params.get("stream", True):
            r = engine.answer(con(), question, mode, backend, model, k, expand=CFG["expand"])
            return self._json(200, engine.slim(r))

        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream; charset=utf-8")
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Accel-Buffering", "no")
        self.send_header("Connection", "close")
        self._cors()
        self.end_headers()
        lock = threading.Lock()

        def send(event: str, data):
            with lock:
                self.wfile.write(f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n".encode("utf-8"))
                self.wfile.flush()

        send("status", {"stage": "retrieving", "backend": backend, "model": model})

        def on_mode(m, info):
            send("mode", {"mode": m, "router": info.get("router"), "cascade": info.get("cascade")})
            send("status", {"stage": "generating", "mode": m})

        def on_token(t):
            send("token", {"t": t})

        try:
            r = engine.answer(con(), question, mode, backend, model, k, expand=CFG["expand"], on_mode=on_mode, on_token=on_token)
        except (BrokenPipeError, ConnectionResetError):
            self.log_message("client disconnected mid-stream")
            return
        except Exception as e:  # noqa: BLE001
            traceback.print_exc()
            send("error", {"error": str(e)})
            return
        s = engine.slim(r)
        send("text", {"mode": s["mode"], "text": s["text"], "cascade": s.get("cascade")})
        send("citations", {"citations": s.get("citations", []), "see_also": s.get("see_also", [])})
        meta = {k2: v for k2, v in s.items() if k2 not in ("text", "citations", "see_also", "parts")}
        send("meta", meta)
        send("done", {"seconds": s.get("seconds")})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default=os.environ.get("VK_HOST", "0.0.0.0"), help="bind address (0.0.0.0 = reachable from the LAN)")
    ap.add_argument("--port", type=int, default=int(os.environ.get("VK_PORT", "8765")))
    ap.add_argument("--backend", default=engine.DEFAULT_BACKEND, choices=engine.BACKENDS)
    ap.add_argument("--model", default=None, help="Ollama model (default: per backend)")
    ap.add_argument("-k", type=int, default=8)
    ap.add_argument("--no-expand", action="store_true", help="prompted backend: skip LLM query expansion")
    args = ap.parse_args()
    CFG.update(backend=args.backend, model=args.model, k=args.k, expand=not args.no_expand)
    h = health()
    print(f"vivekananda api {VERSION}: backend={CFG['backend']} model={h['model']} present={h.get('model_present')} ollama={h.get('ollama')} "
          f"sections={h.get('sections')} paragraphs={h.get('paragraphs')}", flush=True)
    srv = ThreadingHTTPServer((args.host, args.port), Handler)
    srv.daemon_threads = True
    print(f"listening on http://{args.host}:{args.port}/  (PWA at /, API: /ask /search /section/{{id}} /paragraph/{{id}} /health)", flush=True)
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
