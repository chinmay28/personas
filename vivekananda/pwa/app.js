/* Vivekananda PWA: one chat screen over the local API (scripts/server.py). No build step, no dependencies.
   History lives in localStorage only; nothing is stored server side. */
(() => {
  "use strict";
  const $ = (id) => document.getElementById(id);
  const chat = $("chat"), form = $("form"), qBox = $("q"), modeSel = $("mode"), sendBtn = $("send"), statusEl = $("status");
  const drawer = $("drawer"), drawerBody = $("drawerBody"), drawerTitle = $("drawerTitle"), scrim = $("scrim");
  const settings = $("settings"), backendSel = $("backend"), modelInp = $("model"), showMeta = $("showMeta"), healthDump = $("healthDump");
  const HKEY = "vk_history_v1", SKEY = "vk_settings_v1";
  const BADGE = { oracle: "oracle · his own words, verbatim", scholar: "scholar · what he said or wrote, cited",
                  persona: "persona · reasoning beyond his record", mixed: "mixed · what I said / what I would say" };
  let history = load(HKEY, []);
  let prefs = load(SKEY, { backend: "", model: "", showMeta: false });
  let health = null;
  const sectionCache = new Map();

  function load(k, d) { try { return JSON.parse(localStorage.getItem(k)) ?? d; } catch { return d; } }
  function save() { localStorage.setItem(HKEY, JSON.stringify(history.slice(-200))); }
  function savePrefs() { localStorage.setItem(SKEY, JSON.stringify(prefs)); }
  function esc(s) { return String(s).replace(/[&<>"]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c])); }
  function scrollDown() { chat.scrollTop = chat.scrollHeight; }

  // ---------------------------------------------------------------- health / status
  async function refreshHealth() {
    try {
      const r = await fetch("/health", { cache: "no-store" });
      health = await r.json();
      const m = (prefs.model || health.model), b = (prefs.backend || health.backend);
      statusEl.textContent = `${b} · ${m}` + (health.model_present === false ? " (model not in Ollama)" : "");
      statusEl.className = "status" + (health.ok ? "" : " bad");
      $("models").innerHTML = (health.models || []).map((x) => `<option value="${esc(x)}">`).join("");
      healthDump.textContent = JSON.stringify(health, null, 1);
    } catch (e) {
      health = null;
      statusEl.textContent = navigator.onLine ? "server unreachable" : "offline";
      statusEl.className = "status bad";
    }
  }

  // ---------------------------------------------------------------- rendering
  function renderText(text, cites) {
    // [n] -> clickable citation markers; quoted spans in italics
    let html = esc(text);
    html = html.replace(/“([^”]+)”/g, '<span class="quote">“$1”</span>');
    html = html.replace(/\[(\d+)\]/g, (m, n) => {
      const c = (cites || []).find((x) => String(x.n) === n);
      return c ? `<span class="cite" data-sid="${esc(c.section_id)}" data-pid="${esc(c.para_id)}" title="${esc(c.citation)}">${n}</span>` : m;
    });
    return html;
  }

  function entryCard(e) {
    const mode = e.mode || "pending";
    const head = `<div class="head"><span class="badge ${mode}">${esc(e.mode ? (BADGE[e.mode] || e.mode) : (e.status || "thinking…"))}</span>`
      + (e.cascade ? `<span>${esc(e.cascade)}</span>` : "") + (e.router && e.router.reason && e.mode !== "pending" ? `<span>· ${esc(e.router.reason)}</span>` : "") + `</div>`;
    let body;
    if (e.error) body = `<div class="error">${esc(e.error)}</div>`;
    else if (e.text != null) body = `<div class="answer">${renderText(e.text, e.citations)}</div>`;
    else if (e.stream) body = `<div class="answer stream cursor">${esc(e.stream)}</div>`;
    else body = `<div class="answer cursor"></div>`;
    let chips = "";
    if (e.citations && e.citations.length) {
      chips = `<div class="chips">` + e.citations.map((c) =>
        `<button class="chip" data-sid="${esc(c.section_id)}" data-pid="${esc(c.para_id)}"><b>[${c.n}]</b>${esc(c.citation)}</button>`).join("")
        + (e.see_also || []).map((s) => `<button class="chip more" data-sid="${esc(s.section_id)}">see also · ${esc(s.citation)}</button>`).join("") + `</div>`;
    }
    let meta = "";
    if (prefs.showMeta && e.meta) {
      const m = e.meta;
      const g = m.quote_guard || [];
      meta = `<div class="meta">${esc(m.backend || "")} · ${esc(m.model || "")} · ${m.seconds}s` +
        (g.length ? ` · quotes verified ${g.filter((x) => x.verified).length}/${g.length}` : "") +
        (m.stats ? ` · cited ${m.stats.sentences_cited}/${m.stats.sentences} sentences` : "") +
        (m.queries && m.queries.length > 1 ? `\nqueries: ${esc(m.queries.join(" | "))}` : "") +
        (m.retrieved ? `\nretrieved: ${m.retrieved.map((r) => r.citation).join("; ")}` : "") + `</div>`;
    }
    return head + body + chips + meta;
  }

  function render() {
    $("empty").hidden = history.length > 0;
    for (const e of history) {
      let user = document.getElementById("u-" + e.id), card = document.getElementById("c-" + e.id);
      if (!user) {
        user = document.createElement("div"); user.className = "msg user"; user.id = "u-" + e.id;
        user.textContent = e.q + (e.reqMode && e.reqMode !== "auto" ? `  (${e.reqMode})` : "");
        chat.appendChild(user);
        card = document.createElement("div"); card.id = "c-" + e.id; chat.appendChild(card);
      }
      card.className = "card " + (e.mode || "pending");
      card.innerHTML = entryCard(e);
    }
  }
  function rerender(e) {
    const card = document.getElementById("c-" + e.id);
    if (card) { card.className = "card " + (e.mode || "pending"); card.innerHTML = entryCard(e); }
  }

  // ---------------------------------------------------------------- asking (SSE over fetch so POST works)
  async function ask(e) {
    const body = { question: e.q, mode: e.reqMode, stream: true };
    if (prefs.backend) body.backend = prefs.backend;
    if (prefs.model) body.model = prefs.model;
    let resp;
    try {
      resp = await fetch("/ask", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body) });
    } catch (err) {
      e.error = navigator.onLine ? "Server unreachable. Start scripts/server.py on this network and retry." : "You are offline.";
      e.mode = e.mode || "pending"; rerender(e); save(); return;
    }
    if (!resp.ok) {
      let msg = resp.status + " " + resp.statusText;
      try { msg = (await resp.json()).error || msg; } catch {}
      e.error = msg; rerender(e); save(); return;
    }
    const reader = resp.body.getReader(), dec = new TextDecoder();
    let buf = "";
    while (true) {
      const { value, done } = await reader.read();
      if (done) break;
      buf += dec.decode(value, { stream: true });
      let i;
      while ((i = buf.indexOf("\n\n")) >= 0) {
        const chunk = buf.slice(0, i); buf = buf.slice(i + 2);
        let ev = "message", data = "";
        for (const line of chunk.split("\n")) {
          if (line.startsWith("event:")) ev = line.slice(6).trim();
          else if (line.startsWith("data:")) data += line.slice(5).trim();
        }
        let d = {};
        try { d = JSON.parse(data || "{}"); } catch {}
        handle(e, ev, d);
      }
    }
    if (e.text == null && !e.error) e.error = "Connection closed before the answer arrived.";
    rerender(e); save();
  }

  function handle(e, ev, d) {
    if (ev === "status") { e.status = d.stage === "retrieving" ? "searching the Complete Works…" : (d.stage === "generating" ? "composing…" : d.stage); }
    else if (ev === "mode") { e.mode = d.mode; e.router = d.router; if (d.cascade) e.cascade = d.cascade; e.stream = ""; }
    else if (ev === "token") { e.stream = (e.stream || "") + d.t; }
    else if (ev === "text") { e.text = d.text; e.mode = d.mode; if (d.cascade) e.cascade = d.cascade; e.stream = ""; }
    else if (ev === "citations") { e.citations = d.citations || []; e.see_also = d.see_also || []; }
    else if (ev === "meta") { e.meta = d; if (!e.router && d.router) e.router = d.router; }
    else if (ev === "error") { e.error = d.error; }
    else if (ev === "done") { save(); }
    rerender(e);
    scrollDown();
  }

  form.addEventListener("submit", async (ev) => {
    ev.preventDefault();
    const q = qBox.value.trim();
    if (!q) return;
    const e = { id: Date.now().toString(36), q, reqMode: modeSel.value, ts: new Date().toISOString() };
    history.push(e); save();
    qBox.value = ""; qBox.style.height = "auto";
    render(); scrollDown();
    sendBtn.disabled = true;
    try { await ask(e); } finally { sendBtn.disabled = false; qBox.focus(); }
  });
  qBox.addEventListener("input", () => { qBox.style.height = "auto"; qBox.style.height = Math.min(qBox.scrollHeight, 160) + "px"; });
  qBox.addEventListener("keydown", (ev) => { if (ev.key === "Enter" && !ev.shiftKey) { ev.preventDefault(); form.requestSubmit(); } });

  // ---------------------------------------------------------------- citation drawer
  async function openSection(sid, pid) {
    drawerTitle.textContent = "loading…"; drawerBody.innerHTML = "";
    drawer.hidden = false; scrim.hidden = false;
    let s = sectionCache.get(sid);
    if (!s) {
      try { const r = await fetch("/section/" + encodeURIComponent(sid)); s = await r.json(); if (!r.ok) throw new Error(s.error || r.status); sectionCache.set(sid, s); }
      catch (err) { drawerTitle.textContent = "Section unavailable"; drawerBody.innerHTML = `<div class="error">${esc(err.message || err)}</div>`; return; }
    }
    drawerTitle.textContent = s.citation;
    const meta = [s.genre, s.date, s.recipient ? "to " + s.recipient : "", s.place, (s.toc_path || []).join(" › ")].filter(Boolean).join(" · ");
    drawerBody.innerHTML = `<div class="secmeta">${esc(meta)} · ${s.paragraphs.length} paragraphs</div>` +
      s.paragraphs.map((p) => `<p id="dp-${esc(p.id)}" class="${p.id === pid ? "hl" : ""}">${esc(p.text)}</p>`).join("");
    const target = pid && document.getElementById("dp-" + pid);
    if (target) target.scrollIntoView({ block: "start", behavior: "instant" });
    else drawerBody.scrollTop = 0;
  }
  function closeDrawers() { drawer.hidden = true; settings.hidden = true; scrim.hidden = true; }
  chat.addEventListener("click", (ev) => {
    const t = ev.target.closest("[data-sid]");
    if (t) openSection(t.dataset.sid, t.dataset.pid);
  });
  $("drawerClose").addEventListener("click", closeDrawers);
  $("settingsClose").addEventListener("click", closeDrawers);
  scrim.addEventListener("click", closeDrawers);
  document.addEventListener("keydown", (ev) => { if (ev.key === "Escape") closeDrawers(); });

  // ---------------------------------------------------------------- settings
  $("settingsBtn").addEventListener("click", () => { settings.hidden = false; scrim.hidden = false; refreshHealth(); });
  backendSel.value = prefs.backend || ""; modelInp.value = prefs.model || ""; showMeta.checked = !!prefs.showMeta;
  backendSel.addEventListener("change", () => { prefs.backend = backendSel.value; savePrefs(); refreshHealth(); });
  modelInp.addEventListener("change", () => { prefs.model = modelInp.value.trim(); savePrefs(); refreshHealth(); });
  showMeta.addEventListener("change", () => { prefs.showMeta = showMeta.checked; savePrefs(); render(); });
  $("clearHistory").addEventListener("click", () => { if (confirm("Clear the conversation history on this device?")) { history = []; save(); chat.querySelectorAll(".msg, .card").forEach((n) => n.remove()); render(); closeDrawers(); } });
  $("exportHistory").addEventListener("click", () => {
    const blob = new Blob([JSON.stringify(history, null, 1)], { type: "application/json" });
    const a = document.createElement("a"); a.href = URL.createObjectURL(blob); a.download = "vivekananda-history.json"; a.click();
  });

  // ---------------------------------------------------------------- boot
  window.addEventListener("online", refreshHealth);
  window.addEventListener("offline", refreshHealth);
  if ("serviceWorker" in navigator) navigator.serviceWorker.register("/sw.js").catch(() => {});
  render(); scrollDown(); refreshHealth();
  // /?q=...&mode=... asks immediately (shareable links, smoke tests)
  const params = new URLSearchParams(location.search);
  if (params.get("q")) {
    qBox.value = params.get("q");
    if (params.get("mode")) modeSel.value = params.get("mode");
    window.history.replaceState(null, "", location.pathname);
    form.requestSubmit();
  }
})();
