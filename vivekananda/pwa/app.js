/* Vivekananda PWA: one chat screen over the local API (scripts/server.py). No build step, no dependencies.
   Conversations ("sessions") live in localStorage only; nothing is stored server side. A session can be created, renamed, deleted,
   and exported as Markdown (download / share sheet) or PDF (print view, "Save as PDF" in the browser's print dialog). */
(() => {
  "use strict";
  const $ = (id) => document.getElementById(id);
  const chat = $("chat"), form = $("form"), qBox = $("q"), modeSel = $("mode"), sendBtn = $("send"), statusEl = $("status");
  const drawer = $("drawer"), drawerBody = $("drawerBody"), drawerTitle = $("drawerTitle"), scrim = $("scrim");
  const settings = $("settings"), backendSel = $("backend"), modelInp = $("model"), showMeta = $("showMeta"), healthDump = $("healthDump");
  const sessionsDrawer = $("sessions"), sessionList = $("sessionList"), sessionName = $("sessionName");
  const OLD_HKEY = "vk_history_v1", SKEY = "vk_settings_v1", DKEY = "vk_sessions_v2";
  const MAX_SESSIONS = 100, MAX_ENTRIES = 200;
  const BADGE = { oracle: "oracle · his own words, verbatim", scholar: "scholar · what he said or wrote, cited",
                  persona: "persona · reasoning beyond his record", mixed: "mixed · what I said / what I would say" };
  let prefs = load(SKEY, { backend: "", model: "", showMeta: false });
  let db = loadSessions();
  let history = current().entries;   // the active session's entries (alias kept so the rest of the code reads as before)
  let health = null;
  const sectionCache = new Map();

  function load(k, d) { try { return JSON.parse(localStorage.getItem(k)) ?? d; } catch { return d; } }
  function savePrefs() { localStorage.setItem(SKEY, JSON.stringify(prefs)); }
  function esc(s) { return String(s).replace(/[&<>"]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c])); }
  function scrollDown() { chat.scrollTop = chat.scrollHeight; }
  function uid() { return Date.now().toString(36) + Math.random().toString(36).slice(2, 6); }
  function fmtDate(iso) { try { return new Date(iso).toLocaleString(undefined, { dateStyle: "medium", timeStyle: "short" }); } catch { return iso; } }

  // ---------------------------------------------------------------- sessions
  function newSession(title) {
    const now = new Date().toISOString();
    return { id: uid(), title: title || "", created: now, updated: now, entries: [] };
  }
  function loadSessions() {
    let d = load(DKEY, null);
    if (!d || !Array.isArray(d.sessions)) {
      d = { sessions: [], current: null };
      const old = load(OLD_HKEY, []);          // migrate the single pre-session history, if any
      if (Array.isArray(old) && old.length) {
        const s = newSession("Earlier conversation");
        s.entries = old.slice(-MAX_ENTRIES); s.created = old[0].ts || s.created; s.updated = old[old.length - 1].ts || s.updated;
        d.sessions.push(s);
      }
      localStorage.removeItem(OLD_HKEY);
    }
    if (!d.sessions.length) d.sessions.push(newSession());
    if (!d.sessions.some((s) => s.id === d.current)) d.current = d.sessions[d.sessions.length - 1].id;
    return d;
  }
  function current() { return db.sessions.find((s) => s.id === db.current); }
  function save() {
    const s = current();
    s.entries = s.entries.slice(-MAX_ENTRIES);
    if (!s.title && s.entries.length) s.title = s.entries[0].q.slice(0, 60);
    s.updated = new Date().toISOString();
    if (db.sessions.length > MAX_SESSIONS) {   // drop the oldest empty-or-untouched sessions first
      db.sessions.sort((a, b) => a.updated.localeCompare(b.updated));
      while (db.sessions.length > MAX_SESSIONS) { const i = db.sessions.findIndex((x) => x.id !== db.current); if (i < 0) break; db.sessions.splice(i, 1); }
    }
    localStorage.setItem(DKEY, JSON.stringify(db));
  }
  function titleOf(s) { return s.title || (s.entries.length ? s.entries[0].q.slice(0, 60) : "New conversation"); }
  function clearChat() { chat.querySelectorAll(".msg, .card").forEach((n) => n.remove()); }
  function switchTo(id) {
    if (!db.sessions.some((s) => s.id === id)) return;
    db.current = id; history = current().entries;
    localStorage.setItem(DKEY, JSON.stringify(db));
    clearChat(); render(); scrollDown(); updateHeader(); closeDrawers();
  }
  function createSession() {
    const cur = current();
    if (cur && !cur.entries.length) { switchTo(cur.id); return cur; }   // an empty session is already "new"
    const s = newSession(); db.sessions.push(s); switchTo(s.id); return s;
  }
  function deleteSession(id) {
    const i = db.sessions.findIndex((s) => s.id === id);
    if (i < 0) return;
    db.sessions.splice(i, 1);
    if (!db.sessions.length) db.sessions.push(newSession());
    if (db.current === id) db.current = db.sessions[db.sessions.length - 1].id;
    history = current().entries;
    localStorage.setItem(DKEY, JSON.stringify(db));
    clearChat(); render(); updateHeader(); renderSessionList();
  }
  function renameSession(id, title) {
    const s = db.sessions.find((x) => x.id === id);
    if (!s) return;
    s.title = (title || "").trim().slice(0, 80);
    localStorage.setItem(DKEY, JSON.stringify(db));
    updateHeader(); renderSessionList();
  }
  function updateHeader() {
    const s = current();
    sessionName.textContent = titleOf(s);
    document.title = (s.entries.length ? titleOf(s) + " · " : "") + "Vivekananda";
  }
  function renderSessionList() {
    const sorted = db.sessions.slice().sort((a, b) => b.updated.localeCompare(a.updated));
    sessionList.innerHTML = sorted.map((s) => `
      <div class="session ${s.id === db.current ? "current" : ""}" data-id="${esc(s.id)}">
        <button class="open" data-act="open" data-id="${esc(s.id)}">
          <div class="stitle">${esc(titleOf(s))}</div>
          <div class="smeta">${s.entries.length} ${s.entries.length === 1 ? "exchange" : "exchanges"} · ${esc(fmtDate(s.updated))}</div>
        </button>
        <button class="icon-btn" data-act="rename" data-id="${esc(s.id)}" title="Rename" aria-label="Rename">&#9998;</button>
        <button class="icon-btn danger" data-act="delete" data-id="${esc(s.id)}" title="Delete" aria-label="Delete">&times;</button>
      </div>`).join("");
  }
  sessionList.addEventListener("click", (ev) => {
    const b = ev.target.closest("[data-act]");
    if (!b) return;
    const id = b.dataset.id, s = db.sessions.find((x) => x.id === id);
    if (b.dataset.act === "open") switchTo(id);
    else if (b.dataset.act === "rename") { const t = prompt("Conversation name", titleOf(s)); if (t !== null) renameSession(id, t); }
    else if (b.dataset.act === "delete") {
      if (confirm(`Delete "${titleOf(s)}" (${s.entries.length} exchanges) from this device?`)) deleteSession(id);
    }
  });
  $("sessionsBtn").addEventListener("click", () => { renderSessionList(); sessionsDrawer.hidden = false; scrim.hidden = false; });
  $("sessionsClose").addEventListener("click", closeDrawers);
  $("newSession").addEventListener("click", () => { createSession(); qBox.focus(); });
  $("newSessionTop").addEventListener("click", () => { createSession(); qBox.focus(); });

  // ---------------------------------------------------------------- export (current session)
  function plainText(text) { return String(text || ""); }
  function toMarkdown(s) {
    const lines = [`# Swami Vivekananda — ${titleOf(s)}`, "", `_Conversation started ${fmtDate(s.created)}; exported ${fmtDate(new Date().toISOString())}._`,
                   "_Every answer is in his voice and labelled by grounding: oracle = his own words verbatim, scholar = what he said or wrote with citations, persona = reasoning beyond his record, marked as such._", ""];
    for (const e of s.entries) {
      lines.push(`## You`, "", plainText(e.q) + (e.reqMode && e.reqMode !== "auto" ? `  _(mode: ${e.reqMode})_` : ""), "");
      const label = e.mode ? (BADGE[e.mode] || e.mode) : "no answer";
      lines.push(`## Swamiji  _(${label}${e.cascade ? ", " + e.cascade : ""})_`, "");
      if (e.error) lines.push(`> Error: ${e.error}`, "");
      else if (e.text != null) lines.push(...plainText(e.text).split("\n").map((l) => l), "");
      if (e.citations && e.citations.length) {
        lines.push(...e.citations.map((c) => `[${c.n}] ${c.citation}${c.para_id ? "  (" + c.para_id + ")" : ""}`));
        if (e.see_also && e.see_also.length) lines.push(`See also: ${e.see_also.map((x) => x.citation).join("; ")}`);
        lines.push("");
      }
      if (e.rating) lines.push(`_Feedback: ${e.rating === "up" ? "👍" : "👎"}${e.comment ? " — " + e.comment : ""}_`, "");
      if (e.meta && (e.meta.backend || e.meta.model)) lines.push(`<sub>${[e.meta.backend, e.meta.model, e.meta.seconds != null ? e.meta.seconds + " s" : ""].filter(Boolean).join(" · ")}</sub>`, "");
      lines.push("---", "");
    }
    lines.push("Generated by the Vivekananda persona (github.com/chinmay28/personas). Quotations are verbatim from the Complete Works; everything else is model output.");
    return lines.join("\n");
  }
  function localDate(iso) { const d = new Date(iso), p = (n) => String(n).padStart(2, "0"); return `${d.getFullYear()}-${p(d.getMonth() + 1)}-${p(d.getDate())}`; }
  function fileStem(s) { return "vivekananda-" + (titleOf(s).toLowerCase().replace(/[^a-z0-9]+/g, "-").replace(/^-|-$/g, "").slice(0, 40) || "conversation") + "-" + localDate(s.created); }
  async function deliver(blob, name) {
    const file = new File([blob], name, { type: blob.type });
    if (navigator.share && navigator.canShare && navigator.canShare({ files: [file] })) {
      try { await navigator.share({ files: [file], title: name }); return; } catch (err) { if (err && err.name === "AbortError") return; }
    }
    const a = document.createElement("a"); a.href = URL.createObjectURL(blob); a.download = name; document.body.appendChild(a); a.click();
    setTimeout(() => { URL.revokeObjectURL(a.href); a.remove(); }, 2000);
  }
  function exportMarkdown() { const s = current(); return deliver(new Blob([toMarkdown(s)], { type: "text/markdown" }), fileStem(s) + ".md"); }
  function printView(s) {
    const pv = $("printView");
    pv.innerHTML = `<h1>Swami Vivekananda — ${esc(titleOf(s))}</h1>
      <p class="pmeta">Conversation started ${esc(fmtDate(s.created))} · exported ${esc(fmtDate(new Date().toISOString()))}. Oracle = his own words verbatim; scholar = what he said or wrote, cited; persona = reasoning beyond his record, marked.</p>` +
      s.entries.map((e) => `
        <section class="pentry ${esc(e.mode || "")}">
          <p class="pq"><b>You:</b> ${esc(e.q)}${e.reqMode && e.reqMode !== "auto" ? ` <i>(mode: ${esc(e.reqMode)})</i>` : ""}</p>
          <p class="plabel">Swamiji · ${esc(e.mode ? (BADGE[e.mode] || e.mode) : "no answer")}${e.cascade ? " · " + esc(e.cascade) : ""}</p>
          ${e.error ? `<p class="perr">Error: ${esc(e.error)}</p>` : `<div class="ptext">${renderText(e.text || "", e.citations)}</div>`}
          ${e.citations && e.citations.length ? `<ol class="pcites">${e.citations.map((c) => `<li value="${c.n}">${esc(c.citation)}</li>`).join("")}</ol>` : ""}
          ${e.see_also && e.see_also.length ? `<p class="psee">See also: ${e.see_also.map((x) => esc(x.citation)).join("; ")}</p>` : ""}
        </section>`).join("") +
      `<p class="pfoot">Generated by the Vivekananda persona (github.com/chinmay28/personas). Quotations are verbatim from the Complete Works; everything else is model output.</p>`;
    return pv;
  }
  function exportPdf() {
    const s = current();
    if (!s.entries.length) { alert("Nothing to export yet."); return; }
    printView(s);
    document.body.classList.add("printing");
    const done = () => { document.body.classList.remove("printing"); window.removeEventListener("afterprint", done); };
    window.addEventListener("afterprint", done);
    setTimeout(() => { window.print(); setTimeout(done, 1500); }, 50);   // the timeout covers browsers that never fire afterprint
  }
  $("exportMd").addEventListener("click", () => { closeDrawers(); exportMarkdown(); });
  $("exportPdf").addEventListener("click", () => { closeDrawers(); exportPdf(); });
  $("exportMdTop").addEventListener("click", () => { closeDrawers(); exportMarkdown(); });
  $("exportPdfTop").addEventListener("click", () => { closeDrawers(); exportPdf(); });

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
    let fb = "";
    if (!e.error && e.text != null) {
      const r = e.rating || "";
      fb = `<div class="feedback" data-eid="${esc(e.id)}">` +
        `<button class="thumb ${r === "up" ? "on" : ""}" data-rate="up" title="Good answer" aria-label="Thumbs up" aria-pressed="${r === "up"}">&#128077;</button>` +
        `<button class="thumb ${r === "down" ? "on" : ""}" data-rate="down" title="Poor answer" aria-label="Thumbs down" aria-pressed="${r === "down"}">&#128078;</button>` +
        (e.ratingNote ? `<span class="fbnote">${esc(e.ratingNote)}</span>` : "") + `</div>`;
    }
    return head + body + chips + meta + fb;
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
    else if (ev === "meta") { e.meta = d; if (!e.router && d.router) e.router = d.router; if (d.pre_route) e.pre_route = d.pre_route; }
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
    const session = current();
    history.push(e); save(); updateHeader();
    qBox.value = ""; qBox.style.height = "auto";
    render(); scrollDown();
    sendBtn.disabled = true;
    try { await ask(e); } finally { sendBtn.disabled = false; qBox.focus(); if (current() === session) updateHeader(); }
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
  function closeDrawers() { drawer.hidden = true; settings.hidden = true; sessionsDrawer.hidden = true; scrim.hidden = true; }
  chat.addEventListener("click", (ev) => {
    const th = ev.target.closest(".thumb");
    if (th) { rate(th.closest(".feedback").dataset.eid, th.dataset.rate); return; }
    const t = ev.target.closest("[data-sid]");
    if (t) openSection(t.dataset.sid, t.dataset.pid);
  });

  // ---------------------------------------------------------------- feedback (server-side log: eval/feedback.jsonl)
  async function rate(eid, rating) {
    const e = history.find((x) => x.id === eid);
    if (!e) return;
    let comment;
    if (e.rating === rating) rating = "clear";                      // tapping the active thumb withdraws it
    else if (rating === "down") { comment = prompt("What was wrong? (optional)", e.comment || "") ; if (comment === null) return; }
    e.rating = rating === "clear" ? "" : rating;
    e.comment = rating === "down" ? (comment !== undefined ? comment.trim() : (e.comment || "")) : "";   // a comment belongs to a thumbs-down only
    e.ratingNote = "saving…"; rerender(e);
    const m = e.meta || {};
    const body = { rating, entry_id: e.id, session_id: current().id, question: e.q, req_mode: e.reqMode, mode: e.mode, backend: m.backend || prefs.backend || (health && health.backend) || null,
                   model: m.model || prefs.model || (health && health.model) || null, text: e.text, citations: (e.citations || []).map((c) => ({ n: c.n, para_id: c.para_id, section_id: c.section_id, citation: c.citation })),
                   queries: m.queries || null, seconds: m.seconds ?? null, pre_route: e.pre_route || null, cascade: e.cascade || null, comment: e.comment || null, client_ts: new Date().toISOString() };
    try {
      const r = await fetch("/feedback", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body) });
      if (!r.ok) throw new Error((await r.json()).error || r.status);
      e.ratingNote = rating === "clear" ? "" : (rating === "down" && e.comment ? "noted: " + e.comment : "thanks, noted");
    } catch (err) {
      e.ratingNote = "could not reach the server; not recorded";
    }
    save(); rerender(e);
  }
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
  $("clearHistory").addEventListener("click", () => {
    const n = db.sessions.reduce((a, s) => a + s.entries.length, 0);
    if (confirm(`Delete all ${db.sessions.length} conversations (${n} exchanges) on this device?`)) {
      db = { sessions: [newSession()], current: null }; db.current = db.sessions[0].id; history = current().entries;
      localStorage.setItem(DKEY, JSON.stringify(db)); clearChat(); render(); updateHeader(); closeDrawers();
    }
  });
  $("exportHistory").addEventListener("click", () => {
    deliver(new Blob([JSON.stringify(db, null, 1)], { type: "application/json" }), "vivekananda-conversations.json");
  });

  // ---------------------------------------------------------------- boot
  window.addEventListener("online", refreshHealth);
  window.addEventListener("offline", refreshHealth);
  if ("serviceWorker" in navigator) navigator.serviceWorker.register("/sw.js").catch(() => {});
  render(); scrollDown(); updateHeader(); refreshHealth();
  // /?q=...&mode=... asks immediately (shareable links, smoke tests); &new=1 starts it in a fresh conversation
  const params = new URLSearchParams(location.search);
  if (params.get("new")) createSession();
  if (params.get("q")) {
    qBox.value = params.get("q");
    if (params.get("mode")) modeSel.value = params.get("mode");
    window.history.replaceState(null, "", location.pathname);
    form.requestSubmit();
  }
  // test/debug hooks (not used by the UI)
  window.vk = { sessions: () => db, current, createSession, deleteSession, renameSession, switchTo, toMarkdown: () => toMarkdown(current()), printView: () => printView(current()).innerHTML };
})();
