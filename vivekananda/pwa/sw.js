/* Service worker: caches the app shell so the installed PWA opens offline (and then reports the server as unreachable).
   API calls (/ask, /search, /section, /paragraph, /health) always go to the network. */
const CACHE = "vk-shell-v3";   // bumped 2026-10-17: sessions + export, feedback
const SHELL = ["/", "/index.html", "/app.js", "/style.css", "/manifest.webmanifest", "/icon-192.png", "/icon-512.png"];

self.addEventListener("install", (ev) => {
  ev.waitUntil(caches.open(CACHE).then((c) => c.addAll(SHELL)).then(() => self.skipWaiting()));
});
self.addEventListener("activate", (ev) => {
  ev.waitUntil(caches.keys().then((keys) => Promise.all(keys.filter((k) => k !== CACHE).map((k) => caches.delete(k)))).then(() => self.clients.claim()));
});
self.addEventListener("fetch", (ev) => {
  const url = new URL(ev.request.url);
  if (ev.request.method !== "GET" || url.origin !== location.origin) return;
  if (/^\/(ask|search|section|paragraph|health|feedback)(\/|$|\?)/.test(url.pathname)) return;  // network only
  ev.respondWith(
    caches.match(ev.request, { ignoreSearch: true }).then((cached) => {
      const fresh = fetch(ev.request).then((resp) => {
        if (resp && resp.ok) caches.open(CACHE).then((c) => c.put(ev.request, resp.clone()));
        return resp;
      }).catch(() => cached || (ev.request.mode === "navigate" ? caches.match("/index.html") : undefined));
      return cached || fresh;
    })
  );
});
