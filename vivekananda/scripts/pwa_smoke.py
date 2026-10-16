"""PWA smoke test: drives headless Chrome over the DevTools protocol against a running server (scripts/server.py on 127.0.0.1:8765):
asks one question via /?q=, waits for the answer card, screenshots it (/tmp/pwa_answer.png), opens the first citation chip and checks the
drawer shows the section with the cited paragraph highlighted (/tmp/pwa_drawer.png). Needs google-chrome and websocket-client.

    .venv/bin/python scripts/pwa_smoke.py
"""
import json, subprocess, time, base64, urllib.request, sys
import websocket
PORT = 9333
chrome = subprocess.Popen(["google-chrome", "--headless=new", "--disable-gpu", "--no-sandbox", "--hide-scrollbars", f"--remote-debugging-port={PORT}",
                           "--window-size=430,900", "--user-data-dir=/tmp/cdp-profile", "about:blank"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
try:
    for _ in range(50):
        try:
            tabs = json.load(urllib.request.urlopen(f"http://127.0.0.1:{PORT}/json")); break
        except Exception: time.sleep(0.2)
    page = next(t for t in tabs if t["type"] == "page")
    ws = websocket.create_connection(page["webSocketDebuggerUrl"], suppress_origin=True); ws.settimeout(90)
    mid = [0]
    def call(method, **params):
        mid[0] += 1; ws.send(json.dumps({"id": mid[0], "method": method, "params": params}))
        while True:
            m = json.loads(ws.recv())
            if m.get("id") == mid[0]: return m.get("result", {})
    def js(expr):
        r = call("Runtime.evaluate", expression=expr, returnByValue=True); return r.get("result", {}).get("value")
    call("Page.enable"); call("Runtime.enable")
    call("Emulation.setDeviceMetricsOverride", width=430, height=900, deviceScaleFactor=1, mobile=True)
    call("Page.navigate", url="http://127.0.0.1:8765/?q=What%20is%20the%20ideal%20of%20womanhood%3F")
    t0 = time.time()
    while time.time() - t0 < 400:
        n = js("document.querySelectorAll('.card .chips .chip').length")
        err = js("(document.querySelector('.card .error')||{}).textContent||''")
        if n or err: break
        time.sleep(2)
    print("chips:", n, "error:", err, "after", round(time.time()-t0), "s")
    print("badge:", js("(document.querySelector('.card .badge')||{}).textContent"))
    print("text head:", (js("(document.querySelector('.card .answer')||{}).textContent") or "")[:160])
    print("inline markers:", js("document.querySelectorAll('.card .answer .cite').length"))
    open("/tmp/pwa_answer.png","wb").write(base64.b64decode(call("Page.captureScreenshot", format="png")["data"]))
    js("document.querySelector('.card .chips .chip').click()")
    t0 = time.time()
    while time.time() - t0 < 30:
        if js("document.querySelectorAll('#drawerBody p').length"): break
        time.sleep(0.5)
    print("drawer title:", js("document.getElementById('drawerTitle').textContent"), "| paras:", js("document.querySelectorAll('#drawerBody p').length"),
          "| highlighted:", js("document.querySelectorAll('#drawerBody p.hl').length"))
    open("/tmp/pwa_drawer.png","wb").write(base64.b64decode(call("Page.captureScreenshot", format="png")["data"]))
    print("history entries:", js("JSON.parse(localStorage.getItem('vk_history_v1')||'[]').length"), "| sw registered:", js("navigator.serviceWorker.controller ? 'yes' : 'pending'"))
finally:
    chrome.terminate()
