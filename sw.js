/* Service worker: makes GPTTrace installable and usable offline, and receives
 * files shared to the installed app (manifest share_target).
 * - Pages: network first, cached copy when offline.
 * - Scripts, styles, runtimes, icons: served from cache, refreshed in the
 *   background.
 * - Models are not touched here: the engine workers keep them in their own
 *   cache ("gpttrace-models-v1").
 * - A share (POST /share/) is read here, inside the browser, and parked in a
 *   local cache for the page to pick up. It is never sent to a server. */
const SHELL = "gpttrace-shell-v1";
const SHARE = "gpttrace-share";

self.addEventListener("install", (e) => {
  e.waitUntil(caches.open(SHELL).then((c) => c.add("/")).then(() => self.skipWaiting()));
});
self.addEventListener("activate", (e) => {
  e.waitUntil(caches.keys().then((ks) => Promise.all(ks.filter((k) => k.startsWith("gpttrace-shell-") && k !== SHELL).map((k) => caches.delete(k)))).then(() => self.clients.claim()));
});

self.addEventListener("fetch", (e) => {
  const req = e.request, url = new URL(req.url);
  if (url.origin !== location.origin) return;
  if (req.method === "POST" && url.pathname === "/share/") return e.respondWith(share(req));
  if (req.method !== "GET" || url.pathname.startsWith("/models/") || url.pathname.startsWith("/__models/")) return;
  if (req.mode === "navigate") return e.respondWith(page(req));
  if (/^\/(js|css|vendor|assets)\//.test(url.pathname) || url.pathname === "/manifest.webmanifest") return e.respondWith(asset(req, e));
});

async function page(req) {
  const c = await caches.open(SHELL);
  try {
    const res = await fetch(req);
    if (res.ok) c.put(req, res.clone());
    return res;
  } catch {
    return (await c.match(req, { ignoreSearch: true })) || (await c.match("/")) || Response.error();
  }
}

async function asset(req, e) {
  const c = await caches.open(SHELL);
  const hit = await c.match(req);
  const net = fetch(req).then((res) => { if (res.ok) c.put(req, res.clone()); return res; });
  if (hit) { e.waitUntil(net.catch(() => {})); return hit; }
  return net;
}

async function share(req) {
  const form = await req.formData();
  const c = await caches.open(SHARE);
  await Promise.all((await c.keys()).map((k) => c.delete(k)));
  const file = form.getAll("file").find((f) => f && f.size);
  if (file) {
    await c.put("/__share/file", new Response(file, { headers: { "Content-Type": file.type || "application/octet-stream", "X-Name": encodeURIComponent(file.name || "shared") } }));
  } else {
    const text = [form.get("title"), form.get("text"), form.get("url")].filter(Boolean).join("\n\n");
    if (text) await c.put("/__share/text", new Response(text));
  }
  return Response.redirect("/?shared=1", 303);
}
