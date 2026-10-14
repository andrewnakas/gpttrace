/* The only script every page loads. Finds [data-tool] shells rendered by the
 * page builder and imports that tool's module; engines and models load
 * later, on first use. */
const TOOLS = { text: "./text-tool.js", image: "./image-tool.js", audio: "./audio-tool.js", video: "./video-tool.js", c2pa: "./c2pa-tool.js", pro: "./pro.js" };
const ACCEPT = { image: /^image\//, video: /^video\//, audio: /^audio\// };
const handles = {};
const ready = {};

function boot() {
  document.querySelectorAll("[data-tool]").forEach((el) => {
    const mod = TOOLS[el.dataset.tool];
    if (!mod) return;
    ready[el.dataset.tool] = import(mod).then((m) => { handles[el.dataset.tool] = m.mount(el, el.dataset.preset ? JSON.parse(el.dataset.preset) : {}); })
      .catch((e) => { el.querySelector(".tool-status")?.replaceChildren(document.createTextNode("This tool failed to load: " + e.message)); });
  });
  document.querySelectorAll("[data-tabs]").forEach((box) => {
    box.querySelectorAll("[data-tab]").forEach((b) => b.addEventListener("click", () => activate(box, b.dataset.tab)));
  });
  // A tool that receives a file of another kind hands it on.
  document.addEventListener("gt:route", (e) => {
    const f = e.detail;
    const tool = Object.keys(ACCEPT).find((k) => ACCEPT[k].test(f.type) || (k === "audio" && /\.(m4a|opus|flac)$/i.test(f.name)));
    if (!tool) return;
    const panel = document.querySelector(`[data-panel="${tool}"]`);
    if (panel) activate(panel.closest("[data-tabs]"), tool);
    const h = handles[tool];
    if (h && h.handle) h.handle(f);
    else if (!panel) location.href = `/ai-${tool}-detector/`;
  });
  const y = document.getElementById("year");
  if (y) y.textContent = new Date().getFullYear();
  if (new URLSearchParams(location.search).has("shared")) takeShare();
  // Installable app + offline use. Skipped under test automation so the
  // browser tests see the network exactly as a first visit does.
  if ("serviceWorker" in navigator && !navigator.webdriver) navigator.serviceWorker.register("/sw.js").catch(() => {});
}
// Something was shared to the installed app (see sw.js): the service worker
// parked it in a local cache; hand it to the right tool and clear it.
async function takeShare() {
  history.replaceState(null, "", location.pathname);
  if (!("caches" in self)) return;
  const c = await caches.open("gpttrace-share");
  const f = await c.match("/__share/file"), t = await c.match("/__share/text");
  await Promise.all((await c.keys()).map((k) => c.delete(k)));
  if (f) {
    const blob = await f.blob();
    deliver(new File([blob], decodeURIComponent(f.headers.get("X-Name") || "shared"), { type: blob.type }));
  } else if (t) deliver(null, await t.text());
}
async function deliver(file, text) {
  if (file) {
    await Promise.all(Object.values(ready));
    document.dispatchEvent(new CustomEvent("gt:route", { detail: file }));
  } else if (text) {
    await ready.text;
    const panel = document.querySelector('[data-panel="text"]');
    if (panel) activate(panel.closest("[data-tabs]"), "text");
    handles.text?.setText(text);
  }
}
// The native apps (apps/mobile) call this when something is shared to them:
// { text } or { url, name, type } for a file the app copied to local storage.
window.gtShare = async (d) => {
  if (d.text) return deliver(null, d.text);
  const blob = await (await fetch(d.url)).blob();
  deliver(new File([blob], d.name || "shared", { type: d.type || blob.type }));
};
function activate(box, tab) {
  box.querySelectorAll("[data-tab]").forEach((b) => { b.classList.toggle("active", b.dataset.tab === tab); b.setAttribute("aria-selected", b.dataset.tab === tab); });
  box.querySelectorAll("[data-panel]").forEach((p) => (p.hidden = p.dataset.panel !== tab));
}
document.readyState === "loading" ? document.addEventListener("DOMContentLoaded", boot) : boot();
