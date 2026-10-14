/* Image tool: dropzone / paste → image engine → findings + ELA / spectrum
 * views. */
import { esc, verdictHTML, evidenceHTML, metricGrid, actionsHTML, bindActions, reportText, progressHTML, errorHTML, bindDropzone, onPaste, pref } from "./findings.js";
import { MODELS, mb } from "../engine/models.js";
import { bindPro } from "./pro-toggle.js";

let engine = null;
const loadEngine = () => (engine ||= import("../engine/image/index.js"));

export function mount(root) {
  const $ = (r) => root.querySelector(`[data-role="${r}"]`);
  const dz = $("dropzone"), input = $("file"), result = $("result"), preview = $("preview"), deep = $("deep");
  const views = { original: $("view-original"), ela: $("view-ela"), fft: $("view-fft") };
  if (!result.dataset.empty) result.dataset.empty = result.innerHTML;
  deep.checked = pref.get("deep-image", true);
  deep.closest("label").querySelector("[data-role=deep-size]").textContent = mb(MODELS.image.bytes);
  deep.addEventListener("change", () => pref.set("deep-image", deep.checked));
  const pro = bindPro(root);
  let current = null;

  bindDropzone(dz, input, handle);
  onPaste(/^image\//, handle);
  dz.addEventListener("pointerenter", loadEngine, { once: true });
  $("clear").addEventListener("click", () => { current = null; preview.hidden = true; dz.hidden = false; input.value = ""; result.innerHTML = result.dataset.empty; });
  $("again").addEventListener("click", () => current && handle(current));
  root.querySelectorAll("[data-view]").forEach((t) => t.addEventListener("click", () => show(t.dataset.view)));

  const api = { handle };
  function show(view) {
    root.querySelectorAll("[data-view]").forEach((t) => t.classList.toggle("active", t.dataset.view === view));
    Object.entries(views).forEach(([k, c]) => (c.hidden = k !== view));
  }
  function paint(canvas, bmp, maxW = 900) {
    const s = Math.min(1, maxW / bmp.width);
    canvas.width = Math.round(bmp.width * s); canvas.height = Math.round(bmp.height * s);
    canvas.getContext("2d").drawImage(bmp, 0, 0, canvas.width, canvas.height);
  }

  let seq = 0;
  async function handle(file) {
    const my = ++seq; // a newer file supersedes this one
    if (!/^image\//.test(file.type) && !/\.(heic|heif|avif|webp|png|jpe?g|gif|bmp|tiff?)$/i.test(file.name)) {
      if (/^(video|audio)\//.test(file.type)) { document.dispatchEvent(new CustomEvent("gt:route", { detail: file })); return; }
      result.innerHTML = errorHTML("Not an image", `${file.name} looks like ${file.type || "an unknown type"}.`);
      return;
    }
    current = file;
    dz.hidden = true; preview.hidden = false; show("original");
    result.innerHTML = progressHTML("Loading the analyser…");
    try {
      const { analyze } = await loadEngine();
      const r = await analyze(file, {
        deep: deep.checked,
        pro: pro.on,
        onStage: (m) => (result.innerHTML = progressHTML(m)),
        onModelProgress: (p) => (result.innerHTML = progressHTML(`Downloading the neural model (${mb(MODELS.image.bytes)}, once)…`, p))
      });
      if (my !== seq) return;
      paint(views.original, r.bitmap);
      if (r.pixels) { paint(views.ela, r.pixels.ela.image); paint(views.fft, r.pixels.spectrum.image, 512); }
      render(r, file);
    } catch (e) {
      result.innerHTML = errorHTML("Could not analyse this image", e.message + (/decode|source image/i.test(e.message) ? " — this browser may not decode the format (HEIC needs Safari)." : ""));
    }
  }

  function render(r, file) {
    const m = r.meta, c = m.c2pa;
    let h = verdictHTML(r, `${esc(file.name)} · ${r.dims.width}×${r.dims.height} · ${(file.size / 1024).toFixed(0)} KB`);
    if (!deep.checked) h += `<p class="note">The neural classifier is off, so this is a metadata and forensics check only. Turn on “Neural model” for images without metadata.</p>`;
    h += evidenceHTML(r);
    if (c.present && c.active) {
      const a = c.active;
      h += `<div class="section-title">Content Credentials (C2PA)</div>` + metricGrid([
        ["Claim generator", a.claimGenerator || a.generators.join(", ") || "—"], ["Signed by", c.signedBy || "—"],
        ["Actions", a.actions.map((x) => x.action.replace(/^c2pa\./, "")).join(", ") || "—"], ["Manifests", c.manifests.length]
      ]) + `<p class="note">Manifest structure read locally; the cryptographic signature is not verified here. Use the <a href="/c2pa-viewer/">C2PA viewer</a> for the full manifest.</p>`;
    }
    const keys = ["Make", "Model", "Software", "DateTimeOriginal", "LensModel", "FNumber", "ExposureTime", "ISO", "FocalLength", "Artist", "CreatorTool", "DigitalSourceType", "Credit"];
    const rows = keys.filter((k) => m.exif[k] != null && m.exif[k] !== "").map((k) => [k, String(m.exif[k]).slice(0, 60)]);
    if (rows.length) h += `<div class="section-title">Metadata</div>` + metricGrid(rows);
    if (m.png.length) h += `<details class="measures"><summary>PNG text chunks (${m.png.length})</summary><div class="findings">${m.png.map((p) => `<div class="finding"><div class="finding-head"><span class="sev low">${esc(p.chunk)}</span> ${esc(p.key)}</div><p class="finding-detail mono">${esc(p.value.slice(0, 600))}${p.value.length > 600 ? "…" : ""}</p></div>`).join("")}</div></details>`;
    h += actionsHTML() + `<p class="note">Evidence, not proof. Screenshots and social-media re-uploads strip metadata, and no detector catches every generator. For news or legal use, check the original source too.</p>`;
    result.innerHTML = h;
    bindActions(result, () => reportText(r, `image analysis of ${file.name}`), () => ({ ...r, bitmap: undefined, pixels: r.pixels && { ...r.pixels, ela: { ...r.pixels.ela, image: undefined }, spectrum: { ...r.pixels.spectrum, image: undefined } } }), "image");
  }
  return api;
}
