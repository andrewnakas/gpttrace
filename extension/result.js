/* Result window. Takes the job the service worker left in
 * chrome.storage.session, runs the site's own engine on it (js/engine,
 * copied in unchanged by tools/build-extension.js) and renders it with the
 * site's findings.js, so a verdict here is the verdict gpttrace.com gives
 * for the same input. Nothing is sent anywhere: the only network requests
 * are the right-clicked image itself and the model files (GET, cached). */
import { esc, verdictHTML, evidenceHTML, metricGrid, actionsHTML, bindActions, reportText, progressHTML, errorHTML } from "./js/ui/findings.js";
import { MODELS, mb } from "./js/engine/models.js";

const SITE = "https://gpttrace.com";
const FULL = { text: "/ai-text-detector/", image: "/ai-image-detector/" };
const $ = (r) => document.querySelector(`[data-role="${r}"]`);
const result = $("result"), source = $("source");

let textEngine = null, imageEngine = null;
const loadText = () => (textEngine ||= import("./js/engine/text/index.js"));
const loadImage = () => (imageEngine ||= import("./js/engine/image/index.js"));

const prefs = {
  async get(k, d) { try { const v = (await chrome.storage.local.get(k))[k]; return v == null ? d : v; } catch (_) { return d; } },
  set(k, v) { return chrome.storage.local.set({ [k]: v }).catch(() => {}); }
};

// Model files are cached by the engine's workers under their full URL.
async function cached(url) {
  try { return !!(await (await caches.open("gpttrace-models-v1")).match(url)); } catch (_) { return false; }
}
const textModelURL = () => new URL(MODELS.text.base[0] + MODELS.text.files.model, location.href).href;

let seq = 0, lastJob = null;
async function take() {
  const { job } = await chrome.storage.session.get("job");
  if (!job || (lastJob && job.id === lastJob.id)) return;
  lastJob = job;
  await chrome.storage.session.remove("job"); // keep the content only in this page
  if (job.kind === "text") checkText(job.text);
  else if (job.kind === "image") checkImage(job.src);
}
chrome.storage.onChanged.addListener((ch, area) => { if (area === "session" && ch.job && ch.job.newValue) take(); });
take();

function done(r) {
  document.title = `${r.verdict.label} · GPTTrace`;
  Object.assign(result.dataset, { verdict: r.verdict.key, prob: r.prob.toFixed(4) });
}

function fullReport(kind) {
  const how = kind === "image" ? "Drop the image there for the error-level and spectrum views." : "Paste the text there to see the flagged phrases highlighted.";
  return `<div class="ext-full"><a class="btn ghost sm" href="${SITE}${FULL[kind]}" target="_blank" rel="noopener noreferrer" data-role="full">Full report on gpttrace.com</a><p class="note">${how} Nothing is passed in the link.</p></div>`;
}

/* ---- text ------------------------------------------------------------ */

async function checkText(text, forceDeep) {
  const my = ++seq; // a newer check supersedes this one
  delete result.dataset.verdict;
  source.hidden = false;
  source.innerHTML = `<div class="section-title">Selected text</div><p class="excerpt">${esc(text.slice(0, 1500))}${text.length > 1500 ? "…" : ""}</p>`;
  if (!text.trim()) { result.innerHTML = errorHTML("No text selected", "Select some text on the page, then right-click it."); return; }
  const deep = forceDeep || (await prefs.get("deepText", false));
  const M = MODELS.text;
  result.innerHTML = progressHTML("Analysing…");
  try {
    const { analyze } = await loadText();
    const r = await analyze(text, {
      deep,
      onModelProgress: (p) => { if (my === seq) result.innerHTML = progressHTML(`Downloading the deep-scan model (${mb(M.bytes)}, once)…`, p); }
    });
    if (my !== seq) return;
    const s = r.stats;
    const ranDeep = r.evidence.some((e) => e.id === "neural-text");
    let h = verdictHTML(r, `${s.wordCount.toLocaleString()} words${ranDeep ? " · deep scan" : ""}`);
    if (s.wordCount < 150) h += `<p class="note warn">Only ${s.wordCount} words. Every detector is unreliable on short text; 150+ words is the minimum for a meaningful estimate.</p>`;
    h += metricGrid([["Sentences", s.sentCount], ["Avg sentence", s.avgLen + " w"], ["Burstiness", s.burstiness], ["Vocabulary", s.ttr]]);
    h += await deepPanel(ranDeep, s.wordCount);
    h += evidenceHTML(r) + actionsHTML() + fullReport("text") +
      `<p class="note">An estimate from writing patterns, not proof. People write like this too — especially in formal or translated English. Never use a detector score alone to accuse someone.</p>`;
    result.innerHTML = h;
    bindActions(result, () => reportText(r, "text analysis"), () => ({ ...r, signs: r.signs.map((x) => ({ ...x, matches: x.matches.slice(0, 50) })) }), "text");
    const run = result.querySelector("[data-act=deep]"), always = result.querySelector("[data-act=always-deep]");
    if (run) run.addEventListener("click", () => checkText(text, true));
    if (always) always.addEventListener("change", () => { prefs.set("deepText", always.checked); if (always.checked && !ranDeep) checkText(text, true); });
    done(r);
  } catch (e) {
    if (my === seq) result.innerHTML = errorHTML("Analysis failed", e.message);
  }
}

// The deep scan is opt-in because of its size; say so before downloading.
async function deepPanel(ranDeep, words) {
  const M = MODELS.text, always = await prefs.get("deepText", false);
  const toggle = `<label class="toggle"><input type="checkbox" data-act="always-deep"${always ? " checked" : ""}> Always run the deep scan</label>`;
  if (ranDeep) return `<div class="ext-deep"><p><strong>Deep scan included.</strong> The neural text model ran on this device and is weighed with the other signals below.</p>${toggle}</div>`;
  if (words < 30) return `<div class="ext-deep"><p>The deep scan needs at least 30 words.</p></div>`;
  const have = await cached(textModelURL());
  return `<div class="ext-deep"><p><strong>Standard check.</strong> The deep scan adds a neural model (Ettin-150M fine-tuned on the RAID and MAGE benchmarks). On our 2,467-passage test set it lifts accuracy from AUC 0.80 to 0.96.
    ${have ? "The model is already on this device." : `It is a <strong>${mb(M.bytes)} download</strong> from gpttrace.com, once; after that it is cached here. Your text is not sent.`}</p>
    <button type="button" class="btn primary sm" data-act="deep">${have ? "Run deep scan" : `Download ${mb(M.bytes)} and run deep scan`}</button>${toggle}</div>`;
}

/* ---- image ----------------------------------------------------------- */

async function checkImage(src) {
  const my = ++seq;
  delete result.dataset.verdict;
  source.hidden = false;
  source.innerHTML = `<div class="section-title">Image</div><canvas data-role="preview" hidden></canvas><div class="src-meta">${esc(src.startsWith("data:") ? "Embedded image (data: URL)" : src.slice(0, 300))}</div>`;
  result.innerHTML = progressHTML("Reading the image…");
  let file;
  try { file = await fetchImage(src); }
  catch (e) { if (my === seq) await cannotRead(src, e); return; }
  if (my !== seq) return;
  const M = MODELS.image;
  try {
    const { analyze } = await loadImage();
    const r = await analyze(file, {
      deep: true,
      onStage: (m) => { if (my === seq) result.innerHTML = progressHTML(m); },
      onModelProgress: (p) => { if (my === seq) result.innerHTML = progressHTML(`Downloading the image models (${mb(M.bytes + (MODELS.image2 ? MODELS.image2.bytes : 0))}, once)…`, p); }
    });
    if (my !== seq) return;
    const c = source.querySelector("[data-role=preview]"), b = r.bitmap, k = Math.min(1, 760 / b.width);
    c.width = Math.round(b.width * k); c.height = Math.round(b.height * k);
    c.getContext("2d").drawImage(b, 0, 0, c.width, c.height);
    c.hidden = false;
    renderImage(r, file);
    done(r);
  } catch (e) {
    if (my === seq) result.innerHTML = errorHTML("Could not analyse this image", e.message + (/decode|source image/i.test(e.message) ? " — Chrome can't decode this format." : ""));
  }
}

async function fetchImage(src) {
  const u = new URL(src);
  if (!/^(https?|data):$/.test(u.protocol)) throw Object.assign(new Error(`a ${u.protocol} image`), { local: true });
  const res = await fetch(src, { cache: "force-cache" });
  if (!res.ok) throw new Error(`the server answered ${res.status}`);
  const blob = await res.blob();
  let name = "image";
  if (u.protocol !== "data:") try { name = decodeURIComponent(u.pathname.split("/").pop()) || name; } catch (_) {}
  return new File([blob], name.slice(0, 120), { type: blob.type });
}

// Most image hosts allow other pages to read their images; some don't. Then
// the extension asks for that one host, at the moment it's needed.
async function cannotRead(src, err) {
  const u = new URL(src);
  const site = `<a href="${SITE}${FULL.image}" target="_blank" rel="noopener noreferrer">gpttrace.com${FULL.image}</a>`;
  if (err.local) { result.innerHTML = errorHTML("Can't read this image", `It exists only inside the page (${u.protocol} URL), so the extension can't fetch it. Save it and drop it on the full detector.`) + `<p class="note">Full detector: ${site}</p>`; return; }
  const origins = [`${u.protocol}//${u.hostname}/*`]; // match patterns take no port
  if (await chrome.permissions.contains({ origins })) {
    result.innerHTML = errorHTML("Could not download the image", `${err.message}. Save it and drop it on the full detector instead.`) + `<p class="note">Full detector: ${site}</p>`;
    return;
  }
  result.innerHTML = `<div class="finding"><div class="finding-head"><span class="sev info">access</span> Allow GPTTrace to read images from ${esc(u.hostname)}?</div>
    <p class="finding-detail">${esc(u.hostname)} doesn't let other pages read its images, so your browser needs your permission for the extension to download this one, the same way the page did. The image is then checked here; nothing is sent to ${esc(u.hostname)} or anywhere else. You can remove the permission any time in the extension's details.</p>
    <div class="btn-row" style="margin-top:10px"><button type="button" class="btn primary sm" data-act="grant">Allow ${esc(u.hostname)}</button></div></div>`;
  result.querySelector("[data-act=grant]").addEventListener("click", async () => {
    if (await chrome.permissions.request({ origins })) checkImage(src);
  });
}

function renderImage(r, file) {
  const m = r.meta, c = m.c2pa;
  let h = verdictHTML(r, `${esc(file.name)} · ${r.dims.width}×${r.dims.height} · ${(file.size / 1024).toFixed(0)} KB`);
  h += evidenceHTML(r);
  if (c.present && c.active) {
    const a = c.active;
    h += `<div class="section-title">Content Credentials (C2PA)</div>` + metricGrid([
      ["Claim generator", a.claimGenerator || a.generators.join(", ") || "—"], ["Signed by", c.signedBy || "—"],
      ["Actions", a.actions.map((x) => x.action.replace(/^c2pa\./, "")).join(", ") || "—"], ["Manifests", c.manifests.length]
    ]) + `<p class="note">Manifest structure read locally; the cryptographic signature is not verified here. Use the <a href="${SITE}/c2pa-viewer/" target="_blank" rel="noopener noreferrer">C2PA viewer</a> for the full manifest.</p>`;
  }
  const keys = ["Make", "Model", "Software", "DateTimeOriginal", "LensModel", "FNumber", "ExposureTime", "ISO", "FocalLength", "Artist", "CreatorTool", "DigitalSourceType", "Credit"];
  const rows = keys.filter((k) => m.exif[k] != null && m.exif[k] !== "").map((k) => [k, String(m.exif[k]).slice(0, 60)]);
  if (rows.length) h += `<div class="section-title">Metadata</div>` + metricGrid(rows);
  h += actionsHTML() + fullReport("image") +
    `<p class="note">Evidence, not proof. Screenshots and social-media re-uploads strip metadata, and no detector catches every generator. For news or legal use, check the original source too.</p>`;
  result.innerHTML = h;
  bindActions(result, () => reportText(r, `image analysis of ${file.name}`), () => ({ ...r, bitmap: undefined, pixels: r.pixels && { ...r.pixels, ela: { ...r.pixels.ela, image: undefined }, spectrum: { ...r.pixels.spectrum, image: undefined } } }), "image");
}
