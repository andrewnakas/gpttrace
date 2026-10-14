/* Shared result rendering (GPTTrace's renderGenericFinding/escapeHTML,
 * generalised to the evidence model in engine/common/score.js). */

export const esc = (s) => String(s ?? "").replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" })[c]);

const TONE_COLOR = { ai: "var(--danger)", "likely-ai": "var(--warn-2)", mixed: "var(--warn)", "likely-human": "var(--accent)", human: "var(--good)" };

export function verdictHTML(r, subtitle) {
  const pct = Math.round(r.prob * 100);
  const color = TONE_COLOR[r.verdict.tone] || "var(--accent)";
  return `<div class="verdict tone-${r.verdict.tone}" style="--meter:${pct};--meter-color:${color}">
    <div class="verdict-meter" role="img" aria-label="${pct}% likely AI"><span>${pct}<small>%</small></span></div>
    <div class="verdict-body">
      <h3>${esc(r.verdict.label)}</h3>
      <p>${pct}% estimated chance of AI · confidence: <strong>${esc(r.confidence)}</strong>${subtitle ? " · " + subtitle : ""}</p>
    </div>
  </div>`;
}

const SEV = (e) => e.sev || (e.decisive ? "high" : e.contrib > 0.8 ? "high" : e.contrib > 0.25 ? "med" : e.contrib < -0.25 ? "good" : "low");
const dir = (e) => (e.contrib > 0.05 ? `<span class="push ai" title="pushes toward AI">▲ AI</span>` : e.contrib < -0.05 ? `<span class="push human" title="pushes toward human / authentic">▼ human</span>` : "");

export function evidenceItem(e) {
  let h = `<div class="finding">
    <div class="finding-head"><span class="sev ${SEV(e)}">${e.decisive ? "proof" : esc(SEV(e) === "good" ? "real" : SEV(e))}</span><span class="finding-label">${esc(e.label)}</span>${dir(e)}</div>
    <p class="finding-detail">${esc(e.detail || "")}${e.wiki ? ` <a href="${esc(e.wiki)}" target="_blank" rel="noopener">Wikipedia</a>` : ""}</p>`;
  if (e.list && e.list.length) h += `<ul class="finding-list">${e.list.map((x) => `<li>${esc(String(x).slice(0, 160))}</li>`).join("")}</ul>`;
  return h + "</div>";
}

// Evidence that moved the score, strongest first; zero-weight measurements
// go in a collapsed section so they're auditable without being presented as
// reasons.
export function evidenceHTML(r, opts = {}) {
  const reasons = r.evidence.filter((e) => !e.measure && (e.decisive || Math.abs(e.contrib) >= (opts.min ?? 0.05) || (e.sev && e.sev !== "low" && e.x)));
  const info = r.evidence.filter((e) => !e.measure && !reasons.includes(e) && e.sev === "info");
  const measures = r.evidence.filter((e) => e.measure || (!reasons.includes(e) && !info.includes(e)));
  let h = "";
  if (reasons.length) h += `<div class="section-title">Why — evidence that moved the score</div><div class="findings">${reasons.map(evidenceItem).join("")}</div>`;
  else h += `<div class="finding"><div class="finding-head"><span class="sev info">info</span> No strong signals either way</div><p class="finding-detail">Nothing here pushed the estimate far from the base rate.</p></div>`;
  if (info.length) h += `<div class="section-title">Context</div><div class="findings">${info.map(evidenceItem).join("")}</div>`;
  if (measures.length) h += `<details class="measures"><summary>All measurements (${measures.length})</summary><div class="findings">${measures.map(evidenceItem).join("")}</div></details>`;
  return h;
}

export function metricGrid(pairs) {
  return `<div class="metric-grid">${pairs.map(([k, v]) => `<div class="metric"><div class="metric-label">${esc(k)}</div><div class="metric-value" title="${esc(v)}">${esc(v)}</div></div>`).join("")}</div>`;
}

export function reportText(r, title) {
  const lines = [`GPTTrace — ${title}`, `Verdict: ${r.verdict.label} (${Math.round(r.prob * 100)}% AI, confidence ${r.confidence})`, ""];
  for (const e of r.evidence) if (e.decisive || Math.abs(e.contrib) >= 0.05) lines.push(`${e.contrib >= 0 ? "+" : "−"} ${e.label}: ${e.detail || ""}`);
  lines.push("", "Analysed locally in the browser at gpttrace.com. Evidence, not proof.");
  return lines.join("\n");
}

export function actionsHTML() {
  return `<div class="btn-row result-actions"><button type="button" class="btn ghost sm" data-act="copy">Copy report</button><button type="button" class="btn ghost sm" data-act="json">Download JSON</button></div>`;
}

export function bindActions(root, getReport, getJSON, name) {
  root.querySelector('[data-act="copy"]')?.addEventListener("click", async (e) => {
    try { await navigator.clipboard.writeText(getReport()); e.target.textContent = "Copied"; } catch (_) { e.target.textContent = "Copy failed"; }
    setTimeout(() => (e.target.textContent = "Copy report"), 1500);
  });
  root.querySelector('[data-act="json"]')?.addEventListener("click", () => {
    const blob = new Blob([JSON.stringify(getJSON(), (k, v) => (v instanceof ImageBitmap || k === "bitmap" || k === "image" ? undefined : v), 2)], { type: "application/json" });
    const a = Object.assign(document.createElement("a"), { href: URL.createObjectURL(blob), download: `gpttrace-${name}.json` });
    a.click();
    setTimeout(() => URL.revokeObjectURL(a.href), 1000);
  });
}

export function progressHTML(msg, p) {
  return `<div class="analyzing" role="status"><span class="spinner"></span><span>${esc(msg)}</span>${p != null ? `<progress max="1" value="${p}"></progress>` : ""}</div>`;
}

export function errorHTML(title, msg) {
  return `<div class="finding"><div class="finding-head"><span class="sev high">error</span> ${esc(title)}</div><p class="finding-detail">${esc(msg)}</p></div>`;
}

// File input: click, keyboard, drag-drop and paste, for one dropzone.
export function bindDropzone(dz, input, onFile) {
  dz.addEventListener("click", () => input.click());
  dz.addEventListener("keydown", (e) => { if (e.key === "Enter" || e.key === " ") { e.preventDefault(); input.click(); } });
  ["dragenter", "dragover"].forEach((ev) => dz.addEventListener(ev, (e) => { e.preventDefault(); dz.classList.add("drag"); }));
  ["dragleave", "drop"].forEach((ev) => dz.addEventListener(ev, (e) => { e.preventDefault(); dz.classList.remove("drag"); }));
  dz.addEventListener("drop", (e) => { const f = e.dataTransfer.files && e.dataTransfer.files[0]; if (f) onFile(f); });
  input.addEventListener("change", () => { const f = input.files && input.files[0]; if (f) onFile(f); });
}

export function onPaste(accept, onFile) {
  document.addEventListener("paste", (e) => {
    const t = e.target;
    if (t && (t.tagName === "TEXTAREA" || t.tagName === "INPUT")) return;
    const f = [...(e.clipboardData?.files || [])].find((x) => accept.test(x.type));
    if (f) { e.preventDefault(); onFile(f); }
  });
}

export const pref = {
  get(k, d) { try { const v = localStorage.getItem("gt:" + k); return v == null ? d : JSON.parse(v); } catch (_) { return d; } },
  set(k, v) { try { localStorage.setItem("gt:" + k, JSON.stringify(v)); } catch (_) {} }
};
