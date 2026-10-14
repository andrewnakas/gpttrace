/* Video tool: file → video engine → frame strip + findings. */
import { esc, verdictHTML, evidenceHTML, metricGrid, actionsHTML, bindActions, reportText, progressHTML, errorHTML, bindDropzone, pref } from "./findings.js";
import { MODELS, mb } from "../engine/models.js";

let engine = null;
const loadEngine = () => (engine ||= import("../engine/video/index.js"));

export function mount(root) {
  const $ = (r) => root.querySelector(`[data-role="${r}"]`);
  const dz = $("dropzone"), input = $("file"), result = $("result"), preview = $("preview"), player = $("player"), strip = $("frames"), deep = $("deep");
  if (!result.dataset.empty) result.dataset.empty = result.innerHTML;
  deep.checked = pref.get("deep-image", true);
  deep.addEventListener("change", () => pref.set("deep-image", deep.checked));
  let url = null;
  bindDropzone(dz, input, handle);
  $("clear").addEventListener("click", () => { preview.hidden = true; dz.hidden = false; input.value = ""; strip.innerHTML = ""; result.innerHTML = result.dataset.empty; if (url) URL.revokeObjectURL(url); });

  async function handle(file) {
    if (/^(image|audio)\//.test(file.type)) { document.dispatchEvent(new CustomEvent("gt:route", { detail: file })); return; }
    dz.hidden = true; preview.hidden = false; strip.innerHTML = "";
    if (url) URL.revokeObjectURL(url);
    url = URL.createObjectURL(file); player.src = url;
    result.innerHTML = progressHTML("Loading the analyser…");
    const figs = [];
    try {
      const { analyze } = await loadEngine();
      const r = await analyze(file, {
        deep: deep.checked,
        onStage: (m) => (result.innerHTML = progressHTML(m)),
        onModelProgress: (p) => (result.innerHTML = progressHTML(`Downloading the neural model (${mb(MODELS.image.bytes)}, once)…`, p)),
        onFrame: (i, bmp, t) => {
          const f = document.createElement("figure"), c = document.createElement("canvas"), cap = document.createElement("figcaption");
          c.width = 160; c.height = Math.round((160 * bmp.height) / bmp.width) || 90;
          c.getContext("2d").drawImage(bmp, 0, 0, c.width, c.height);
          cap.textContent = t.toFixed(1) + "s";
          f.append(c, cap); strip.append(f); figs.push(cap);
        }
      });
      r.frames.forEach((fr, i) => { if (figs[i] && fr.classifier && !fr.classifier.error) figs[i].textContent = `${fr.t.toFixed(1)}s · ${Math.round(100 / (1 + Math.exp(-fr.classifier.centre)))}%`; });
      let out = verdictHTML(r, `${esc(file.name)} · ${r.width}×${r.height}${r.duration ? " · " + r.duration.toFixed(1) + " s" : ""}`) + evidenceHTML(r);
      out += `<div class="section-title">Container</div>` + metricGrid([["Format", (r.type || "").toUpperCase()], ["Frames checked", r.frames.length], ["Encoder strings", r.encoders.slice(0, 3).join(" · ") || "—"]]);
      out += actionsHTML() + `<p class="note">Evidence, not proof. Frame checks are weaker on video than on photos because compression smears the pixel traces; Content Credentials and visible watermarks are the most reliable signs for Sora, Veo and similar tools.</p>`;
      result.innerHTML = out;
      bindActions(result, () => reportText(r, `video analysis of ${file.name}`), () => r, "video");
    } catch (e) {
      result.innerHTML = errorHTML("Could not analyse this video", e.message);
    }
  }
  return { handle };
}
