/* Audio tool: file → audio engine → spectrogram + findings. */
import { esc, verdictHTML, evidenceHTML, metricGrid, actionsHTML, bindActions, reportText, progressHTML, errorHTML, bindDropzone, pref } from "./findings.js";
import { MODELS, mb } from "../engine/models.js";
import { bindPro } from "./pro-toggle.js";

let engine = null;
const loadEngine = () => (engine ||= import("../engine/audio/index.js"));

export function mount(root) {
  const $ = (r) => root.querySelector(`[data-role="${r}"]`);
  const dz = $("dropzone"), input = $("file"), result = $("result"), preview = $("preview"), player = $("player"), spec = $("spectrogram");
  if (!result.dataset.empty) result.dataset.empty = result.innerHTML;
  let url = null, last = null;
  const deep = $("deep");
  if (deep) {
    deep.checked = pref.get("deep-audio", false);
    deep.closest("label").querySelector("[data-role=deep-size]").textContent = mb(MODELS.audio.bytes);
    deep.addEventListener("change", () => { pref.set("deep-audio", deep.checked); if (last && !preview.hidden) handle(last); });
  }
  const pro = bindPro(root);
  bindDropzone(dz, input, handle);
  dz.addEventListener("pointerenter", loadEngine, { once: true });
  $("clear").addEventListener("click", () => { preview.hidden = true; dz.hidden = false; input.value = ""; result.innerHTML = result.dataset.empty; if (url) URL.revokeObjectURL(url); });

  let seq = 0;
  async function handle(file) {
    const my = ++seq; // a newer file supersedes this one
    last = file;
    if (/^(image|video)\//.test(file.type)) { document.dispatchEvent(new CustomEvent("gt:route", { detail: file })); return; }
    dz.hidden = true; preview.hidden = false;
    if (url) URL.revokeObjectURL(url);
    url = URL.createObjectURL(file); player.src = url;
    spec.getContext("2d").clearRect(0, 0, spec.width, spec.height);
    result.innerHTML = progressHTML("Loading the analyser…");
    try {
      const { analyze } = await loadEngine();
      const r = await analyze(file, {
        onStage: (m) => (result.innerHTML = progressHTML(m)),
        deep: deep && deep.checked,
        pro: pro.on,
        onModelProgress: (p) => (result.innerHTML = progressHTML(`Downloading the voice model… ${Math.round(p * 100)}% (once; it stays on your device)`))
      });
      if (my !== seq) return;
      if (r.signal) {
        const im = r.signal.spectrogram;
        spec.width = im.width; spec.height = im.height;
        spec.getContext("2d").putImageData(new ImageData(new Uint8ClampedArray(im.data), im.width, im.height), 0, 0);
      }
      render(r, file);
    } catch (e) {
      result.innerHTML = errorHTML("Could not analyse this file", e.message);
    }
  }

  function render(r, file) {
    const h = r.header, s = r.signal;
    let out = verdictHTML(r, `${esc(file.name)}${r.duration ? " · " + r.duration.toFixed(1) + " s" : ""}`);
    if (r.decodeError) out += `<p class="note warn">${esc(r.decodeError)} Only the tags were checked.</p>`;
    if (s && !r.isSpeech) out += `<p class="note">This doesn’t sound like a single speaking voice, so the speech checks (pitch, pauses, noise floor) were skipped. For songs, the band-limit and metadata checks still apply.</p>`;
    if (deep && !deep.checked && r.isSpeech) out += `<p class="note">Turn on <strong>Neural voice-clone check</strong> for a speech model trained on real and synthetic voices (on the In-the-Wild deepfake benchmark it reaches a 5.8% equal error rate).</p>`;
    out += evidenceHTML(r);
    const fmt = [["Format", [h.codec, h.format].filter(Boolean).join(" ").toUpperCase() || "—"], ["Sample rate", h.sampleRate ? h.sampleRate / 1000 + " kHz" : "—"], ["Channels", h.channels || "—"], ["Bitrate", h.bitrate ? h.bitrate + " kbps" : "—"]];
    if (h.encoder) fmt.push(["Encoder", h.encoder]);
    if (s) fmt.push(["Bandwidth", (s.bandwidth.edgeHz / 1000).toFixed(1) + " kHz"], ["Noise floor", s.silence.noiseFloorDb + " dBFS"]);
    if (s && s.pitch.f0Median) fmt.push(["Median pitch", s.pitch.f0Median + " Hz"]);
    out += `<div class="section-title">File &amp; signal</div>` + metricGrid(fmt);
    if (r.tags.length) out += `<details class="measures"><summary>Tags (${r.tags.length})</summary>` + metricGrid(r.tags.slice(0, 40).map((t) => [t.name, String(t.value).slice(0, 120)])) + `</details>`;
    out += actionsHTML() + `<p class="note">Evidence, not proof. Phone calls, voice notes and heavy noise reduction can look synthetic, and the best voice clones can pass these checks. If a call asks for money or secrets, verify through another channel no matter what any detector says.</p>`;
    result.innerHTML = out;
    bindActions(result, () => reportText(r, `audio analysis of ${file.name}`), () => ({ ...r, signal: s && { ...s, spectrogram: undefined, pitch: { ...s.pitch, track: undefined } } }), "audio");
  }
  return { handle };
}
