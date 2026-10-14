/* Neural audio-deepfake check (WavLM-base-plus fine-tuned on ASVspoof 2019,
 * WaveFake and LJSpeech with codec augmentation; eval/train/audio) on
 * onnxruntime-web WASM. Input: mono 16 kHz samples. Scores up to 4 evenly
 * spaced 4-second windows, each normalised to zero mean / unit variance, and
 * averages the logits — exactly as eval/train/audio/audio_train.py evaluates. */
import * as ort from "../../../vendor/ort-1.30.0/ort.wasm.min.mjs";
import { MODELS } from "../models.js";

const M = MODELS.audio;
const WIN = 64000, WINDOWS = 4;
ort.env.wasm.wasmPaths = new URL("../../../vendor/ort-1.30.0/", import.meta.url).href;
ort.env.wasm.numThreads = self.crossOriginIsolated ? Math.min(4, navigator.hardwareConcurrency || 1) : 1;
let session = null;

// The model from the first mirror that has it. Over the host's per-file
// limit it is split into M.parts files, fetched in turn and joined; the local
// dev mirror serves it whole.
async function fetchModel(progress) {
  const cache = typeof caches !== "undefined" ? await caches.open("gpttrace-models-v1") : null;
  const local = /^(localhost|127\.0\.0\.1)$/.test(self.location.hostname) ? [new URL("/__models/" + M.id + "/", self.location.origin).href] : [];
  // Our host serves the model in M.parts files; the local dev mirror and the
  // Hugging Face mirrors (M.mirrors) serve it whole.
  const mirrors = [...local.map((base) => ({ base, files: M.files, whole: true })), ...M.base.map((base) => ({ base, files: M.files })), ...(M.mirrors || []).map((m) => ({ ...m, whole: true }))];
  for (const m of mirrors) {
    const url = new URL(m.base + m.files.model, self.location.origin).href;
    try {
      const hit = cache && (await cache.match(url));
      if (hit) return new Uint8Array(await hit.arrayBuffer());
      const parts = M.parts && !m.whole ? Array.from({ length: M.parts }, (_, i) => url + ".part" + i) : [url];
      const chunks = []; let got = 0;
      for (const u of parts) {
        const res = await fetch(u);
        if (!res.ok) throw new Error(res.status);
        const reader = res.body.getReader();
        for (;;) { const { done, value } = await reader.read(); if (done) break; chunks.push(value); got += value.length; progress(Math.min(1, got / M.bytes)); }
      }
      if (got !== M.bytes) throw new Error(`model size ${got} ≠ ${M.bytes}`);
      const buf = new Uint8Array(got); let o = 0;
      for (const c of chunks) { buf.set(c, o); o += c.length; }
      if (cache) await cache.put(url, new Response(buf)).catch(() => {});
      return buf;
    } catch (e) { /* next mirror */ }
  }
  throw new Error("could not download the audio model");
}

function windows(x) {
  const n = Math.max(1, Math.min(WINDOWS, Math.floor(x.length / WIN)));
  // np.linspace(0, len - WIN, n).astype(int), as in training (n = 1 → start 0).
  const starts = x.length <= WIN ? [0] : Array.from({ length: n }, (_, i) => (n === 1 ? 0 : Math.floor((i * (x.length - WIN)) / (n - 1))));
  return starts.map((s) => {
    const w = new Float32Array(WIN);
    const seg = x.subarray(s, s + WIN);
    // Short clips are repeat-padded, as in training (np.tile).
    for (let i = 0; i < WIN; i++) w[i] = seg.length ? seg[i % seg.length] : 0;
    let mean = 0; for (let i = 0; i < WIN; i++) mean += w[i]; mean /= WIN;
    let v = 0; for (let i = 0; i < WIN; i++) v += (w[i] - mean) ** 2;
    const sd = Math.sqrt(v / WIN) + 1e-7;
    for (let i = 0; i < WIN; i++) w[i] = (w[i] - mean) / sd;
    return w;
  });
}

self.onmessage = async (e) => {
  const { id, samples } = e.data;
  try {
    session ||= await ort.InferenceSession.create(await fetchModel((p) => self.postMessage({ type: "progress", p })), { executionProviders: ["wasm"], graphOptimizationLevel: "all" });
    const logits = [];
    for (const w of windows(samples)) {
      const out = await session.run({ [session.inputNames[0]]: new ort.Tensor("float32", w, [1, WIN]) });
      const [h, a] = out[session.outputNames[0]].data;
      logits.push(a - h);
    }
    self.postMessage({ id, ok: true, result: { logit: logits.reduce((s, v) => s + v, 0) / logits.length, windows: logits.length, model: M.id } });
  } catch (err) {
    self.postMessage({ id, ok: false, error: String((err && err.message) || err) });
  }
};
