/* Neural AI-image classifier (Community Forensics ViT-S, sigmoid output =
 * P(generated)) on onnxruntime-web WASM. Replaces the old ImageNet/CLIP
 * "AI model" detectors, which were never trained to tell AI from real.
 *
 * The main thread transfers an ImageBitmap; nothing is read from a URL, so
 * the old revoked-blob-URL bug can't happen. */
import * as ort from "../../../vendor/ort-1.30.0/ort.wasm.min.mjs";
import { MODELS } from "../models.js";

const M = MODELS.image;
// Optional second classifier (our fine-tune of M); logits are averaged.
const SPECS = [M, MODELS.image2].filter(Boolean);
ort.env.wasm.wasmPaths = new URL("../../../vendor/ort-1.30.0/", import.meta.url).href;
ort.env.wasm.numThreads = self.crossOriginIsolated ? Math.min(4, navigator.hardwareConcurrency || 1) : 1;

let sessions = null;

async function fetchModel(M, progress) {
  const cache = typeof caches !== "undefined" ? await caches.open("gpttrace-models-v1") : null;
  for (const raw of M.urls) {
    const url = new URL(raw, self.location.origin).href;
    try {
      const hit = cache && (await cache.match(url));
      if (hit) return new Uint8Array(await hit.arrayBuffer());
      const res = await fetch(url);
      if (!res.ok) throw new Error(res.status);
      const total = +res.headers.get("content-length") || M.bytes;
      const reader = res.body.getReader();
      const chunks = []; let got = 0;
      for (;;) {
        const { done, value } = await reader.read();
        if (done) break;
        chunks.push(value); got += value.length;
        progress(Math.min(1, got / total));
      }
      const buf = new Uint8Array(got); let o = 0;
      for (const c of chunks) { buf.set(c, o); o += c.length; }
      if (cache) await cache.put(url, new Response(buf, { headers: { "content-type": "application/octet-stream" } })).catch(() => {});
      return buf;
    } catch (e) { /* try the next mirror */ }
  }
  throw new Error("could not download the model");
}

async function load(progress) {
  if (sessions) return sessions;
  const out = [];
  for (let i = 0; i < SPECS.length; i++) {
    const bytes = await fetchModel(SPECS[i], (p) => progress((i + p) / SPECS.length));
    out.push(await ort.InferenceSession.create(bytes, { executionProviders: ["wasm"], graphOptimizationLevel: "all" }));
  }
  return (sessions = out);
}

function tensorFrom(bmp, sx, sy, sw, sh) {
  const S = M.input.crop;
  const c = new OffscreenCanvas(S, S);
  const x = c.getContext("2d", { willReadFrequently: true });
  x.imageSmoothingQuality = "high";
  x.drawImage(bmp, sx, sy, sw, sh, 0, 0, S, S);
  const d = x.getImageData(0, 0, S, S).data;
  const t = new Float32Array(3 * S * S);
  const [m0, m1, m2] = M.input.mean, [s0, s1, s2] = M.input.std;
  for (let i = 0, p = 0; i < S * S; i++, p += 4) {
    t[i] = (d[p] / 255 - m0) / s0;
    t[S * S + i] = (d[p + 1] / 255 - m1) / s1;
    t[2 * S * S + i] = (d[p + 2] / 255 - m2) / s2;
  }
  return new ort.Tensor("float32", t, [1, 3, S, S]);
}

// Preprocessing as the model card specifies: shortest edge → 440, centre
// crop 384. Expressed as a source rectangle so it is one drawImage.
function centreRect(w, h) {
  const scale = M.input.shortestEdge / Math.min(w, h);
  const side = M.input.crop / scale;
  return [(w - side) / 2, (h - side) / 2, side, side];
}

async function run(bmp) {
  const ss = await load((p) => self.postMessage({ type: "progress", p }));
  const logitOf = async (rect) => {
    const t = tensorFrom(bmp, ...rect);
    let sum = 0;
    for (const s of ss) sum += (await s.run({ [s.inputNames[0]]: t }))[s.outputNames[0]].data[0];
    return sum / ss.length;
  };
  const centre = await logitOf(centreRect(bmp.width, bmp.height));
  // Native-resolution crops (no resampling) — artifacts survive here when a
  // big image is shrunk for the centre view. Reported separately.
  const crops = [];
  const C = M.input.crop;
  if (Math.min(bmp.width, bmp.height) >= C * 2) {
    for (const [fx, fy] of [[0.25, 0.25], [0.75, 0.25], [0.25, 0.75], [0.75, 0.75]])
      crops.push(await logitOf([Math.round(bmp.width * fx - C / 2), Math.round(bmp.height * fy - C / 2), C, C]));
  }
  return { centre, crops, model: SPECS.map((x) => x.id).join("+") };
}

self.onmessage = async (e) => {
  const { id, bitmap, warm } = e.data;
  try {
    if (warm) { await load((p) => self.postMessage({ type: "progress", p })); self.postMessage({ id, ok: true }); return; }
    self.postMessage({ id, ok: true, result: await run(bitmap) });
  } catch (err) {
    self.postMessage({ id, ok: false, error: String((err && err.message) || err) });
  } finally {
    bitmap && bitmap.close && bitmap.close();
  }
};
