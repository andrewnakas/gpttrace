/* Optional neural text check (Ettin-150M fine-tuned on RAID and MAGE; see
 * js/engine/models.js) on onnxruntime-web WASM. Off by default: the model is
 * 151 MB. Its output is one feature among the others (see /methodology/). */
import * as ort from "../../../vendor/ort-1.30.0/ort.wasm.min.mjs";
import { Tokenizer } from "../../../vendor/tokenizers-0.2.0/tokenizers.min.mjs";
import { MODELS } from "../models.js";
import { normalizeText } from "./normalize.js";

const M = MODELS.text;
ort.env.wasm.wasmPaths = new URL("../../../vendor/ort-1.30.0/", import.meta.url).href;
ort.env.wasm.numThreads = self.crossOriginIsolated ? Math.min(4, navigator.hardwareConcurrency || 1) : 1;
let state = null;

// One file (a key of M.files) from the first mirror that has it. On our own
// host the model is split into M.parts files (over the per-file limit) that
// are fetched part by part and joined; the local dev mirror and the Hugging
// Face mirrors (M.mirrors) serve it whole.
async function get(key, progress, binary) {
  const cache = typeof caches !== "undefined" ? await caches.open("gpttrace-models-v1") : null;
  let lastErr;
  // On a local dev/test server, try a local mirror first (tools/serve.js and
  // the test harness serve ~/.cache/gpttrace/hf at /__models/).
  const local = /^(localhost|127\.0\.0\.1)$/.test(self.location.hostname) ? [new URL("/__models/" + M.id + "/", self.location.origin).href] : [];
  const mirrors = [...local.map((base) => ({ base, files: M.files, whole: true })), ...M.base.map((base) => ({ base, files: M.files })), ...(M.mirrors || []).map((m) => ({ ...m, whole: true }))];
  for (const m of mirrors) {
    const url = new URL(m.base + m.files[key], self.location.origin).href;
    try {
      const hit = cache && (await cache.match(url));
      if (hit) return binary ? new Uint8Array(await hit.arrayBuffer()) : hit.json();
      const parts = binary && M.parts && !m.whole ? Array.from({ length: M.parts }, (_, i) => url + ".part" + i) : [url];
      const chunks = [];
      let got = 0;
      for (const u of parts) {
        const res = await fetch(u);
        if (!res.ok) throw new Error(res.status + " " + u);
        if (binary && res.body) {
          const reader = res.body.getReader();
          for (;;) { const { done, value } = await reader.read(); if (done) break; chunks.push(value); got += value.length; progress(Math.min(1, got / M.bytes)); }
        } else { const b = new Uint8Array(await res.arrayBuffer()); chunks.push(b); got += b.length; }
      }
      const buf = new Uint8Array(got); let o = 0;
      for (const c of chunks) { buf.set(c, o); o += c.length; }
      if (binary && M.bytes && got !== M.bytes) throw new Error(`model size ${got} ≠ ${M.bytes}`);
      if (cache) await cache.put(url, new Response(buf)).catch(() => {});
      return binary ? buf : JSON.parse(new TextDecoder().decode(buf));
    } catch (e) { lastErr = e; }
  }
  throw lastErr || new Error("model download failed");
}

async function load(progress) {
  if (state) return state;
  const [tj, tc] = await Promise.all([get("tokenizer", () => {}), get("config", () => {})]);
  const tokenizer = new Tokenizer(tj, tc);
  const session = await ort.InferenceSession.create(await get("model", progress, true), { executionProviders: ["wasm"], graphOptimizationLevel: "all" });
  return (state = { tokenizer, session });
}

self.onmessage = async (e) => {
  const { id, text } = e.data;
  try {
    const { tokenizer, session } = await load((p) => self.postMessage({ type: "progress", p }));
    let ids = tokenizer.encode(normalizeText(text)).ids;
    if (ids.length > M.maxTokens) ids = [...ids.slice(0, M.maxTokens - 1), ids[ids.length - 1]]; // keep </s>
    const L = ids.length;
    const feeds = {};
    for (const name of session.inputNames) {
      if (/mask/.test(name)) feeds[name] = new ort.Tensor("int64", BigInt64Array.from({ length: L }, () => 1n), [1, L]);
      else if (/type/.test(name)) feeds[name] = new ort.Tensor("int64", new BigInt64Array(L), [1, L]);
      else feeds[name] = new ort.Tensor("int64", BigInt64Array.from(ids, (v) => BigInt(v)), [1, L]);
    }
    const out = await session.run(feeds);
    const [h, a] = out[session.outputNames[0]].data;
    self.postMessage({ id, ok: true, result: { logit: a - h, tokens: L, truncated: L >= M.maxTokens } });
  } catch (err) {
    self.postMessage({ id, ok: false, error: String((err && err.message) || err) });
  }
};
