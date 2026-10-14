/* Node mirror of js/engine/text/classifier.worker.js, so the deep text check
 * can be evaluated without a browser. It loads the same vendored ORT Web WASM
 * build as the site: native onnxruntime-node uses different int8 kernels and
 * drifts by up to 0.3 logit (see eval/parity-text.js). Tokenisation,
 * truncation and the logit are computed exactly as in the worker. Results
 * are cached per model in ~/.cache/gpttrace/eval/neural/, keyed by row id,
 * so re-runs only score new rows.
 *
 *   import { neuralLogits } from "./neural-text.js";
 *   const logits = await neuralLogits(rows, { dir, file, maxTokens, name }); */
import fs from "node:fs";
import os from "node:os";
import path from "node:path";
import { Tokenizer } from "../vendor/tokenizers-0.2.0/tokenizers.min.mjs";
import { CACHE } from "./manifest.js";
import { normalizeText } from "../js/engine/text/normalize.js";

const HF = path.join(os.homedir(), ".cache", "gpttrace", "hf");

// The previous deep-scan model (TMR RoBERTa), kept for comparison.
export const TMR = { name: "tmr-q8", dir: path.join(HF, "onnx-community/tmr-ai-text-detector-ONNX"), file: "onnx/model_quantized.onnx", maxTokens: 512 };
// Ettin-150M fine-tunes (eval/train/text): v2 = RAID only (leaderboard PR),
// v3 = RAID + MAGE, the shipped deep-scan model (js/engine/models.js).
export const ETTIN = { name: "ettin150-v2-q8", dir: path.join(HF, "gpttrace/ettin150-raid-v2/onnx"), file: "model_quantized.onnx", maxTokens: 512 };
export const ETTIN3 = { ...ETTIN, name: "ettin150-v3-q8", dir: path.join(HF, "gpttrace/ettin150-raid-v3/onnx") };
// NEURAL=tmr|ettin|ettin3 selects the model in eval/run.js --deep (default: shipped).
export const SPECS = { tmr: TMR, ettin: ETTIN, ettin3: ETTIN3 };

export async function loadModel(spec) {
  const ort = await import("../vendor/ort-1.30.0/ort.wasm.min.mjs");
  ort.env.wasm.wasmPaths = new URL("../vendor/ort-1.30.0/", import.meta.url).pathname;
  ort.env.wasm.numThreads = +process.env.THREADS || 4;
  const tj = JSON.parse(fs.readFileSync(path.join(spec.dir, "tokenizer.json"), "utf8"));
  const tc = JSON.parse(fs.readFileSync(path.join(spec.dir, "tokenizer_config.json"), "utf8"));
  const tokenizer = new Tokenizer(tj, tc);
  const session = await ort.InferenceSession.create(fs.readFileSync(path.join(spec.dir, spec.file)), { executionProviders: ["wasm"], graphOptimizationLevel: "all" });
  return { ort, tokenizer, session, spec };
}

export async function logitFor(m, text) {
  const { ort, tokenizer, session, spec } = m;
  let ids = tokenizer.encode(normalizeText(text)).ids;
  if (ids.length > spec.maxTokens) ids = [...ids.slice(0, spec.maxTokens - 1), ids[ids.length - 1]]; // keep </s>
  const L = ids.length;
  const feeds = {};
  for (const name of session.inputNames) {
    if (/mask/.test(name)) feeds[name] = new ort.Tensor("int64", BigInt64Array.from({ length: L }, () => 1n), [1, L]);
    else if (/type/.test(name)) feeds[name] = new ort.Tensor("int64", new BigInt64Array(L), [1, L]);
    else feeds[name] = new ort.Tensor("int64", BigInt64Array.from(ids, (v) => BigInt(v)), [1, L]);
  }
  const out = await session.run(feeds);
  const d = out[session.outputNames[0]].data;
  // Two-class heads give [human, ai]; single-logit heads give the AI logit.
  return d.length >= 2 ? d[spec.aiIndex ?? 1] - d[spec.aiIndex === 0 ? 1 : 0] : d[0];
}

// Returns { [row.id]: logit } for every row, scoring only uncached rows.
export async function neuralLogits(rows, spec = ETTIN3) {
  const file = path.join(CACHE, "neural", spec.name + ".json");
  fs.mkdirSync(path.dirname(file), { recursive: true });
  const cache = fs.existsSync(file) ? JSON.parse(fs.readFileSync(file, "utf8")) : {};
  const todo = rows.filter((r) => cache[r.id] == null);
  if (todo.length) {
    const m = await loadModel(spec);
    const t0 = Date.now();
    for (let i = 0; i < todo.length; i++) {
      cache[todo[i].id] = +(await logitFor(m, todo[i].text)).toFixed(4);
      if ((i + 1) % 200 === 0 || i === todo.length - 1) {
        fs.writeFileSync(file, JSON.stringify(cache));
        process.stderr.write(`\r${spec.name}: ${i + 1}/${todo.length} (${((Date.now() - t0) / (i + 1)).toFixed(0)} ms/row)`);
      }
    }
    process.stderr.write("\n");
  }
  return cache;
}
