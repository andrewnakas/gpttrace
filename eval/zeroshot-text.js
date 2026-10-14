/* Zero-shot text statistics from a small causal language model, for the
 * hybrid detector experiment (workstream A3). One forward pass over the text
 * gives, for every token, the model's full next-token distribution. From it:
 *   ll    mean log-likelihood of the actual tokens (−log-perplexity)
 *   ent   mean entropy of the predicted distribution
 *   bino  −ll / ent: Binoculars-style ratio with one model as observer and
 *         performer (Hans et al. 2024 use two; one keeps it browser-sized)
 *   fdg   Fast-DetectGPT analytic curvature (Bao et al. 2024) with the same
 *         model for sampling and scoring: (Σ ll − Σ E[ll]) / sqrt(Σ Var[ll])
 *   rank  mean log-rank of the actual tokens (approximate: counts logits
 *         above the actual one)
 * These hold up better than a fine-tuned classifier on paraphrased text and
 * unseen generators, which is where ours is weakest. Runs on the vendored ORT
 * Web WASM build, so the numbers match what a browser would compute.
 *
 *   import { zeroshotFeatures, SMOL135 } from "./zeroshot-text.js";
 *   const feats = await zeroshotFeatures(rows, SMOL135);   // { [id]: {ll, ent, bino, fdg, rank, n} }
 * Results are cached per model in ~/.cache/gpttrace/eval/neural/<name>.json. */
import fs from "node:fs";
import os from "node:os";
import path from "node:path";
import { Tokenizer } from "../vendor/tokenizers-0.2.0/tokenizers.min.mjs";
import { CACHE } from "./manifest.js";
import { normalizeText } from "../js/engine/text/normalize.js";

const HF = path.join(os.homedir(), ".cache", "gpttrace", "hf");
export const SMOL135 = { name: "zs-smollm2-135m-q8", dir: path.join(HF, "onnx-community/SmolLM2-135M-ONNX"), file: "onnx/model_quantized.onnx", maxTokens: 512 };
export const SPECS = { smol135: SMOL135 };

export async function loadLM(spec) {
  const ort = await import("../vendor/ort-1.30.0/ort.wasm.min.mjs");
  ort.env.wasm.wasmPaths = new URL("../vendor/ort-1.30.0/", import.meta.url).pathname;
  ort.env.wasm.numThreads = +process.env.THREADS || 4;
  const cfg = JSON.parse(fs.readFileSync(path.join(spec.dir, "config.json"), "utf8"));
  const tj = JSON.parse(fs.readFileSync(path.join(spec.dir, "tokenizer.json"), "utf8"));
  const tc = JSON.parse(fs.readFileSync(path.join(spec.dir, "tokenizer_config.json"), "utf8"));
  const tokenizer = new Tokenizer(tj, tc);
  const model = path.join(spec.dir, spec.file);
  const opts = { executionProviders: ["wasm"], graphOptimizationLevel: "all" };
  if (fs.existsSync(model + "_data")) opts.externalData = [{ path: path.basename(model) + "_data", data: fs.readFileSync(model + "_data") }];
  const session = await ort.InferenceSession.create(fs.readFileSync(model), opts);
  return { ort, tokenizer, session, spec, cfg };
}

export async function featuresFor(m, text) {
  const { ort, tokenizer, session, spec, cfg } = m;
  let ids = tokenizer.encode(normalizeText(text)).ids;
  if (cfg.bos_token_id != null && ids[0] !== cfg.bos_token_id) ids.unshift(cfg.bos_token_id);
  ids = ids.slice(0, spec.maxTokens);
  const L = ids.length;
  if (L < 3) return null;
  const heads = cfg.num_key_value_heads, dim = cfg.head_dim || cfg.hidden_size / cfg.num_attention_heads;
  const feeds = {};
  for (const name of session.inputNames) {
    if (name === "input_ids") feeds[name] = new ort.Tensor("int64", BigInt64Array.from(ids, BigInt), [1, L]);
    else if (name === "attention_mask") feeds[name] = new ort.Tensor("int64", BigInt64Array.from({ length: L }, () => 1n), [1, L]);
    else if (name === "position_ids") feeds[name] = new ort.Tensor("int64", BigInt64Array.from({ length: L }, (_, i) => BigInt(i)), [1, L]);
    else if (name.startsWith("past_key_values")) feeds[name] = new ort.Tensor("float32", new Float32Array(0), [1, heads, 0, dim]);
    else throw new Error("unexpected LM input " + name);
  }
  const out = await session.run(feeds);
  const logits = out.logits.data, V = out.logits.dims[2];
  let sLL = 0, sMean = 0, sVar = 0, sEnt = 0, sRank = 0;
  for (let t = 0; t < L - 1; t++) {
    const off = t * V, next = ids[t + 1];
    let max = -Infinity;
    for (let v = 0; v < V; v++) if (logits[off + v] > max) max = logits[off + v];
    let z = 0;
    for (let v = 0; v < V; v++) z += Math.exp(logits[off + v] - max);
    const lz = Math.log(z) + max;
    let mean = 0, sq = 0, above = 0;
    const actual = logits[off + next];
    for (let v = 0; v < V; v++) {
      const lp = logits[off + v] - lz, p = Math.exp(lp);
      mean += p * lp; sq += p * lp * lp;
      if (logits[off + v] > actual) above++;
    }
    const ll = actual - lz;
    sLL += ll; sMean += mean; sVar += sq - mean * mean; sEnt -= mean; sRank += Math.log(1 + above);
  }
  const n = L - 1;
  const r = (x) => +x.toFixed(4);
  return { ll: r(sLL / n), ent: r(sEnt / n), bino: r(-sLL / sEnt), fdg: r((sLL - sMean) / Math.sqrt(Math.max(sVar, 1e-9))), rank: r(sRank / n), n };
}

// Returns { [row.id]: features } for every row, scoring only uncached rows.
export async function zeroshotFeatures(rows, spec = SMOL135) {
  const file = path.join(CACHE, "neural", spec.name + ".json");
  fs.mkdirSync(path.dirname(file), { recursive: true });
  const cache = fs.existsSync(file) ? JSON.parse(fs.readFileSync(file, "utf8")) : {};
  const todo = rows.filter((r) => !(r.id in cache));
  if (todo.length) {
    const m = await loadLM(spec);
    const t0 = Date.now();
    for (let i = 0; i < todo.length; i++) {
      cache[todo[i].id] = await featuresFor(m, todo[i].text);
      if ((i + 1) % 100 === 0 || i === todo.length - 1) {
        fs.writeFileSync(file, JSON.stringify(cache));
        process.stderr.write(`\r${spec.name}: ${i + 1}/${todo.length} (${((Date.now() - t0) / (i + 1)).toFixed(0)} ms/row)`);
      }
    }
    process.stderr.write("\n");
  }
  return cache;
}
