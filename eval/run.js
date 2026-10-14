#!/usr/bin/env node
/* Runs the real engines over the labelled eval sets.
 *
 *   node eval/run.js text [--quick] [--save] [--deep]
 *   node eval/run.js image|audio [--quick] [--save]    (headless Chrome)
 *   node eval/run.js <modality> --bench=<name>         (a public benchmark
 *     sample from eval/bench/; results go to eval/results/bench/ only)
 *
 * --deep adds the neural text check (eval/neural-text.js, same WASM
 * runtime as the browser) and scores with the text-deep weights.
 *
 * Writes eval/results/<modality>.features.json (feature vectors for
 * fit-weights.js) and, with --save, eval/results/<modality>.json plus the
 * per-page numbers in content/data/eval-summary.json. */
import fs from "node:fs";
import path from "node:path";
import { loadAll, loadBench } from "./manifest.js";
import { summary } from "./metrics.js";
import { featureVector } from "../js/engine/common/score.js";

const ROOT = path.join(path.dirname(new URL(import.meta.url).pathname), "..");
const mod = process.argv[2] || "text";
const QUICK = process.argv.includes("--quick");
const SAVE = process.argv.includes("--save");
const DEEP = process.argv.includes("--deep");
const BENCH = (process.argv.find((a) => a.startsWith("--bench=")) || "").slice(8);
const TAG = (BENCH ? `bench/${mod}-${BENCH}` : mod) + (DEEP ? "-deep" + (process.env.NEURAL ? "-" + process.env.NEURAL : "") : "");

function sample(rows, n) {
  // Deterministic, stratified: every k-th row of each label.
  const by = { ai: rows.filter((r) => r.label === "ai"), human: rows.filter((r) => r.label === "human") };
  return [...by.ai.filter((_, i) => i % Math.ceil(by.ai.length / (n / 2)) === 0), ...by.human.filter((_, i) => i % Math.ceil(by.human.length / (n / 2)) === 0)];
}

async function runText(rows) {
  const { analyze } = await import("../js/engine/text/index.js");
  const { combine } = await import("../js/engine/common/score.js");
  const nt = DEEP && (await import("./neural-text.js"));
  const logits = DEEP ? await nt.neuralLogits(rows, nt.SPECS[process.env.NEURAL || "ettin3"]) : {};
  const out = [];
  for (const r of rows) {
    let res = await analyze(r.text);
    if (DEEP && res.stats.wordCount >= 30) {
      // Same evidence item and weights key as js/engine/text/index.js.
      const evidence = [...res.evidence, { id: "neural-text", x: +Math.max(-8, Math.min(8, logits[r.id])).toFixed(3) }];
      res = { ...res, ...combine("text", evidence, { lowData: res.stats.wordCount < 80, weightsKey: "text-deep" }), evidence };
    }
    out.push({ id: r.id, label: r.label, slice: r.slice, prob: res.prob, words: res.stats.wordCount, features: featureVector(res.evidence) });
  }
  return out;
}

async function runBrowser(rows) {
  const { runInChrome } = await import("./browser.js");
  return runInChrome(mod, rows, DEEP ? { deep: true } : {});
}

let rows = BENCH ? loadBench(mod, BENCH) : await loadAll(mod);
if (!rows.length) { console.error(BENCH ? `no rows for ${mod} benchmark ${BENCH} — see eval/bench/` : `no ${mod} eval data — run: node eval/fetch.js ${mod}`); process.exit(1); }
if (QUICK) rows = sample(rows, 50);
const t0 = Date.now();
const results = mod === "text" ? await runText(rows) : await runBrowser(rows);
const s = summary(results);
s.seconds = +((Date.now() - t0) / 1000).toFixed(1);
console.log(JSON.stringify({ ...s, slices: undefined }, null, 1));
console.table(s.slices);

fs.mkdirSync(path.join(ROOT, "eval/results/bench"), { recursive: true });
fs.writeFileSync(path.join(ROOT, `eval/results/${TAG}.features.local.json`), JSON.stringify(results));
if (BENCH && !QUICK) {
  fs.writeFileSync(path.join(ROOT, `eval/results/${TAG}.json`), JSON.stringify({ date: new Date().toISOString().slice(0, 10), ...s }, null, 1) + "\n");
  console.log("saved", `eval/results/${TAG}.json`);
} else if (SAVE && !QUICK) {
  fs.writeFileSync(path.join(ROOT, `eval/results/${mod}.json`), JSON.stringify({ date: new Date().toISOString().slice(0, 10), ...s }, null, 1) + "\n");
  const sumPath = path.join(ROOT, "content/data/eval-summary.json");
  const all = fs.existsSync(sumPath) ? JSON.parse(fs.readFileSync(sumPath, "utf8")) : {};
  all[mod] = { date: new Date().toISOString().slice(0, 10), n: s.n, nAI: s.nAI, nHuman: s.nHuman, auc: s.auc, atLikelyAI: s.atLikelyAI, slices: s.slices };
  fs.mkdirSync(path.dirname(sumPath), { recursive: true });
  fs.writeFileSync(sumPath, JSON.stringify(all, null, 1) + "\n");
  console.log("saved", sumPath);
}
