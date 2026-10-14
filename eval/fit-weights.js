#!/usr/bin/env node
/* Fits the scoring weights for one modality from eval/run.js output.
 *
 *   node eval/fit-weights.js text [--write]
 *
 * L2-regularised logistic regression on the evidence features, with every
 * row weighted so each data slice counts equally (otherwise one big source
 * dominates). Reports 5-fold cross-validated AUC, then moves the bias so at
 * most 5% of human rows reach the "Likely AI" line (P ≥ 0.7): a false
 * accusation costs more than a miss. --write updates js/engine/weights.js. */
import fs from "node:fs";
import path from "node:path";
import { summary } from "./metrics.js";
import { WEIGHTS } from "../js/engine/weights.js";

const ROOT = path.join(path.dirname(new URL(import.meta.url).pathname), "..");
const mod = process.argv[2] || "text";
const WRITE = process.argv.includes("--write");
const LAMBDA = +(process.argv.find((a) => a.startsWith("--l2=")) || "--l2=0.02").slice(5);
const MAX_FPR = 0.05, LIKELY = 0.7;
const KEY = process.env.KEY || mod; // weights entry to write (e.g. text-deep)
// FIT=a,b: only these weights are learned; the rest stay at their priors.
// Used for images, where the eval files have no camera metadata and the AI
// sets come at native generator sizes — fitting those features would learn
// how the datasets were built, not how AI images look in the wild.
const FIT = process.env.FIT ? new Set(process.env.FIT.split(",")) : null;

// SKIP=a,b: leave out slices whose name contains a or b — e.g. a slice that
// overlaps a model's training data, which would inflate the fitted numbers.
const SKIP = process.env.SKIP ? process.env.SKIP.split(",") : [];
const rows = JSON.parse(fs.readFileSync(path.join(ROOT, `eval/results/${process.env.FEAT || mod}.features.local.json`), "utf8"))
  .filter((r) => !SKIP.some((s) => r.slice.includes(s)));
const keys = [...new Set(rows.flatMap((r) => Object.keys(r.features)))].sort();
const X = rows.map((r) => keys.map((k) => r.features[k] || 0));
const y = rows.map((r) => (r.label === "ai" ? 1 : 0));
const sliceN = {};
rows.forEach((r) => { sliceN[r.slice] = (sliceN[r.slice] || 0) + 1; });
const labelSlices = { 0: new Set(), 1: new Set() };
rows.forEach((r, i) => labelSlices[y[i]].add(r.slice));
// Each slice gets equal total weight within its label; labels get equal weight.
const sw = rows.map((r, i) => 1 / sliceN[r.slice] / labelSlices[y[i]].size);
const sig = (z) => 1 / (1 + Math.exp(-z));
// L2 pulls each weight toward its hand-set prior, not toward zero, so a sign
// that is rare in the data (AI disclaimers, chat markup) keeps its prior
// weight instead of being erased; the data only moves what it can support.
// Image/audio evidence carries its own hand-set prior (`priors`, from e.w).
const rowPrior = (k) => { for (const r of rows) if (r.priors && r.priors[k] != null) return r.priors[k]; return null; };
const PRIOR = keys.map((k) => { const base = WEIGHTS[KEY] || WEIGHTS[mod]; const v = (base.prior || base.w)[k]; return v != null ? v : rowPrior(k) ?? 0; });

function fit(idx) {
  let w = Float64Array.from(PRIOR), b = 0;
  const lr = 0.5;
  let totalW = 0; idx.forEach((i) => { totalW += sw[i]; });
  for (let it = 0; it < 4000; it++) {
    const gw = new Float64Array(keys.length); let gb = 0;
    for (const i of idx) {
      let z = b; for (let j = 0; j < keys.length; j++) z += w[j] * X[i][j];
      const e = (sig(z) - y[i]) * sw[i] / totalW;
      gb += e; for (let j = 0; j < keys.length; j++) gw[j] += e * X[i][j];
    }
    // Every feature is built so that larger = more AI-like. Keep weights ≥ 0:
    // a negative weight would mean the data taught us an artifact of how a
    // dataset was processed (e.g. re-encoding), not a property of AI media.
    for (let j = 0; j < keys.length; j++) if (!FIT || FIT.has(keys[j])) w[j] = Math.max(0, w[j] - lr * (gw[j] + LAMBDA * (w[j] - PRIOR[j])));
    b -= lr * gb;
  }
  return { w, b };
}
const predict = (m, i) => { let z = m.b; for (let j = 0; j < keys.length; j++) z += m.w[j] * X[i][j]; return z; };

// 5-fold CV (deterministic folds by index).
const cv = [];
for (let f = 0; f < 5; f++) {
  const train = [], test = [];
  rows.forEach((_, i) => (i % 5 === f ? test : train).push(i));
  const m = fit(train);
  test.forEach((i) => cv.push({ label: rows[i].label, slice: rows[i].slice, prob: sig(predict(m, i)) }));
}
const cvS = summary(cv);
console.log("5-fold CV AUC", cvS.auc, "TPR@5%FPR", cvS.at5pctFpr.tpr, "TPR@1%FPR", cvS.at1pctFpr.tpr);

const model = fit(rows.map((_, i) => i));
// Calibrate the bias for the false-positive budget at the Likely-AI line.
// CAL=a: hold the budget on the human slices containing a (e.g. the in-the-wild
// set) rather than on all humans pooled, when an easy set would hide a hard one.
const CAL = process.env.CAL || "";
const humanZ = rows.map((r, i) => (y[i] || !r.slice.includes(CAL) ? null : predict(model, i))).filter((z) => z != null).sort((a, b) => b - a);
const target = Math.log(LIKELY / (1 - LIKELY));
const q = humanZ[Math.floor(humanZ.length * MAX_FPR)];
let bias = model.b;
if (q > target) bias -= q - target;
const final = rows.map((r, i) => ({ label: r.label, slice: r.slice, prob: sig(predict(model, i) - model.b + bias) }));
const fs_ = summary(final);
console.log("in-sample AUC", fs_.auc, "at P≥0.7:", fs_.atLikelyAI);
console.table(fs_.slices);
const weights = Object.fromEntries(keys.map((k, j) => [k, +model.w[j].toFixed(3)]));
console.log("bias", bias.toFixed(3), weights);

if (WRITE) {
  const file = path.join(ROOT, "js/engine/weights.js");
  const src = fs.readFileSync(file, "utf8");
  const m = /export const WEIGHTS = (\{[\s\S]*\});\s*$/.exec(src);
  const all = JSON.parse(m[1]);
  const prev = all[KEY] || all[mod];
  all[KEY] = { fitted: true, prior: prev.prior || (Object.keys(prev.w).length ? prev.w : Object.fromEntries(keys.map((k, j) => [k, PRIOR[j]]))), date: new Date().toISOString().slice(0, 10), n: rows.length, cvAuc: cvS.auc, bias: +bias.toFixed(3), w: weights };
  fs.writeFileSync(file, src.slice(0, m.index) + "export const WEIGHTS = " + JSON.stringify(all, null, 2) + ";\n");
  console.log("wrote", file);
}

// --summary: publish the out-of-fold (cross-validated) numbers for this
// weight set in content/data/eval-summary.json under KEY.
if (process.argv.includes("--summary")) {
  // Apply the same bias shift to the CV predictions so the numbers match the shipped threshold.
  const shift = bias - model.b;
  const shifted = cv.map((r) => ({ ...r, prob: sig(Math.log(r.prob / (1 - r.prob)) + shift) }));
  const s = summary(shifted);
  const sumPath = path.join(ROOT, "content/data/eval-summary.json");
  const all = fs.existsSync(sumPath) ? JSON.parse(fs.readFileSync(sumPath, "utf8")) : {};
  all[KEY] = { date: new Date().toISOString().slice(0, 10), crossValidated: true, n: s.n, nAI: s.nAI, nHuman: s.nHuman, auc: s.auc, atLikelyAI: s.atLikelyAI, slices: s.slices };
  fs.writeFileSync(sumPath, JSON.stringify(all, null, 1) + "\n");
  console.log("summary →", sumPath, KEY, s.auc, s.atLikelyAI);
}
