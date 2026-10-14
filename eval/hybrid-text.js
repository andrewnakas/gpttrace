#!/usr/bin/env node
/* Hybrid text detector experiment (workstream A3): does adding zero-shot
 * small-LM statistics (eval/zeroshot-text.js) to the fine-tuned Ettin logit
 * help, especially on paraphrased text and on human text the classifier
 * wrongly flags? Writes one row per text with every feature, for analysis
 * with eval/hybrid-fit.py.
 *   node eval/hybrid-text.js [inhouse] [mage-ood-para] [mage-ood-gpt4] [ghostbuster] ...
 *   → ~/.cache/gpttrace/eval/hybrid/<set>.jsonl  {id, set, label, slice, words, ettin, ll, ent, bino, fdg, rank}
 * LIMIT=n caps rows per set (stratified), for quick looks. */
import fs from "node:fs";
import path from "node:path";
import { CACHE, loadAll, loadBench } from "./manifest.js";
import { neuralLogits, ETTIN3 } from "./neural-text.js";
import { zeroshotFeatures, SPECS } from "./zeroshot-text.js";

const sets = process.argv.slice(2).length ? process.argv.slice(2) : ["inhouse", "mage-ood-para", "mage-ood-gpt4"];
const LIMIT = +process.env.LIMIT || 0;
const LM = SPECS[process.env.LM || "smol135"];
const OUT = path.join(CACHE, "hybrid");
fs.mkdirSync(OUT, { recursive: true });

function cap(rows) {
  if (!LIMIT || rows.length <= LIMIT) return rows;
  const by = { ai: rows.filter((r) => r.label === "ai"), human: rows.filter((r) => r.label === "human") };
  const take = (a, n) => a.filter((_, i) => i % Math.ceil(a.length / n) === 0);
  return [...take(by.ai, LIMIT / 2), ...take(by.human, LIMIT / 2)];
}

for (const set of sets) {
  const rows = cap(set === "inhouse" ? await loadAll("text") : loadBench("text", set));
  console.log(`${set}: ${rows.length} rows`);
  const ettin = await neuralLogits(rows, ETTIN3);
  const zs = await zeroshotFeatures(rows, LM);
  const lines = rows.filter((r) => zs[r.id]).map((r) => JSON.stringify({
    id: r.id, set, label: r.label, slice: r.slice || r.source || "", words: (r.text.match(/\S+/g) || []).length,
    ettin: ettin[r.id], ...zs[r.id]
  }));
  fs.writeFileSync(path.join(OUT, `${set}${LM === SPECS.smol135 ? "" : "-" + LM.name}.jsonl`), lines.join("\n") + "\n");
  console.log(`  → ${lines.length} rows written`);
}
