/* Text engine: analyze(text) → { prob, score, verdict, confidence, evidence,
 * stats, signs }. Pure (no DOM); runs in the page, a worker or Node. */
import { statistics, statEvidence } from "./stats.js";
import { runSigns } from "./signs.js";
import { compressionEvidence } from "./compression.js";
import { combine } from "../common/score.js";
import { workerRPC } from "../common/worker-rpc.js";

let clsW = null;
const classifier = () => (clsW ||= workerRPC(new URL("./classifier.worker.js", import.meta.url)));

export { highlight } from "./highlight.js";

export async function analyze(text, opts = {}) {
  text = String(text || "");
  const stats = statistics(text);
  const ctx = { words: stats.wordCount, sentences: stats.sentCount };
  const signs = runSigns(text, ctx);
  const evidence = signs.map((s) => ({
    id: s.id, group: s.group, label: s.label, sev: s.sev, wiki: s.wiki, desc: s.desc, count: s.count, x: s.x,
    list: [...new Set(s.matches.map((m) => m.value.trim().slice(0, 80)))].slice(0, 8),
    detail: `${s.count} match${s.count === 1 ? "" : "es"}. ${s.desc}`
  }));
  evidence.push(...statEvidence(stats));
  if (opts.compression !== false) {
    const c = await compressionEvidence(text, stats.wordCount);
    if (c) evidence.push(c);
  }
  let neural = null;
  if (opts.deep && stats.wordCount >= 30) {
    try {
      neural = await classifier().call({ text }, [], opts.onModelProgress);
      const p = 1 / (1 + Math.exp(-neural.logit));
      evidence.push({ id: "neural-text", group: "model", label: "Neural text classifier", x: +Math.max(-8, Math.min(8, neural.logit)).toFixed(3),
        detail: `An Ettin encoder fine-tuned on the RAID and MAGE benchmarks gives ${(p * 100).toFixed(1)}%${neural.truncated ? " on the first 512 tokens" : ""}. It is weighed together with the other signals, which keeps it from over-flagging technical or non-native human writing.` });
    } catch (e) {
      evidence.push({ id: "neural-text-error", group: "model", label: "Neural text classifier unavailable", x: 0, w: 0, sev: "info", detail: e.message });
    }
  }
  // Pro cloud model: only when the visitor switched it on for this check.
  let key = neural ? "text-deep" : "text";
  if (opts.pro && stats.wordCount >= 30) {
    const { cloudEvidence, proWeightsKey } = await import("../common/cloud.js");
    const row = await cloudEvidence("text", { text }, "text-deep", "neural-text");
    evidence.push(row);
    if (row.id === "cloud-text") key = proWeightsKey("text", "text-deep");
  }
  const result = combine("text", evidence, { lowData: stats.wordCount < 80, weights: opts.weights, weightsKey: key });
  return { ...result, stats, signs };
}
