#!/usr/bin/env node
/* Checks that eval/neural-text.js (onnxruntime-node) gives the same logits
 * as the browser worker (ORT Web WASM) on a sample of eval rows.
 *   node eval/parity-text.js [n] */
import { withPage } from "../tools/chrome-harness.js";
import { loadAll } from "./manifest.js";
import { SPECS, loadModel, logitFor } from "./neural-text.js";

const n = +process.argv[2] || 20;
const all = await loadAll("text");
const rows = all.filter((_, i) => i % Math.ceil(all.length / n) === 0);
const HARNESS = `<!doctype html><meta charset="utf-8"><script type="module">
  import { workerRPC } from "/js/engine/common/worker-rpc.js";
  const w = workerRPC(new URL("/js/engine/text/classifier.worker.js", location.href));
  window.__logit = (text) => w.call({ text }, [], () => {}).then((r) => r.logit);
  window.__ready = true;
</script>`;
// The model js/engine/models.js ships (NEURAL selects another for comparison).
const m = await loadModel(SPECS[process.env.NEURAL || "ettin3"]);
let worst = 0;
await withPage({ routes: { "/__parity": HARNESS } }, async (page) => {
  await page.goto("/__parity");
  for (const r of rows) {
    const b = await page.eval(`window.__logit(${JSON.stringify(r.text)})`, 300000);
    const nd = await logitFor(m, r.text);
    worst = Math.max(worst, Math.abs(b - nd));
    console.log(r.id.padEnd(28), b.toFixed(4), nd.toFixed(4));
  }
  if (page.logs.length) console.warn(page.logs.slice(0, 5));
});
console.log("max |Δlogit|", worst.toFixed(5), worst < 0.01 ? "OK" : "MISMATCH");
process.exit(worst < 0.01 ? 0 : 1);
