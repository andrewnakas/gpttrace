/* Runs the image / audio engines over eval rows in headless Chrome (real
 * canvas, WebAudio, workers and WASM). Files are served from the eval cache
 * under /__eval/. */
import fs from "node:fs";
import path from "node:path";
import { withPage } from "../tools/chrome-harness.js";
import { CACHE } from "./manifest.js";

const HARNESS = `<!doctype html><meta charset="utf-8"><title>eval</title><script type="module">
  window.__run = async (mod, url, name, type, opts) => {
    const engine = await import("/js/engine/" + mod + "/index.js");
    const { featureVector } = await import("/js/engine/common/score.js");
    const blob = await (await fetch(url)).blob();
    const file = new File([blob], name, { type: type || blob.type });
    const r = await engine.analyze(file, opts || {});
    if (r.bitmap) r.bitmap.close();
    const extra = mod === "image" ? { classifier: r.classifier, dims: r.dims } : {};
    return { prob: r.prob, features: featureVector(r.evidence), priors: Object.fromEntries(r.evidence.map(e => [e.id, e.w ?? null])), ...extra };
  };
  window.__ready = true;
</script>`;

export async function runInChrome(mod, rows, opts = {}) {
  const routes = { "/__eval": HARNESS };
  const out = [];
  await withPage({ routes, port: 0, profile: path.join(CACHE, ".chrome-profile"), headers: true }, async (page) => {
    // Serve cached eval files.
    page.listen("Fetch.requestPaused", (p) => {
      const m = /\/__eval\/(.+)$/.exec(new URL(p.request.url).pathname);
      if (!m) return page.send("Fetch.continueRequest", { requestId: p.requestId });
      const file = path.join(CACHE, decodeURIComponent(m[1]));
      if (!fs.existsSync(file)) return page.send("Fetch.fulfillRequest", { requestId: p.requestId, responseCode: 404 });
      page.send("Fetch.fulfillRequest", { requestId: p.requestId, responseCode: 200, responseHeaders: [{ name: "content-type", value: "application/octet-stream" }], body: fs.readFileSync(file).toString("base64") });
    });
    await page.send("Fetch.enable", { patterns: [{ urlPattern: "*/__eval/*" }] });
    await page.goto("/__eval");
    let i = 0;
    const t0 = Date.now();
    for (const r of rows) {
      try {
        const res = await page.eval(`window.__run(${JSON.stringify(mod)}, "/__eval/${encodeURI(r.file)}", ${JSON.stringify(path.basename(r.file))}, ${JSON.stringify(r.type || "")}, ${JSON.stringify(opts)})`, 180000);
        out.push({ id: r.id, label: r.label, slice: r.slice, ...res });
      } catch (e) {
        console.warn("\n  failed", r.id, String(e.message).slice(0, 200));
      }
      if (++i % 10 === 0) process.stdout.write(`\r  ${i}/${rows.length} (${((Date.now() - t0) / i).toFixed(0)} ms each)`);
    }
    process.stdout.write("\n");
    if (page.logs.length) console.warn("page errors:", page.logs.slice(0, 5));
  });
  return out;
}
