#!/usr/bin/env node
/* Browser extension build: extension/ (manifest, service worker, result page,
 * icons) plus the site's own engine, findings renderer, stylesheet and
 * vendored runtimes → extension/dist, zipped to dist/gpttrace-extension.zip.
 *   node tools/build-extension.js                   Chrome and Edge (one package)
 *   node tools/build-extension.js --target=firefox  → extension/dist-firefox,
 *                                                     dist/gpttrace-extension-firefox.zip
 *   (npm run build:extension builds both)
 *
 * Firefox gets the same code with a different manifest: an event page
 * (background.scripts) instead of a service worker, an add-on id, and the
 * data-collection declaration AMO requires (none). Safari is built from the
 * Chrome package with xcrun safari-web-extension-converter (apps/safari).
 *
 * The engine is copied, not forked, so verdicts match the site's. The one
 * change: model paths in js/engine/models.js are site-relative ("/models/…"),
 * which inside an extension would point at the extension itself, so they are
 * made absolute (https://gpttrace.com/models/…). MV3 forbids remotely hosted
 * code, so the build also fails if any script imports from a URL or a
 * relative import doesn't resolve inside the package. */
import fs from "node:fs";
import path from "node:path";
import { execFileSync } from "node:child_process";
import { fileURLToPath } from "node:url";

const ROOT = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");
const SRC = path.join(ROOT, "extension");
const TARGET = (process.argv.find((a) => a.startsWith("--target=")) || "--target=chrome").slice(9);
if (!["chrome", "firefox"].includes(TARGET)) { console.error(`build-extension: unknown target ${TARGET}`); process.exit(1); }
const OUT = path.join(SRC, TARGET === "chrome" ? "dist" : "dist-" + TARGET);
const ZIP = path.join(ROOT, "dist", TARGET === "chrome" ? "gpttrace-extension.zip" : `gpttrace-extension-${TARGET}.zip`);
const ORIGIN = "https://gpttrace.com";

// [from (repo), to (package)]. Only the vendored libraries the text and image
// engines load; the site's other tool UIs are not needed.
const COPY = [
  ["extension/manifest.json", "manifest.json"],
  ["extension/background.js", "background.js"],
  ["extension/result.html", "result.html"],
  ["extension/result.js", "result.js"],
  ["extension/result.css", "result.css"],
  ["extension/icons", "icons"],
  ["js/engine", "js/engine"],
  ["js/ui/findings.js", "js/ui/findings.js"],
  ["css/styles.css", "css/styles.css"],
  ["vendor/ort-1.30.0", "vendor/ort-1.30.0"],
  ["vendor/tokenizers-0.2.0", "vendor/tokenizers-0.2.0"],
  ["vendor/exifr-7.1.3", "vendor/exifr-7.1.3"]
];

const fail = (msg) => { console.error("build-extension: " + msg); process.exit(1); };

fs.rmSync(OUT, { recursive: true, force: true });
for (const [from, to] of COPY) {
  const src = path.join(ROOT, from);
  if (!fs.existsSync(src)) fail(`missing ${from}`);
  fs.cpSync(src, path.join(OUT, to), { recursive: true, filter: (f) => path.basename(f) !== ".DS_Store" });
}

// Site-relative model paths → absolute. Every "/models/" string in models.js
// is a model location; there must be at least the image and text ones.
const modelsFile = path.join(OUT, "js/engine/models.js");
let n = 0;
const models = fs.readFileSync(modelsFile, "utf8").replace(/(["'])\/models\//g, (_, q) => (n++, `${q}${ORIGIN}/models/`));
if (n < 2) fail(`expected model paths in js/engine/models.js, rewrote ${n}`);
fs.writeFileSync(modelsFile, models);

// No remote code, and every relative import resolves inside the package.
const files = [];
(function walk(d) { for (const f of fs.readdirSync(d)) { const full = path.join(d, f); fs.statSync(full).isDirectory() ? walk(full) : files.push(full); } })(OUT);
const IMPORT = /(?:\bimport\s*(?:[\w*{}\s,]+from\s*)?|\bimport\s*\(\s*|new\s+URL\s*\(\s*)(["'`])([^"'`]+)\1/g;
for (const f of files.filter((x) => /\.(js|mjs)$/.test(x) && !x.includes(`${path.sep}vendor${path.sep}`))) {
  for (const [, , spec] of fs.readFileSync(f, "utf8").matchAll(IMPORT)) {
    if (/^(https?:)?\/\//.test(spec)) fail(`${path.relative(OUT, f)} loads remote code: ${spec}`);
    if (spec.startsWith(".") && !fs.existsSync(path.resolve(path.dirname(f), spec))) fail(`${path.relative(OUT, f)}: ${spec} is not in the package`);
  }
}
for (const f of files.filter((x) => x.endsWith(".html")))
  for (const [, url] of fs.readFileSync(f, "utf8").matchAll(/<(?:script|link)[^>]+(?:src|href)="([^"]+)"/g))
    if (/^(https?:)?\/\//.test(url)) fail(`${path.relative(OUT, f)} loads remote code: ${url}`);
    else if (!fs.existsSync(path.resolve(path.dirname(f), url))) fail(`${path.relative(OUT, f)}: ${url} is not in the package`);
const manifest = JSON.parse(fs.readFileSync(path.join(OUT, "manifest.json"), "utf8"));
if (TARGET === "firefox") {
  delete manifest.minimum_chrome_version;
  manifest.background = { scripts: ["background.js"] };
  manifest.browser_specific_settings = {
    gecko: { id: "extension@gpttrace.com", strict_min_version: "140.0", data_collection_permissions: { required: ["none"] } }
  };
  fs.writeFileSync(path.join(OUT, "manifest.json"), JSON.stringify(manifest, null, 2) + "\n");
}
for (const p of Object.values(manifest.icons)) if (!fs.existsSync(path.join(OUT, p))) fail(`icon ${p} missing (node tools/build-og.js renders them)`);

fs.mkdirSync(path.dirname(ZIP), { recursive: true });
fs.rmSync(ZIP, { force: true });
execFileSync("zip", ["-q", "-r", "-X", ZIP, "."], { cwd: OUT });
const bytes = files.reduce((s, f) => s + fs.statSync(f).size, 0);
console.log(`extension ${manifest.version}: ${files.length} files, ${(bytes / 1048576).toFixed(1)} MB → ${path.relative(ROOT, OUT)}/, ${(fs.statSync(ZIP).size / 1048576).toFixed(1)} MB → ${path.relative(ROOT, ZIP)}`);
