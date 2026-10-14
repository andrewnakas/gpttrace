#!/usr/bin/env node
/* Export the open-source part of GPTTrace into a separate folder, ready to be
 * the public GitHub repo. This private repo stays the source of truth; the
 * export is regenerated, never edited by hand.
 *   node tools/export-oss.js <dir>      (default: ../gpttrace-oss)
 *
 * Public: the detection engines and tool UIs (js/), vendored runtimes,
 * the browser extension, the app shells (apps/), the evaluation and training
 * code (eval/) and the tools those need. Private: site pages and content,
 * marketing, generated HTML, search-engine verification files, deploy config.
 * The export keeps the target's .git, so it can be committed on top. */
import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { execFileSync } from "node:child_process";

const ROOT = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");
const DEST = path.resolve(process.argv[2] || path.join(ROOT, "..", "gpttrace-oss"));

const ALLOW = [
  "js", "css/styles.css", "vendor", "extension", "apps", "eval",
  "tools/build-extension.js", "tools/build-app.js", "tools/chrome-harness.js", "tools/serve.js", "tools/export-oss.js",
  "manifest.webmanifest", "sw.js", "_headers", "assets/icon.svg", "assets/icon-180.png", "assets/icon-512.png",
  ["tools/oss/README.md", "README.md"], ["tools/oss/LICENSE", "LICENSE"], ["tools/oss/gitignore", ".gitignore"],
  ["tools/oss/package.json", "package.json"]
];
// Never exported, even inside an allowed folder.
const DENY = [/(^|\/)\.DS_Store$/, /(^|\/)dist(-[a-z]+)?(\/|$)/, /(^|\/)xcuserdata(\/|$)/, /(^|\/)__pycache__(\/|$)/,
  /(^|\/)node_modules(\/|$)/, /\.local\.json$/, /^extension\/STORE\.md$/, /^extension\/store(\/|$)/, /^js\/ga\.js$/];

if (DEST === ROOT || DEST.startsWith(ROOT + path.sep)) { console.error("export-oss: destination must be outside the repo"); process.exit(1); }
fs.mkdirSync(DEST, { recursive: true });
for (const f of fs.readdirSync(DEST)) if (f !== ".git") fs.rmSync(path.join(DEST, f), { recursive: true, force: true });

// Only committed files: build output (apps/web, Rust targets, Capacitor's
// copies of the site) is git-ignored and never exported.
const TRACKED = new Set(execFileSync("git", ["ls-files"], { cwd: ROOT, encoding: "utf8" }).split("\n").filter(Boolean));
let files = 0, bytes = 0;
const copy = (rel, to) => {
  if (DENY.some((rx) => rx.test(rel))) return;
  const src = path.join(ROOT, rel);
  const st = fs.statSync(src);
  if (st.isDirectory()) return fs.readdirSync(src).forEach((f) => copy(path.join(rel, f), path.join(to, f)));
  if (!TRACKED.has(rel) && !rel.startsWith("tools/oss/")) return;
  fs.mkdirSync(path.dirname(path.join(DEST, to)), { recursive: true });
  fs.copyFileSync(src, path.join(DEST, to));
  files++; bytes += st.size;
};
for (const a of ALLOW) {
  const [from, to] = Array.isArray(a) ? a : [a, a];
  if (!fs.existsSync(path.join(ROOT, from))) { console.error(`export-oss: missing ${from}`); process.exit(1); }
  copy(from, to);
}
console.log(`export-oss: ${files} files, ${(bytes / 1048576).toFixed(1)} MB → ${DEST}`);
