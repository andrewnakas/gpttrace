#!/usr/bin/env node
/* Web bundle for the native apps (Capacitor on iOS/Android, Tauri on
 * desktop): the built site minus server-only and private files → apps/web.
 *   node tools/build-app.js        (run npm run build first)
 *
 * Like the extension build, model paths become absolute
 * (https://gpttrace.com/models/…), so the app downloads each model once and
 * caches it instead of shipping 300 MB inside the installer. Analytics are
 * stubbed out: the apps send nothing. Detection runs on the device exactly
 * as on the site. */
import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";

const ROOT = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");
const OUT = path.join(ROOT, "apps", "web");
const ORIGIN = "https://gpttrace.com";
// Never bundled: deploy-ignored paths, models (downloaded on use), the
// service worker (native shells cache on their own) and verification files.
const ignore = fs.readFileSync(path.join(ROOT, ".assetsignore"), "utf8").split("\n").map((s) => s.trim()).filter(Boolean);
const SKIP = new Set([...ignore, "models", "sw.js", "dist", "apps", "api", "migrations", ".dev.vars", ".dev.vars.example", ".git"]);
const SKIP_RX = [/^google[0-9a-f]+\.html$/, /^[0-9a-f]{32}\.txt$/, /^_headers$/, /^\.DS_Store$/, /\.local\.json$/];

fs.rmSync(OUT, { recursive: true, force: true });
let files = 0;
for (const f of fs.readdirSync(ROOT)) {
  if (SKIP.has(f) || SKIP_RX.some((rx) => rx.test(f))) continue;
  fs.cpSync(path.join(ROOT, f), path.join(OUT, f), { recursive: true, filter: (p) => path.basename(p) !== ".DS_Store" });
}
const modelsFile = path.join(OUT, "js/engine/models.js");
let n = 0;
fs.writeFileSync(modelsFile, fs.readFileSync(modelsFile, "utf8").replace(/(["'])\/models\//g, (_, q) => (n++, `${q}${ORIGIN}/models/`)));
if (n < 2) { console.error(`build-app: expected model paths in models.js, rewrote ${n}`); process.exit(1); }
// Native shells map extensionless paths to the root index.html (SPA mode),
// so page links name the file: /ai-text-detector/ → /ai-text-detector/index.html.
const dirLink = (u) => u.replace(/^(\/[^"'`?#.]*\/)(?=$|[?#])/, "$1index.html");
(function rewrite(d) {
  for (const f of fs.readdirSync(d)) {
    const p = path.join(d, f);
    if (fs.statSync(p).isDirectory()) { if (f !== "vendor") rewrite(p); continue; }
    if (f.endsWith(".html") || f.endsWith(".js")) fs.writeFileSync(p, fs.readFileSync(p, "utf8").replace(/(href=")(\/[^"]*)"/g, (_, a, u) => a + dirLink(u) + '"'));
  }
})(OUT);
const mountFile = path.join(OUT, "js/ui/mount.js");
fs.writeFileSync(mountFile, fs.readFileSync(mountFile, "utf8").replace("`/ai-${tool}-detector/`", "`/ai-${tool}-detector/index.html`"));
fs.writeFileSync(path.join(OUT, "js/ga.js"), "/* Analytics are off in the apps. */\n");
(function walk(d) { for (const f of fs.readdirSync(d)) { const p = path.join(d, f); fs.statSync(p).isDirectory() ? walk(p) : files++; } })(OUT);
console.log(`build-app: ${files} files → ${path.relative(ROOT, OUT)}/ (models from ${ORIGIN})`);
