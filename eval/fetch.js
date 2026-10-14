#!/usr/bin/env node
/* Downloads every eval source in manifest.js into ~/.cache/gpttrace/eval.
 *   node eval/fetch.js [text|image|audio]   (resumable; skips files present) */
import fs from "node:fs";
import path from "node:path";
import { execFileSync } from "node:child_process";
import { SOURCES, CACHE } from "./manifest.js";

const only = process.argv[2];
for (const [mod, list] of Object.entries(SOURCES)) {
  if (only && only !== mod) continue;
  for (const s of list) {
    for (const [rel, url] of Object.entries(s.files || {})) {
      const dest = path.join(CACHE, rel);
      if (fs.existsSync(dest)) continue;
      fs.mkdirSync(path.dirname(dest), { recursive: true });
      console.log("fetch", rel);
      execFileSync("curl", ["-sL", "--fail", "-C", "-", "--retry", "4", "-o", dest + ".part", url]);
      fs.renameSync(dest + ".part", dest);
    }
    if (s.fetch) await s.fetch();
  }
}
console.log("eval data in", CACHE);
