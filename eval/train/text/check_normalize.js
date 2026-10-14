#!/usr/bin/env node
/* Checks js/engine/text/normalize.js and eval/train/text/normalize.py agree
 * on attacked, clean and non-Latin texts (plus the local text eval set). */
import { execFileSync } from "node:child_process";
import { normalizeText } from "../../../js/engine/text/normalize.js";
import { loadAll } from "../../manifest.js";

const cases = [
  "Thе quiсk​ brоwn fox­ jumps⁠ over﻿",
  "Москва — столица России, крупнейший город. Оса и сор.",
  "Hello Мир, the capital is Москва and the weather is good today. Сарас",
  "Ηello ΑΒC", "Αθήνα είναι η πρωτεύουσα της Ελλάδας", "café naïve 東京 emoji 👍🏽 ❤️",
  ...(await loadAll("text")).slice(0, 300).map((r) => r.text)
];
const py = JSON.parse(execFileSync("python3", ["-c", "import json,sys; sys.path.insert(0, sys.argv[1]); from normalize import normalize_text; print(json.dumps([normalize_text(t) for t in json.load(sys.stdin)]))", new URL(".", import.meta.url).pathname], { input: JSON.stringify(cases), maxBuffer: 1 << 28 }).toString());
const bad = cases.filter((t, i) => normalizeText(t) !== py[i]);
bad.slice(0, 5).forEach((t) => console.log("DIFF:", JSON.stringify(t.slice(0, 120)), "\n js:", JSON.stringify(normalizeText(t).slice(0, 120)), "\n py:", JSON.stringify(py[cases.indexOf(t)].slice(0, 120))));
console.log(`${cases.length - bad.length}/${cases.length} agree`);
process.exit(bad.length ? 1 : 0);
