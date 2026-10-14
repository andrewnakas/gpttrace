/* Labelled evaluation sources. Media is downloaded to ~/.cache/gpttrace/eval
 * by eval/fetch.js and never committed (licences vary). Each loader returns
 * [{ id, label: "ai"|"human", slice, text | file }]. */
import fs from "node:fs";
import os from "node:os";
import path from "node:path";

export const CACHE = path.join(os.homedir(), ".cache", "gpttrace", "eval");
const HF = "https://huggingface.co/datasets";

export const SOURCES = {
  text: [
    { name: "textsight-2026", license: "CC BY-SA 4.0 / public domain",
      files: {
        "text/ts_ai_generated_2026-09-29.jsonl": `${HF}/textsightai/ai-text-detection-benchmark-2026/resolve/main/data/ai_generated_2026-09-29.jsonl`,
        "text/ts_human_literature_publicdomain.jsonl": `${HF}/textsightai/ai-text-detection-benchmark-2026/resolve/main/data/human_literature_publicdomain.jsonl`,
        "text/ts_human_pmc_ccby_nonanglophone.jsonl": `${HF}/textsightai/ai-text-detection-benchmark-2026/resolve/main/data/human_pmc_ccby_nonanglophone.jsonl`,
        "text/ts_human_wikipedia_pre2022.jsonl": `${HF}/textsightai/ai-text-detection-benchmark-2026/resolve/main/data/human_wikipedia_pre2022.jsonl`
      },
      load() {
        const out = [];
        for (const f of Object.keys(this.files)) {
          for (const line of readLines(f)) {
            const d = JSON.parse(line);
            const slice = d.label === "ai" ? "ai:" + (d.source || "unknown") : d.esl === "True" ? "human:esl-academic" : "human:" + d.source;
            out.push({ id: d.id, label: d.label, slice, text: d.text });
          }
        }
        return out;
      } },
    { name: "hc3", license: "CC BY-SA 4.0",
      files: {
        "text/hc3_open_qa.jsonl": `${HF}/Hello-SimpleAI/HC3/resolve/main/open_qa.jsonl`,
        "text/hc3_wiki_csai.jsonl": `${HF}/Hello-SimpleAI/HC3/resolve/main/wiki_csai.jsonl`
      },
      load() {
        const out = [];
        for (const f of Object.keys(this.files)) {
          const tag = f.includes("open_qa") ? "open_qa" : "wiki_csai";
          readLines(f).forEach((line, i) => {
            const d = JSON.parse(line);
            const h = [].concat(d.human_answers || [])[0], a = [].concat(d.chatgpt_answers || [])[0];
            // Short answers say nothing; keep the ones a person would paste in.
            if (h && h.split(/\s+/).length >= 120) out.push({ id: `hc3-${tag}-${i}-h`, label: "human", slice: "human:hc3-" + tag, text: h });
            if (a && a.split(/\s+/).length >= 120) out.push({ id: `hc3-${tag}-${i}-a`, label: "ai", slice: "ai:chatgpt-3.5", text: a });
          });
        }
        return out;
      } },
    { name: "gpttrace-local", license: "ours", files: {},
      // Text we wrote or generated ourselves, committed under eval/local/.
      load() {
        const dir = new URL("./local/text/", import.meta.url).pathname;
        if (!fs.existsSync(dir)) return [];
        return fs.readdirSync(dir).filter((f) => f.endsWith(".jsonl")).flatMap((f) =>
          fs.readFileSync(path.join(dir, f), "utf8").split("\n").filter(Boolean).map((l) => JSON.parse(l)));
      } }
  ],
  image: [
    { name: "hf-datasets-server", license: "per dataset (see eval/hf-images.js)", files: {},
      async fetch() { const m = await import("./hf-images.js"); for (const s of m.IMAGE_SOURCES) { try { await m.fetchImageSource(s, +process.env.EVAL_N || 40); } catch (e) { console.warn("skip", s.id, e.message); } } },
      load() { return [] } }
  ]
};

function readLines(rel) {
  const p = path.join(CACHE, rel);
  if (!fs.existsSync(p)) return [];
  return fs.readFileSync(p, "utf8").split("\n").filter(Boolean);
}

// A public benchmark sample written by eval/bench/*: rows in
// bench/<modality>/<name>/index.json (files) or index.jsonl (text).
export function loadBench(modality, name) {
  const dir = path.join(CACHE, "bench", modality, name);
  const jsonl = path.join(dir, "index.jsonl");
  if (fs.existsSync(jsonl)) return fs.readFileSync(jsonl, "utf8").split("\n").filter(Boolean).map((l) => JSON.parse(l));
  const json = path.join(dir, "index.json");
  return fs.existsSync(json) ? JSON.parse(fs.readFileSync(json, "utf8")) : [];
}

export async function loadAll(modality) {
  if (modality === "image") return (await import("./hf-images.js")).loadImageRows();
  return SOURCES[modality].flatMap((s) => s.load());
}
