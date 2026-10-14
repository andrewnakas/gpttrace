/* Image eval sources from Hugging Face datasets, fetched through the
 * datasets-server rows API. That API serves every image as a JPEG, real
 * and AI alike, so the two classes share one re-encoding — the classifier
 * can't win by spotting "PNG = AI". It also mirrors how images reach people:
 * re-compressed on the web. Original-format sources are tagged "original".
 */
import fs from "node:fs";
import path from "node:path";
import { CACHE } from "./manifest.js";

export const IMAGE_SOURCES = [
  // AI
  { id: "nano-banana", dataset: "bitmind/nano-banana", label: "ai", slice: "ai:gemini-nano-banana" },
  { id: "midjourney-v6", dataset: "brivangl/midjourney-v6-llava", label: "ai", slice: "ai:midjourney-v6" },
  { id: "journeydb", dataset: "bitmind/JourneyDB", label: "ai", slice: "ai:midjourney-v5" },
  { id: "flux-dev", dataset: "LukasT9/Flux-1-Dev-Images-1k", label: "ai", slice: "ai:flux-dev" },
  { id: "flux-schnell", dataset: "LukasT9/Flux-1-Schnell-Images-1k", label: "ai", slice: "ai:flux-schnell" },
  { id: "sdxl-dogs", dataset: "Voxel51/SDXL-Generated-Stanford-Dogs", label: "ai", slice: "ai:sdxl" },
  { id: "gpt-image", dataset: "neonforestmist/GPT_Watercolor_Anime_Style_Images", label: "ai", slice: "ai:gpt-image" },
  { id: "klingai", dataset: "bitmind/klingai-images", label: "ai", slice: "ai:kling" },
  { id: "leonardo", dataset: "bitmind/Deepfake-leonardo-stablecog", label: "ai", slice: "ai:leonardo-stablecog" },
  { id: "bm-imagine", dataset: "bitmind/bm-imagine", label: "ai", slice: "ai:bitmind-imagine-mix" },
  // Real
  { id: "real-fullsize", dataset: "bitmind/bm-subnet-real-fullsize", label: "human", slice: "human:fullsize-photos" },
  { id: "open-images-sub", dataset: "bitmind/open-images-v7-subset", label: "human", slice: "human:open-images-photos" },
  { id: "lfw", dataset: "bitmind/lfw", label: "human", slice: "human:lfw-faces" },
  { id: "caltech", dataset: "bitmind/caltech-256", label: "human", slice: "human:caltech-objects" },
  { id: "coco", dataset: "bitmind/MS-COCO", label: "human", slice: "human:coco-photos" },
  { id: "ffhq", dataset: "bitmind/ffhq", label: "human", slice: "human:ffhq-faces" },
  { id: "celeba-hq", dataset: "bitmind/celeb-a-hq", label: "human", slice: "human:celeba-faces" }
];

const API = "https://datasets-server.huggingface.co";
const get = async (u) => { const r = await fetch(u); if (!r.ok) throw new Error(`${r.status} ${u}`); return r.json(); };

export async function fetchImageSource(src, n = 40) {
  const dir = path.join(CACHE, "image", src.id);
  const index = path.join(dir, "index.json");
  if (fs.existsSync(index) && JSON.parse(fs.readFileSync(index, "utf8")).length >= n) return;
  fs.mkdirSync(dir, { recursive: true });
  const splits = (await get(`${API}/splits?dataset=${encodeURIComponent(src.dataset)}`)).splits;
  const sp = splits.find((s) => s.split === "test") || splits.find((s) => s.split === "train") || splits[0];
  let total = 2000;
  try {
    const info = await get(`${API}/size?dataset=${encodeURIComponent(src.dataset)}`);
    const c = info.size.splits.find((s) => s.config === sp.config && s.split === sp.split);
    if (c) total = c.num_rows;
  } catch (e) { /* use default */ }
  const out = [];
  const step = Math.max(10, Math.floor(total / Math.ceil(n / 5)));
  for (let off = 0; out.length < n && off < total; off += step) {
    let rows;
    try { rows = (await get(`${API}/rows?dataset=${encodeURIComponent(src.dataset)}&config=${encodeURIComponent(sp.config)}&split=${sp.split}&offset=${off}&length=5`)).rows; }
    catch (e) { console.warn("  rows failed at", off, e.message); continue; }
    for (const r of rows) {
      if (out.length >= n) break;
      const imgCol = Object.values(r.row).find((v) => v && typeof v === "object" && v.src);
      if (!imgCol) continue;
      const file = path.join(dir, `${r.row_idx}.jpg`);
      if (!fs.existsSync(file)) {
        const res = await fetch(imgCol.src);
        if (!res.ok) continue;
        fs.writeFileSync(file, Buffer.from(await res.arrayBuffer()));
      }
      out.push({ id: `${src.id}-${r.row_idx}`, file: path.relative(CACHE, file), label: src.label, slice: src.slice, w: imgCol.width, h: imgCol.height });
    }
    process.stdout.write(`\r  ${src.id}: ${out.length}/${n}`);
  }
  process.stdout.write("\n");
  fs.writeFileSync(index, JSON.stringify(out));
}

export function loadImageRows() {
  return IMAGE_SOURCES.flatMap((s) => {
    const index = path.join(CACHE, "image", s.id, "index.json");
    return fs.existsSync(index) ? JSON.parse(fs.readFileSync(index, "utf8")) : [];
  });
}
