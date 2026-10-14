/* Video engine. Container metadata and C2PA (MP4/MOV boxes), then frames
 * sampled through the clip are run through the image engine's forensics
 * and classifier. Replaces the old analyzeVideo stub. */
import { readBytes, sniffType, latin1 } from "../common/bytes.js";
import { readC2PA } from "../common/c2pa.js";
import { matchGenerators } from "../common/generators.js";
import { combine } from "../common/score.js";
import { c2paEvidence, generatorEvidence } from "../common/provenance.js";
import { tags, header } from "../audio/tags.js";
import { analyzeBitmap } from "../image/index.js";

const clip = (v, lo, hi) => Math.max(lo, Math.min(hi, v));
const median = (a) => { const s = [...a].sort((x, y) => x - y); return s.length ? s[s.length >> 1] : 0; };

async function frames(file, n, onFrame) {
  const url = URL.createObjectURL(file);
  const v = document.createElement("video");
  v.muted = true; v.playsInline = true; v.preload = "auto"; v.src = url;
  try {
    await new Promise((res, rej) => { v.onloadedmetadata = res; v.onerror = () => rej(new Error("This browser cannot decode this video.")); });
    const dur = isFinite(v.duration) ? v.duration : 0;
    const out = [];
    for (let i = 0; i < n; i++) {
      const t = dur ? Math.min(dur - 0.05, (dur * (i + 0.5)) / n) : 0;
      await new Promise((res) => { v.onseeked = res; v.currentTime = Math.max(0, t); setTimeout(res, 3000); });
      const bmp = await createImageBitmap(v);
      out.push({ t, bmp });
      onFrame && onFrame(i, bmp, t);
      if (!dur) break;
    }
    return { frames: out, width: v.videoWidth, height: v.videoHeight, duration: dur };
  } finally { URL.revokeObjectURL(url); }
}

export async function analyze(file, opts = {}) {
  const stage = opts.onStage || (() => {});
  stage("Reading container metadata and Content Credentials…");
  const u = await readBytes(file);
  const type = sniffType(u, file.name);
  const c2pa = readC2PA(u, type);
  const t = tags(u, type);
  const h = header(u, type);
  // Encoder strings sit in udta/ilst or as plain text near the start/end.
  const raw = latin1(u.subarray(0, Math.min(u.length, 2 << 20))) + latin1(u.subarray(Math.max(0, u.length - (1 << 20))));
  const enc = [...raw.matchAll(/(?:Lavf|Lavc|HandBrake|Google|Sora|Veo|Runway|Kling|Pika|Luma|Hailuo|MiniMax|Midjourney|CapCut|Adobe|Apple|iPhone|Android|Samsung|DJI|GoPro)[\x20-\x7e]{0,40}/g)].map((m) => m[0]).slice(0, 20);
  const strong = matchGenerators([...t.list.filter((x) => /too|swr|encoder/i.test(x.key + x.name)).map((x) => x.value), ...enc], "video").concat(c2pa.generators || []);
  const weak = matchGenerators(t.list.map((x) => `${x.name}: ${x.value}`), "video").map((g) => ({ ...g, weak: true }));
  const camera = /iPhone|Apple|Android|Samsung|DJI|GoPro|com\.apple\.quicktime\.(make|model)/.test(raw);

  stage("Sampling frames…");
  const n = opts.frames || 8;
  const fr = await frames(file, n, opts.onFrame);
  const per = [];
  for (let i = 0; i < fr.frames.length; i++) {
    stage(`Checking frame ${i + 1} of ${fr.frames.length}…`);
    per.push(await analyzeBitmap(fr.frames[i].bmp, { deep: opts.deep, onModelProgress: opts.onModelProgress }));
    fr.frames[i].bmp.close();
  }
  const evidence = [...c2paEvidence(c2pa, "video"), ...generatorEvidence(strong, weak, "video")];
  if (camera) evidence.push({ id: "camera-container", group: "metadata", label: "Recorded-on-device metadata", x: -1, w: 1, sev: "info", detail: "The container names a phone or camera maker (QuickTime make/model or encoder tags). Re-encoding usually strips this, so its presence counts for authenticity — though it can be faked." });
  const logits = per.map((p) => p.classifier && !p.classifier.error ? p.classifier.centre : null).filter((v) => v != null);
  if (logits.length) {
    const m = median(logits), mx = Math.max(...logits);
    const pr = (z) => Math.round(100 / (1 + Math.exp(-z))) + "%";
    evidence.push({ id: "frame-classifier", group: "model", label: "Neural classifier on sampled frames", w: 0.5, x: clip(m, -8, 8),
      detail: `Median ${pr(m)} across ${logits.length} frames (highest ${pr(mx)}): ${logits.map(pr).join(", ")}. The model was trained on still images, so video compression lowers its confidence.` });
  }
  const lat = per.map((p) => p.pixels && p.pixels.spectrum.latticePeak).filter((v) => v != null);
  if (lat.length) evidence.push({ id: "frame-lattice", group: "forensics", label: "Spectral lattice across frames", w: 0.15, x: clip((median(lat) - 1.8) / 0.7, -1, 2), detail: `Median isolated spectral peak ${median(lat).toFixed(2)} across frames.` });
  const scored = combine("video", evidence, { weights: opts.weights, lowData: !logits.length, weightsKey: logits.length ? "video" : "image-basic" });
  return { ...scored, type, header: h, c2pa, tags: t.list, encoders: enc, frames: per.map((p, i) => ({ t: fr.frames[i].t, classifier: p.classifier, lattice: p.pixels && p.pixels.spectrum.latticePeak })), width: fr.width, height: fr.height, duration: fr.duration };
}
