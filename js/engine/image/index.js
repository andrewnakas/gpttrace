/* Image engine: analyze(file, opts) → { prob, verdict, evidence, meta,
 * pixels, classifier, views }.
 *
 * Evidence order of strength: provenance (signed C2PA, IPTC source type,
 * generator settings) → camera plausibility → neural classifier → pixel
 * forensics. Provenance settles the verdict; the rest is weighed by
 * score.js. */
import { readBytes, sniffType } from "../common/bytes.js";
import { combine } from "../common/score.js";
import { workerRPC } from "../common/worker-rpc.js";
import { readImageMeta, generatorSize } from "./metadata.js";
import { c2paEvidence, generatorEvidence } from "../common/provenance.js";

const clip = (v, lo, hi) => Math.max(lo, Math.min(hi, v));
let forensicsW = null, classifierW = null;
const forensics = () => (forensicsW ||= workerRPC(new URL("./forensics.worker.js", import.meta.url)));
const classifier = () => (classifierW ||= workerRPC(new URL("./classifier.worker.js", import.meta.url)));

export async function decode(file) {
  try {
    return await createImageBitmap(file, { imageOrientation: "from-image", colorSpaceConversion: "none" });
  } catch (e) {
    // Safari / HEIC: fall back to <img>, which can decode more formats.
    const url = URL.createObjectURL(file);
    try {
      const img = new Image();
      img.src = url;
      await img.decode();
      return await createImageBitmap(img);
    } finally { URL.revokeObjectURL(url); }
  }
}

// Pixel + classifier evidence for one bitmap (also used per video frame).
export async function analyzeBitmap(bitmap, opts = {}) {
  const jobs = [];
  const copy = await createImageBitmap(bitmap);
  jobs.push(forensics().call({ bitmap: copy }, [copy]));
  if (opts.deep !== false) {
    const c2 = await createImageBitmap(bitmap);
    jobs.push(classifier().call({ bitmap: c2 }, [c2], opts.onModelProgress).catch((e) => ({ error: e.message })));
  } else jobs.push(Promise.resolve(null));
  const [px, cls] = await Promise.all(jobs);
  return { pixels: px, classifier: cls };
}

export function pixelEvidence(px, cls, fileSize) {
  const ev = [];
  if (cls && !cls.error) {
    const z = clip(cls.centre, -8, 8);
    const p = 1 / (1 + Math.exp(-cls.centre));
    ev.push({ id: "classifier", group: "model", label: "Neural AI-image classifier", w: 0.6, x: z,
      detail: `Community Forensics ViT (trained on 4,803 generators) gives ${(p * 100).toFixed(1)}% that this image was generated.` +
        (cls.crops.length ? ` Native-resolution crops: ${cls.crops.map((c) => Math.round(100 / (1 + Math.exp(-c))) + "%").join(", ")}.` : "") });
    if (cls.crops.length) {
      const m = Math.max(...cls.crops);
      ev.push({ id: "classifier-crops-max", group: "model", label: "Classifier, strongest native crop", w: 0, measure: true, x: clip(m, -8, 8), detail: `${Math.round(100 / (1 + Math.exp(-m)))}% on the most suspicious full-resolution crop.` });
    }
  } else if (cls && cls.error) {
    ev.push({ id: "classifier-error", group: "model", label: "Neural classifier unavailable", w: 0, x: 0, sev: "info", detail: cls.error });
  }
  if (px) {
    const e = px.ela, s = px.spectrum, p = px.pixels;
    if (fileSize == null || fileSize > 80 * 1024)
      ev.push({ id: "ela-uniform", group: "forensics", label: "Error-level analysis", w: 0.15, x: clip((1.4 - e.mean) / 0.7, -1.5, 1.5),
        detail: `Mean re-save error ${e.mean} (variance ${e.variance}). Uniformly low error across textured areas is common in rendered images; camera JPEGs usually show strong error on edges.` });
    ev.push({ id: "spectral-spike", group: "forensics", label: "Radial spectrum spike", w: 0.2, x: clip((s.spikeRatio - 1.3) / 0.2, -1, 2),
      detail: `Peak-to-mean ratio ${s.spikeRatio} in the 0.5–0.9 frequency band of a ${s.size}×${s.size} native-resolution crop.` });
    ev.push({ id: "lattice-peak", group: "forensics", label: "Periodic lattice in the spectrum", w: 0.2, x: clip((s.latticePeak - 1.5) / 0.7, -1, 2),
      detail: `Strongest isolated spectral peak is ${s.latticePeak} above its ring. Upsampling and decoder layers leave grid-like peaks; camera sensors rarely do.` });
    ev.push({ id: "spectral-slope", group: "forensics", label: "Spectral fall-off", w: 0, measure: true, x: clip((s.slope + 1.0) / 0.3, -2, 2), detail: `Log-magnitude slope ${s.slope} (natural photos fall off roughly as 1/f).` });
    ev.push({ id: "duplicate-pixels", group: "forensics", label: "Duplicate neighbour pixels", w: 0, measure: true, x: clip((p.duplicateRatio - 0.09) / 0.06, -1.5, 2), detail: `${(p.duplicateRatio * 100).toFixed(1)}% of sampled pixel pairs are identical (nearest-neighbour upscaling raises this; so do flat graphics).` });
    ev.push({ id: "checkerboard", group: "forensics", label: "Checkerboard pattern", w: 0, measure: true, x: clip((p.checkerboardRatio - 0.02) / 0.03, -1, 2), detail: `${(p.checkerboardRatio * 100).toFixed(2)}% of 2×2 blocks alternate (a transposed-convolution artifact of GANs).` });
    ev.push({ id: "noise-residual", group: "forensics", label: "Noise residual", w: 0, measure: true, x: clip((4 - p.noise) / 2, -1.5, 1.5), detail: `Mean high-pass residual ${p.noise}. Camera sensors add noise; renders are often cleaner — unless noise was added or the photo was denoised.` });
    ev.push({ id: "texture-entropy", group: "forensics", label: "Luminance entropy", w: 0, measure: true, x: clip((7.2 - p.entropy) / 0.5, -1.5, 1.5), detail: `${p.entropy} bits.` });
    if (px.stealth) ev.push({ id: "stealth-pnginfo", group: "provenance", label: "NovelAI stealth metadata in alpha channel", decisive: "ai", x: 4, w: 1, sev: "high", detail: `The alpha channel's low bits begin with “${px.stealth.slice(0, 15)}”, NovelAI's hidden copy of its generation settings.` });
  }
  return ev;
}

export function metaEvidence(meta, dims) {
  const ev = [];
  ev.push(...c2paEvidence(meta.c2pa, "image"));
  if (meta.sourceType) {
    const s = meta.sourceType;
    if (s.ai && !s.partial && !s.weak) ev.push({ id: "iptc-ai", group: "provenance", label: "IPTC: made with generative AI", decisive: "ai", x: 4, w: 1, sev: "high", detail: `Digital Source Type is ${s.key}: ${s.label}. Google, Meta, Adobe and OpenAI write this field on AI images.` });
    else if (s.ai) ev.push({ id: "iptc-ai-edit", group: "provenance", label: "IPTC: edited with AI", x: 2.5, w: 1, sev: "high", detail: s.label });
    else ev.push({ id: "iptc-source", group: "provenance", label: "IPTC digital source type", x: s.key === "digitalCapture" ? -1 : 0, w: 1, sev: "info", detail: s.label });
  }
  if (meta.sdParams) ev.push({ id: "sd-params", group: "provenance", label: "Generation settings embedded", decisive: "ai", x: 4, w: 1, sev: "high",
    detail: `“${meta.sdParams.key}” (${meta.sdParams.chunk}) holds diffusion settings — prompt, steps, sampler, seed or a node graph.`, list: [meta.sdParams.value.slice(0, 300)] });
  ev.push(...generatorEvidence(meta.generators.filter((g) => !g.weak), meta.generators.filter((g) => g.weak), "image"));
  const cam = meta.camera;
  if (cam.plausible) ev.push({ id: "camera-exif", group: "metadata", label: "Camera metadata present", x: -1.5, w: 1, sev: "info",
    detail: `${cam.present.length} camera fields: ${cam.present.join(", ")}${meta.exif.Make ? ` (${meta.exif.Make} ${meta.exif.Model || ""})` : ""}. Metadata can be copied onto a fake, so this lowers — not removes — suspicion.` });
  else if (cam.present.length) ev.push({ id: "camera-sparse", group: "metadata", label: "Sparse camera metadata", x: 0.3, w: 1, sev: "low", detail: `Only: ${cam.present.join(", ")}.` });
  else ev.push({ id: "camera-none", group: "metadata", label: "No camera metadata", x: 0.6, w: 1, sev: "low",
    detail: "No make, model, exposure or lens fields. Real photos carry these — unless a messaging app, social network or screenshot stripped them, which is very common." });
  if (dims) {
    const g = generatorSize(dims.width, dims.height);
    if (g) ev.push({ id: "generator-size", group: "metadata", label: g === "exact" ? "Typical generator resolution" : "Dimensions are multiples of 64", x: g === "exact" ? 1 : 0.4, w: 0.5, sev: "low",
      detail: `${dims.width}×${dims.height}. ${g === "exact" ? "A native output size of popular image generators." : "Diffusion models work in 64-pixel blocks; camera sensors rarely produce such sizes."}` });
  }
  return ev;
}

export async function analyze(file, opts = {}) {
  const stage = opts.onStage || (() => {});
  stage("Reading metadata and Content Credentials…");
  const u = await readBytes(file);
  const type = sniffType(u, file.name);
  const meta = await readImageMeta(u, type);
  stage("Decoding pixels…");
  const bitmap = await decode(file);
  const dims = { width: bitmap.width, height: bitmap.height };
  stage(opts.deep === false ? "Running pixel forensics…" : "Running forensics and the neural classifier…");
  const { pixels, classifier: cls } = await analyzeBitmap(bitmap, opts);
  const evidence = [...metaEvidence(meta, dims), ...pixelEvidence(pixels, cls, file.size)];
  // Fitted weights assume the classifier ran; without it, use the hand-set ones.
  const hasModel = evidence.some((e) => e.id === "classifier");
  let key = hasModel ? "image" : "image-basic";
  // Pro cloud model: only when the visitor switched it on for this check.
  if (opts.pro) {
    stage("Checking with the large cloud model…");
    const { cloudEvidence, proWeightsKey } = await import("../common/cloud.js");
    const row = await cloudEvidence("image", file, "image", "classifier");
    evidence.push(row);
    if (row.id === "cloud-image") key = proWeightsKey("image", "image");
  }
  const scored = combine("image", evidence, { weights: opts.weights, weightsKey: key });
  return { ...scored, type, dims, meta, pixels, classifier: cls, bitmap };
}

export function warmClassifier(onProgress) {
  return classifier().call({ warm: true }, [], onProgress);
}
