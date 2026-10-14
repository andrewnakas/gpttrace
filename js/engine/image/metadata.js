/* Image metadata and provenance: EXIF/XMP/IPTC (exifr, self-hosted), raw XMP
 * packets, PNG text chunks (GPTTrace scanPngText + the Ai_writing_detector
 * PNG/JPEG/WebP parsers), JPEG comments, C2PA, camera plausibility. */
import { latin1, ascii, u32be, u16be, decodeUtf8, decodeLatin1, clean } from "../common/bytes.js";
import { readC2PA } from "../common/c2pa.js";
import { matchGenerators, digitalSourceType } from "../common/generators.js";

let exifrMod = null;
async function exifr() {
  if (!exifrMod) exifrMod = await import("../../../vendor/exifr-7.1.3/full.esm.mjs");
  return exifrMod.default || exifrMod;
}

// Stable-Diffusion-style parameter strings ("Steps: 25, Sampler: …").
export const SD_PARAM_RX = /\bSteps:\s*\d+|\bSampler:\s*\w+|\bCFG scale:\s*[\d.]+|\bSeed:\s*\d+|\bModel hash:\s*\w+/i;
const SD_KEYS = /^(parameters|prompt|workflow|invokeai_metadata|sd-metadata|dream|negative_prompt|fooocus_scheme|generation_data|comfyui)$/i;

export function scanPngText(u) {
  const items = [];
  if (u.length < 8 || u[0] !== 0x89 || ascii(u, 1, 3) !== "PNG") return items;
  let p = 8;
  while (p + 12 <= u.length && items.length < 48) {
    const len = u32be(u, p), type = ascii(u, p + 4, 4);
    if (p + 12 + len > u.length) break;
    if (type === "tEXt" || type === "iTXt" || type === "zTXt") {
      const d = u.subarray(p + 8, p + 8 + len);
      const z = d.indexOf(0);
      if (z > 0) {
        const key = decodeLatin1(d.subarray(0, z));
        let val = "";
        if (type === "tEXt") val = decodeLatin1(d.subarray(z + 1));
        else if (type === "iTXt") {
          // keyword\0 compFlag compMethod lang\0 transKey\0 text
          const comp = d[z + 1];
          let q = d.indexOf(0, z + 3); q = d.indexOf(0, q + 1);
          val = comp ? "(compressed iTXt)" : decodeUtf8(d.subarray(q + 1));
        } else val = "(compressed zTXt)";
        items.push({ key, value: clean(val).slice(0, 20000), chunk: type });
      }
    } else if (type === "eXIf") items.push({ key: "eXIf", value: "(EXIF block)", chunk: type });
    if (type === "IEND") break;
    p += 12 + len;
  }
  return items;
}

// zTXt / compressed iTXt (ComfyUI sometimes compresses its workflow).
export async function inflatePngText(u, items) {
  if (typeof DecompressionStream === "undefined" || !items.some((i) => /compressed/.test(i.value))) return items;
  let p = 8;
  const out = [...items];
  while (p + 12 <= u.length) {
    const len = u32be(u, p), type = ascii(u, p + 4, 4);
    if (type === "zTXt" || type === "iTXt") {
      const d = u.subarray(p + 8, p + 8 + len);
      const z = d.indexOf(0);
      const key = decodeLatin1(d.subarray(0, z));
      let data = null;
      if (type === "zTXt") data = d.subarray(z + 2);
      else if (d[z + 1] === 1) { let q = d.indexOf(0, z + 3); q = d.indexOf(0, q + 1); data = d.subarray(q + 1); }
      if (data) {
        try {
          const s = await new Response(new Blob([data]).stream().pipeThrough(new DecompressionStream("deflate"))).text();
          const i = out.findIndex((x) => x.key === key && /compressed/.test(x.value));
          if (i >= 0) out[i] = { ...out[i], value: clean(s).slice(0, 20000) };
        } catch (e) { /* leave as is */ }
      }
    }
    if (type === "IEND") break;
    p += 12 + len;
  }
  return out;
}

function jpegComments(u) {
  const out = [];
  if (u[0] !== 0xff || u[1] !== 0xd8) return out;
  let p = 2;
  while (p + 4 < u.length && u[p] === 0xff) {
    const m = u[p + 1];
    if (m === 0xda || m === 0xd9) break;
    const len = u16be(u, p + 2);
    if (m === 0xfe) out.push(clean(decodeUtf8(u.subarray(p + 4, p + 2 + len))));
    p += 2 + len;
  }
  return out;
}

// Raw XMP packets: exifr merges XMP into its output, but drops fields it
// doesn't know (Iptc4xmpExt:DigitalSourceType variants, custom AI namespaces).
function xmpPackets(u) {
  const s = latin1(u, Math.min(u.length, 8 * 1024 * 1024));
  const out = [];
  const rx = /<x:xmpmeta[\s\S]*?<\/x:xmpmeta>/g;
  let m;
  while ((m = rx.exec(s)) && out.length < 4) out.push(m[0].replace(/\0/g, ""));
  return out;
}

const CAMERA_KEYS = ["Make", "Model", "LensModel", "LensMake", "ExposureTime", "FNumber", "ISO", "ISOSpeedRatings", "FocalLength", "DateTimeOriginal", "GPSLatitude", "GPSLongitude", "SubSecTimeOriginal", "BodySerialNumber"];
export function cameraSignals(exif) {
  if (!exif) return { plausible: false, present: [] };
  const present = CAMERA_KEYS.filter((k) => exif[k] !== undefined && exif[k] !== null && exif[k] !== "");
  return { plausible: present.length >= 4 && present.includes("Make") && (present.includes("ExposureTime") || present.includes("FNumber")), present };
}

const STRONG_FIELDS = ["Software", "CreatorTool", "HistorySoftwareAgent", "GenAIToolName", "Generator", "parameters", "prompt", "workflow", "Make", "Model", "ProcessingSoftware"];
const TEXT_FIELDS = ["Artist", "Copyright", "ImageDescription", "UserComment", "XPComment", "XPSubject", "XPTitle", "XPKeywords", "Source", "Description", "Comment", "Credit", "Rights", "Title", "Label", "Subject", "Keywords", "Caption", "Headline", "CaptionAbstract", "SpecialInstructions", "ContentType"];

export async function readImageMeta(u, type) {
  const res = { type, exif: {}, png: [], comments: [], xmp: [], c2pa: null, generators: [], strongGenerators: [], sourceType: null, sdParams: null, camera: null };
  res.c2pa = readC2PA(u, type);
  if (type === "png") res.png = await inflatePngText(u, scanPngText(u));
  if (type === "jpeg") res.comments = jpegComments(u);
  res.xmp = xmpPackets(u);
  try {
    const ex = await exifr();
    res.exif = (await ex.parse(u, { xmp: true, iptc: true, icc: false, jfif: false, ihdr: true, tiff: true, exif: true, gps: true, mergeOutput: true, translateValues: true, reviveValues: false })) || {};
  } catch (e) { res.exif = {}; }

  const val = (v) => (typeof v === "string" ? v : v && typeof v === "object" ? JSON.stringify(v) : v == null ? "" : String(v));
  const strong = [], weak = [];
  STRONG_FIELDS.forEach((k) => { if (res.exif[k] != null) strong.push(`${k}: ${val(res.exif[k])}`); });
  TEXT_FIELDS.forEach((k) => { if (res.exif[k] != null) weak.push(`${k}: ${val(res.exif[k])}`); });
  res.png.forEach((p) => (SD_KEYS.test(p.key) || /^software$/i.test(p.key) ? strong : weak).push(`${p.key}: ${p.value.slice(0, 2000)}`));
  res.comments.forEach((c) => weak.push(`JPEG comment: ${c}`));
  const xmpTool = res.xmp.flatMap((x) => [...x.matchAll(/(?:CreatorTool|softwareAgent|Software)\s*(?:=\s*"|>)([^"<]{2,200})/g)].map((m) => `XMP tool: ${m[1]}`));
  strong.push(...xmpTool);

  // IPTC Digital Source Type, from exifr or straight from the XMP packet.
  const dstRaw = [res.exif.DigitalSourceType, res.exif.DigitalSourceFileType, ...res.xmp.flatMap((x) => [...x.matchAll(/DigitalSourceType\s*(?:=\s*"|>\s*)([^"<\s]+)/g)].map((m) => m[1]))].filter(Boolean);
  res.sourceType = dstRaw.map((s) => digitalSourceType(String(s))).find(Boolean) || null;

  res.sdParams = res.png.find((p) => SD_KEYS.test(p.key) && (SD_PARAM_RX.test(p.value) || /"class_type"|"inputs"|"nodes"|negative_prompt|sampler/i.test(p.value))) || null;
  if (!res.sdParams) {
    const uc = [res.exif.UserComment, res.exif.ImageDescription, ...res.comments].map(val).find((s) => SD_PARAM_RX.test(s || ""));
    if (uc) res.sdParams = { key: "UserComment", value: uc, chunk: "EXIF" };
  }

  res.strongGenerators = matchGenerators(strong, "image").concat(res.c2pa.generators || []);
  const strongIds = new Set(res.strongGenerators.map((g) => g.id));
  res.generators = res.strongGenerators.concat(matchGenerators(weak, "image").filter((g) => !strongIds.has(g.id)).map((g) => ({ ...g, weak: true })));
  res.camera = cameraSignals(res.exif);
  return res;
}

// Common native output sizes of image generators (w×h, either orientation).
const GEN_SIZES = new Set(["512x512", "768x768", "1024x1024", "2048x2048", "1536x1024", "1792x1024", "1344x768", "1216x832", "1152x896", "1536x640", "1456x816", "1232x928", "1664x928", "1472x1104", "1584x1056", "1328x1328", "2048x1152", "1920x1088", "896x1152", "640x1536", "1280x1280", "1440x1440", "1408x1408", "2016x1152", "1856x1024"]);
export function generatorSize(w, h) {
  const key = w >= h ? `${w}x${h}` : `${h}x${w}`;
  if (GEN_SIZES.has(key)) return "exact";
  if (w % 64 === 0 && h % 64 === 0 && Math.max(w, h) <= 2048 && Math.min(w, h) >= 512) return "multiple-of-64";
  return null;
}
