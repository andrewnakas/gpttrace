/* C2PA Content Credentials reader.
 *
 * Finds the JUMBF manifest store in JPEG (APP11), PNG (caBX), RIFF (C2PA
 * chunk: WebP, WAV, AVI), ISO-BMFF (uuid box: MP4, MOV, HEIC, AVIF, M4A),
 * MP3 (ID3 GEOB, via tags.js) or a raw .c2pa sidecar, walks the boxes and
 * decodes the claim, the assertions and the signer certificate.
 *
 * It does NOT verify the signature or the hard binding; the result says so
 * (`validated: false`). It answers "what does the file claim, and who signed
 * the claim", which is what an AI-provenance check needs. GPTTrace's
 * original scanC2PA string scan is kept as the fallback.
 */
import { ascii, latin1, u16be, u32be, u64be, indexOfBytes, bytesOf, decodeUtf8 } from "./bytes.js";
import { decodeCBOR, untag } from "./cbor.js";
import { digitalSourceType, matchGenerators } from "./generators.js";

const C2PA_UUID = "d8fec3d61b0e483c92975828877ec481";
const hex = (u8) => Array.from(u8, (b) => b.toString(16).padStart(2, "0")).join("");

// ---- containers --------------------------------------------------------

function fromJPEG(u) {
  const groups = new Map();
  let p = 2;
  while (p + 4 < u.length) {
    if (u[p] !== 0xff) break;
    const m = u[p + 1];
    if (m === 0xd9 || m === 0xda) break;
    if (m === 0x01 || (m >= 0xd0 && m <= 0xd7)) { p += 2; continue; }
    const len = u16be(u, p + 2);
    if (m === 0xeb && u[p + 4] === 0x4a && u[p + 5] === 0x50) { // APP11 "JP"
      const d = u.subarray(p + 4, p + 2 + len);
      const en = u16be(d, 2), z = u32be(d, 4);
      let body = d.subarray(8);
      if (groups.has(en) && z > 1) body = body.subarray(u32be(body, 0) === 1 ? 16 : 8);
      if (!groups.has(en)) groups.set(en, []);
      groups.get(en).push(body);
    }
    p += 2 + len;
  }
  for (const parts of groups.values()) {
    const all = concat(parts);
    if (ascii(all, 4, 4) === "jumb") return all;
  }
  return null;
}

function fromPNG(u) {
  let p = 8;
  while (p + 12 <= u.length) {
    const len = u32be(u, p), type = ascii(u, p + 4, 4);
    if (type === "caBX") return u.subarray(p + 8, p + 8 + len);
    if (type === "IEND") break;
    p += 12 + len;
  }
  return null;
}

function fromRIFF(u) {
  let p = 12;
  while (p + 8 <= u.length) {
    const id = ascii(u, p, 4), len = u[p + 4] | (u[p + 5] << 8) | (u[p + 6] << 16) | (u[p + 7] << 24);
    if (id === "C2PA") return u.subarray(p + 8, p + 8 + len);
    if (len < 0) break;
    p += 8 + len + (len & 1);
  }
  return null;
}

function fromBMFF(u) {
  let p = 0;
  while (p + 8 <= u.length) {
    let size = u32be(u, p);
    const type = ascii(u, p + 4, 4);
    let hdr = 8;
    if (size === 1) { size = u64be(u, p + 8); hdr = 16; }
    else if (size === 0) size = u.length - p;
    if (size < hdr) break;
    if (type === "uuid" && hex(u.subarray(p + hdr, p + hdr + 16)) === C2PA_UUID) {
      let q = p + hdr + 16 + 4; // version + flags
      const z = u.indexOf(0, q);
      const purpose = ascii(u, q, z - q);
      q = z + 1;
      if (purpose === "manifest") q += 8; // merkle offset
      if (purpose === "manifest" || purpose === "original") return u.subarray(q, p + size);
    }
    p += size;
  }
  return null;
}

// Last resort: any byte run that looks like a C2PA superbox.
function fromScan(u) {
  const JUMB = bytesOf("jumb");
  let i = 0;
  while ((i = indexOfBytes(u, JUMB, i + 1)) > 3) {
    const start = i - 4;
    const len = u32be(u, start);
    if (len > 24 && start + len <= u.length && ascii(u, i + 8, 4) === "jumd") {
      const box = u.subarray(start, start + len);
      const tree = walk(box, 0, box.length);
      if (tree[0] && tree[0].label === "c2pa") return box;
    }
  }
  return null;
}

function concat(parts) {
  const out = new Uint8Array(parts.reduce((a, c) => a + c.length, 0));
  let o = 0; parts.forEach((c) => { out.set(c, o); o += c.length; });
  return out;
}

export function extractJUMBF(u, type) {
  try {
    if (ascii(u, 4, 4) === "jumb") return { bytes: u, source: "sidecar" };
    let b = null, source = null;
    if (type === "jpeg") { b = fromJPEG(u); source = "JPEG APP11 segment"; }
    else if (type === "png") { b = fromPNG(u); source = "PNG caBX chunk"; }
    else if (type === "webp" || type === "wav" || type === "avi") { b = fromRIFF(u); source = "RIFF C2PA chunk"; }
    else if (/^(mp4|mov|heic|avif|m4a)$/.test(type)) { b = fromBMFF(u); source = "ISO-BMFF uuid box"; }
    if (!b) { b = fromScan(u); source = "embedded JUMBF"; }
    return b ? { bytes: b, source } : null;
  } catch (e) {
    return null;
  }
}

// ---- JUMBF -------------------------------------------------------------

function walk(u, start, end) {
  const out = [];
  let p = start;
  while (p + 8 <= end) {
    let size = u32be(u, p);
    const type = ascii(u, p + 4, 4);
    let hdr = 8;
    if (size === 1) { size = u64be(u, p + 8); hdr = 16; }
    else if (size === 0) size = end - p;
    if (size < hdr || p + size > end) break;
    const box = { type, start: p + hdr, end: p + size };
    if (type === "jumb") {
      box.children = walk(u, box.start, box.end);
      const d = box.children[0];
      if (d && d.type === "jumd") {
        box.uuid = hex(u.subarray(d.start, d.start + 16));
        const toggles = u[d.start + 16];
        if (toggles & 2) {
          const z = u.indexOf(0, d.start + 17);
          box.label = decodeUtf8(u.subarray(d.start + 17, z < 0 || z > d.end ? d.end : z));
        }
      }
    }
    out.push(box);
    p += size;
  }
  return out;
}

function content(u, jumb) {
  const c = (jumb.children || []).find((b) => b.type !== "jumd");
  if (!c) return null;
  const body = u.subarray(c.start, c.end);
  try {
    if (c.type === "cbor") return decodeCBOR(body);
    if (c.type === "json") return JSON.parse(decodeUtf8(body));
  } catch (e) { return { _error: String(e.message || e) }; }
  return { _type: c.type, _bytes: body.length };
}

// ---- X.509 (just enough to name the signer) ----------------------------

const OIDS = { "550403": "CN", "55040a": "O", "55040b": "OU", "550406": "C" };

function der(u, p) {
  const tag = u[p];
  let len = u[p + 1], hl = 2;
  if (len & 0x80) {
    const n = len & 0x7f; len = 0;
    for (let i = 0; i < n; i++) len = len * 256 + u[p + 2 + i];
    hl = 2 + n;
  }
  return { tag, start: p + hl, end: p + hl + len };
}
function children(u, node) {
  const out = [];
  for (let p = node.start; p < node.end;) { const c = der(u, p); out.push(c); p = c.end; }
  return out;
}
function name(u, node) {
  const o = {};
  for (const set of children(u, node)) for (const atv of children(u, set)) {
    const [oid, val] = children(u, atv);
    const k = OIDS[hex(u.subarray(oid.start, oid.end))];
    if (k) o[k] = decodeUtf8(u.subarray(val.start, val.end));
  }
  return o;
}
function time(u, n) {
  const s = ascii(u, n.start, n.end - n.start);
  const y = n.tag === 0x17 ? (+s.slice(0, 2) < 50 ? "20" : "19") + s.slice(0, 2) : s.slice(0, 4);
  const r = n.tag === 0x17 ? s.slice(2) : s.slice(4);
  return `${y}-${r.slice(0, 2)}-${r.slice(2, 4)}`;
}
export function parseCert(u) {
  try {
    const cert = der(u, 0);
    const tbs = children(u, cert)[0];
    let f = children(u, tbs);
    if (f[0].tag === 0xa0) f = f.slice(1);
    const validity = children(u, f[3]);
    return { issuer: name(u, f[2]), subject: name(u, f[4]), notBefore: time(u, validity[0]), notAfter: time(u, validity[1]) };
  } catch (e) { return null; }
}

function signer(sig) {
  try {
    const cose = untag(sig);
    if (!Array.isArray(cose)) return null;
    const prot = cose[0] && cose[0].length ? decodeCBOR(cose[0]) : {};
    const unprot = cose[1] || {};
    const algs = { "-7": "ES256", "-35": "ES384", "-36": "ES512", "-37": "PS256", "-38": "PS384", "-39": "PS512", "-8": "Ed25519" };
    let chain = prot[33] || unprot[33] || prot.x5chain || unprot.x5chain;
    if (chain instanceof Uint8Array) chain = [chain];
    const certs = (chain || []).map(parseCert).filter(Boolean);
    return { alg: algs[prot[1]] || String(prot[1] || ""), cert: certs[0] || null, chain: certs, timestamped: !!(unprot.sigTst || unprot.sigTst2) };
  } catch (e) { return null; }
}

// ---- manifest model ----------------------------------------------------

function agentName(a) {
  if (!a) return "";
  if (typeof a === "string") return a;
  return [a.name, a.version].filter(Boolean).join(" ");
}

function readManifest(u, m) {
  const out = { label: m.label, claimGenerator: "", generators: [], title: "", format: "", actions: [], assertions: [], ingredients: [], signer: null };
  for (const part of m.children || []) {
    if (part.type !== "jumb") continue;
    const L = part.label || "";
    if (L === "c2pa.assertions") {
      for (const a of part.children || []) {
        if (a.type !== "jumb" || !a.label) continue;
        out.assertions.push(a.label);
        if (/^c2pa\.actions/.test(a.label)) {
          const v = content(u, a);
          for (const act of (v && v.actions) || []) {
            out.actions.push({
              action: act.action || "",
              digitalSourceType: act.digitalSourceType || (act.parameters && act.parameters.digitalSourceType) || "",
              softwareAgent: agentName(act.softwareAgent),
              description: act.description || (act.parameters && act.parameters.description) || "",
              when: act.when || ""
            });
          }
        } else if (/^c2pa\.ingredient/.test(a.label)) {
          const v = content(u, a);
          if (v && v.title) out.ingredients.push(v.title);
        } else if (/^stds\.schema-org\.CreativeWork/.test(a.label) || /^stds\.iptc/.test(a.label)) {
          const v = content(u, a);
          if (v) out.creativeWork = JSON.stringify(v).slice(0, 2000);
        }
      }
    } else if (/^c2pa\.claim/.test(L)) {
      const c = content(u, part) || {};
      out.claimGenerator = c.claim_generator || "";
      out.generators = (c.claim_generator_info ? [].concat(c.claim_generator_info) : []).map(agentName);
      out.title = c["dc:title"] || c.title || "";
      out.format = c["dc:format"] || c.format || "";
      out.instanceId = c.instanceID || c.instance_id || "";
    } else if (L === "c2pa.signature") {
      out.signer = signer(content(u, part));
    }
  }
  const dst = out.actions.map((a) => digitalSourceType(a.digitalSourceType)).filter(Boolean);
  out.aiGenerated = dst.some((d) => d.ai && !d.partial && !d.weak);
  out.aiEdited = dst.some((d) => d.ai && d.partial);
  out.sourceTypes = dst.map((d) => d.label);
  return out;
}

// Strings in the raw bytes, from GPTTrace's original scanC2PA.
export function scanMarkers(u) {
  const asc = latin1(u, 5 * 1024 * 1024);
  const markers = [];
  if (/jumbf/i.test(asc)) markers.push("jumbf box");
  if (/c2pa/i.test(asc)) markers.push("c2pa box");
  if (/contentcredential/i.test(asc)) markers.push("contentcredential string");
  if (/c2pa\.actions/i.test(asc)) markers.push("c2pa.actions");
  if (/com\.adobe\.generative-ai/i.test(asc)) markers.push("Adobe generative-ai assertion");
  if (/trainedAlgorithmicMedia/.test(asc)) markers.push("trainedAlgorithmicMedia");
  return markers;
}

export function readC2PA(u, type, extraJumbf) {
  const found = extraJumbf ? { bytes: extraJumbf, source: "ID3 GEOB frame" } : extractJUMBF(u, type);
  const res = { present: false, source: null, manifests: [], active: null, validated: false, markers: [], aiGenerated: false, aiEdited: false, signedBy: "", generators: [] };
  if (!found) {
    res.markers = scanMarkers(u);
    return res;
  }
  res.present = true;
  res.source = found.source;
  const j = found.bytes;
  const store = walk(j, 0, j.length).find((b) => b.type === "jumb");
  if (!store) return res;
  res.manifests = (store.children || []).filter((b) => b.type === "jumb").map((m) => readManifest(j, m));
  res.active = res.manifests[res.manifests.length - 1] || null;
  const a = res.active;
  if (a) {
    res.aiGenerated = a.aiGenerated;
    // An earlier (ingredient) manifest that was AI makes the active one an AI edit.
    res.aiEdited = a.aiEdited || (!a.aiGenerated && res.manifests.some((m) => m.aiGenerated || m.aiEdited));
    const s = a.signer && a.signer.cert && a.signer.cert.subject;
    res.signedBy = s ? [s.O, s.CN].filter(Boolean).filter((v, i, x) => x.indexOf(v) === i).join(" · ") : "";
    const hay = [a.claimGenerator, ...a.generators, res.signedBy, ...a.actions.map((x) => x.softwareAgent + " " + x.description)];
    res.generators = matchGenerators(hay);
  }
  return res;
}
