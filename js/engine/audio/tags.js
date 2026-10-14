/* Audio container headers and tags.
 *
 * header(): codec, native sample rate, channels, bit depth, bitrate — read
 * from the file, because decodeAudioData resamples and would hide them.
 * tags(): ID3v2/ID3v1, RIFF INFO, MP4 ilst, Vorbis comments; plus the
 * C2PA store when it travels in an ID3 GEOB frame. */
import { ascii, u16be, u32be, u32le, u16le, synchsafe, decodeUtf8, decodeLatin1, decodeUtf16, clean } from "../common/bytes.js";

// ---- MP3 ----------------------------------------------------------------
const BITRATES = {
  "1-3": [0, 32, 40, 48, 56, 64, 80, 96, 112, 128, 160, 192, 224, 256, 320],
  "2-3": [0, 8, 16, 24, 32, 40, 48, 56, 64, 80, 96, 112, 128, 144, 160]
};
const RATES = { 3: [44100, 48000, 32000], 2: [22050, 24000, 16000], 0: [11025, 12000, 8000] };

function id3End(u) {
  if (ascii(u, 0, 3) !== "ID3") return 0;
  return 10 + synchsafe(u, 6) + (u[5] & 0x10 ? 10 : 0);
}

function mp3Header(u) {
  let p = id3End(u);
  for (let tries = 0; p < u.length - 4 && tries < 65536; p++, tries++) {
    if (u[p] !== 0xff || (u[p + 1] & 0xe0) !== 0xe0) continue;
    const ver = (u[p + 1] >> 3) & 3, layer = (u[p + 1] >> 1) & 3, bri = u[p + 2] >> 4, sri = (u[p + 2] >> 2) & 3;
    if (ver === 1 || layer !== 1 || bri === 0 || bri === 15 || sri === 3) continue; // Layer III only
    const sampleRate = RATES[ver][sri];
    const bitrate = BITRATES[ver === 3 ? "1-3" : "2-3"][bri];
    const channels = (u[p + 3] >> 6) === 3 ? 1 : 2;
    const out = { codec: "mp3", sampleRate, channels, bitrate, mpegVersion: ver === 3 ? 1 : ver === 2 ? 2 : 2.5 };
    // Xing/Info + LAME tag: encoder version and the lowpass it applied.
    const side = ver === 3 ? (channels === 1 ? 17 : 32) : channels === 1 ? 9 : 17;
    const x = p + 4 + side;
    const tag = ascii(u, x, 4);
    if (tag === "Xing" || tag === "Info") {
      out.vbr = tag === "Xing";
      const flags = u32be(u, x + 4);
      let q = x + 8;
      if (flags & 1) { out.frames = u32be(u, q); q += 4; }
      if (flags & 2) { out.bytes = u32be(u, q); q += 4; }
      if (flags & 4) q += 100;
      if (flags & 8) q += 4;
      const enc = ascii(u, q, 9);
      if (/^(LAME|Lavc|Lavf|GOGO|L3.99)/.test(enc)) {
        out.encoder = enc.replace(/[^\x20-\x7e]/g, "").trim();
        out.lowpass = u[q + 10] * 100 || null; // Hz
      }
      if (out.frames) out.duration = (out.frames * (ver === 3 ? 1152 : 576)) / sampleRate;
    } else if (ascii(u, x, 4) === "VBRI") out.vbr = true;
    return out;
  }
  return { codec: "mp3" };
}

// ---- WAV / RIFF -----------------------------------------------------------
function riffChunks(u) {
  const out = [];
  let p = 12;
  while (p + 8 <= u.length && out.length < 64) {
    const id = ascii(u, p, 4), len = u32le(u, p + 4);
    out.push({ id, start: p + 8, len });
    if (len > u.length) break;
    p += 8 + len + (len & 1);
  }
  return out;
}
const WAV_FORMATS = { 1: "PCM", 3: "IEEE float", 6: "A-law", 7: "µ-law", 0x11: "IMA ADPCM", 0x55: "MP3", 0xfffe: "PCM (extensible)" };
function wavHeader(u) {
  const fmt = riffChunks(u).find((c) => c.id === "fmt ");
  if (!fmt) return { codec: "wav" };
  const f = fmt.start;
  return { codec: "wav", format: WAV_FORMATS[u16le(u, f)] || "0x" + u16le(u, f).toString(16), channels: u16le(u, f + 2), sampleRate: u32le(u, f + 4), bitsPerSample: u16le(u, f + 14), bitrate: Math.round((u32le(u, f + 8) * 8) / 1000) };
}

// ---- FLAC / OGG -----------------------------------------------------------
function flacBlocks(u) {
  const out = [];
  let p = 4;
  for (let i = 0; i < 128 && p + 4 <= u.length; i++) {
    const last = u[p] & 0x80, type = u[p] & 0x7f, len = (u[p + 1] << 16) | (u[p + 2] << 8) | u[p + 3];
    out.push({ type, start: p + 4, len });
    p += 4 + len;
    if (last) break;
  }
  return out;
}
function flacHeader(u) {
  const si = flacBlocks(u).find((b) => b.type === 0);
  if (!si) return { codec: "flac" };
  const s = si.start;
  const sampleRate = (u[s + 10] << 12) | (u[s + 11] << 4) | (u[s + 12] >> 4);
  return { codec: "flac", sampleRate, channels: ((u[s + 12] >> 1) & 7) + 1, bitsPerSample: (((u[s + 12] & 1) << 4) | (u[s + 13] >> 4)) + 1 };
}
function oggHeader(u) {
  const seg = u[26], data = 27 + seg;
  if (ascii(u, data, 8) === "OpusHead") return { codec: "opus", channels: u[data + 9], sampleRate: 48000, inputSampleRate: u32le(u, data + 12) };
  if (ascii(u, data + 1, 6) === "vorbis") return { codec: "vorbis", channels: u[data + 11], sampleRate: u32le(u, data + 12), bitrate: Math.round(u32le(u, data + 20) / 1000) || null };
  return { codec: "ogg" };
}

// ---- MP4 / M4A ------------------------------------------------------------
function boxes(u, start, end, out, depth = 0) {
  let p = start;
  while (p + 8 <= end) {
    let size = u32be(u, p);
    const type = ascii(u, p + 4, 4);
    let hdr = 8;
    if (size === 1) { size = u32be(u, p + 8) * 4294967296 + u32be(u, p + 12); hdr = 16; }
    else if (size === 0) size = end - p;
    if (size < hdr || p + size > end) break;
    out.push({ type, start: p + hdr, end: p + size, depth });
    const kids = { moov: 0, trak: 0, mdia: 0, minf: 0, stbl: 0, udta: 0, ilst: 0, edts: 0 };
    if (type in kids) boxes(u, p + hdr, p + size, out, depth + 1);
    else if (type === "meta") boxes(u, p + hdr + 4, p + size, out, depth + 1);
    else if (type === "stsd") boxes(u, p + hdr + 8, p + size, out, depth + 1);
    else if (depth >= 1 && /^(\xa9...|----|desc|ldes|cprt|aART|tool)$/.test(type)) boxes(u, p + hdr, p + size, out, depth + 1);
    p += size;
  }
  return out;
}
export function mp4Boxes(u) { return boxes(u, 0, u.length, []); }

function mp4Header(u, list) {
  const a = list.find((b) => b.type === "mp4a" || b.type === "alac" || b.type === "Opus" || b.type === "fLaC" || b.type === "ac-3");
  const out = { codec: a ? (a.type === "mp4a" ? "aac" : a.type.toLowerCase()) : "mp4" };
  if (a) { out.channels = u16be(u, a.start + 16); out.bitsPerSample = u16be(u, a.start + 18); out.sampleRate = u16be(u, a.start + 24); }
  const esds = list.find((b) => b.type === "esds");
  if (esds) {
    // DecoderConfigDescriptor (tag 4): avgBitrate at +9 from its payload.
    for (let p = esds.start + 4; p < esds.end - 13; p++) if (u[p] === 0x04) {
      let q = p + 1; while (u[q] & 0x80) q++; q++;
      out.bitrate = Math.round(u32be(u, q + 9) / 1000) || null; break;
    }
  }
  return out;
}

export function header(u, type) {
  try {
    if (type === "mp3") return mp3Header(u);
    if (type === "wav") return wavHeader(u);
    if (type === "flac") return flacHeader(u);
    if (type === "ogg") return oggHeader(u);
    if (type === "m4a" || type === "mp4" || type === "mov") return mp4Header(u, mp4Boxes(u));
  } catch (e) { /* fall through */ }
  return { codec: type };
}

// ---- tags -----------------------------------------------------------------
const ID3_NAMES = { TIT2: "Title", TPE1: "Artist", TALB: "Album", TSSE: "Encoder settings", TENC: "Encoded by", TXXX: "User text", COMM: "Comment", WXXX: "URL", WOAS: "Source URL", TCOP: "Copyright", TPUB: "Publisher", TCON: "Genre", TDRC: "Date", TYER: "Year", TOOL: "Tool", PRIV: "Private", GEOB: "Object", USLT: "Lyrics", TCOM: "Composer" };

function id3Text(b) {
  const enc = b[0], s = b.subarray(1);
  const t = enc === 1 || enc === 2 ? decodeUtf16(s) : enc === 3 ? decodeUtf8(s) : decodeLatin1(s);
  return clean(t.replace(/\0+/g, " · "));
}

function id3v2(u, out) {
  if (ascii(u, 0, 3) !== "ID3") return null;
  const ver = u[3], end = Math.min(u.length, id3End(u));
  let p = 10;
  if (u[5] & 0x40) p += ver === 4 ? synchsafe(u, 10) : u32be(u, 10) + 4;
  let geob = null;
  while (p + 10 < end) {
    const id = ascii(u, p, 4);
    if (!/^[A-Z0-9]{4}$/.test(id)) break;
    const size = ver === 4 ? synchsafe(u, p + 4) : u32be(u, p + 4);
    const b = u.subarray(p + 10, p + 10 + size);
    if (id === "GEOB") {
      // enc, mime\0, filename\0, description\0, data
      const z1 = b.indexOf(0, 1), mime = decodeLatin1(b.subarray(1, z1));
      const term = b[0] === 1 || b[0] === 2 ? [0, 0] : [0];
      let q = z1 + 1;
      for (let k = 0; k < 2; k++) { while (q < b.length && !(b[q] === 0 && (term.length === 1 || b[q + 1] === 0))) q += term.length; q += term.length; }
      if (/c2pa/i.test(mime)) geob = b.subarray(q);
      out.push({ key: "GEOB", name: "Object", value: mime });
    } else if (id === "PRIV") {
      const z = b.indexOf(0);
      out.push({ key: "PRIV", name: "Private", value: decodeLatin1(b.subarray(0, z)) });
    } else if (id[0] === "T" || id === "COMM" || id === "USLT") {
      const body = id === "COMM" || id === "USLT" ? Uint8Array.of(b[0], ...b.subarray(4)) : b;
      out.push({ key: id, name: ID3_NAMES[id] || id, value: id3Text(body).slice(0, 500) });
    } else if (id[0] === "W") {
      out.push({ key: id, name: ID3_NAMES[id] || id, value: clean(decodeLatin1(id === "WXXX" ? b.subarray(b.indexOf(0, 1) + 1) : b)) });
    }
    p += 10 + size;
  }
  return geob;
}

function id3v1(u, out) {
  const p = u.length - 128;
  if (p < 0 || ascii(u, p, 3) !== "TAG") return;
  const f = (o, n) => clean(decodeLatin1(u.subarray(p + o, p + o + n)).replace(/\0.*$/, ""));
  [["Title", 3, 30], ["Artist", 33, 30], ["Album", 63, 30], ["Comment", 97, 28]].forEach(([n, o, l]) => { const v = f(o, l); if (v) out.push({ key: "ID3v1 " + n, name: n, value: v }); });
}

const RIFF_INFO = { ISFT: "Software", INAM: "Title", IART: "Artist", ICMT: "Comment", ICOP: "Copyright", IENG: "Engineer", ICRD: "Date", IPRD: "Album", IGNR: "Genre", ISRC: "Source", ITCH: "Technician" };
function riffTags(u, out) {
  for (const c of riffChunks(u)) {
    if (c.id === "LIST" && ascii(u, c.start, 4) === "INFO") {
      let p = c.start + 4;
      while (p + 8 <= c.start + c.len) {
        const id = ascii(u, p, 4), len = u32le(u, p + 4);
        out.push({ key: id, name: RIFF_INFO[id] || id, value: clean(decodeLatin1(u.subarray(p + 8, p + 8 + len)).replace(/\0+$/, "")) });
        p += 8 + len + (len & 1);
      }
    } else if (c.id === "bext") out.push({ key: "bext", name: "Broadcast ext.", value: clean(decodeLatin1(u.subarray(c.start, c.start + 256 + 32 + 32))) });
    else if (c.id === "iXML" || c.id === "_PMX") out.push({ key: c.id, name: c.id === "_PMX" ? "XMP" : "iXML", value: clean(decodeUtf8(u.subarray(c.start, c.start + Math.min(c.len, 4000)))) });
    else if (!["fmt ", "data", "fact", "LIST", "C2PA", "PEAK", "cue ", "smpl", "JUNK", "id3 ", "ID3 "].includes(c.id)) out.push({ key: "chunk", name: "RIFF chunk", value: c.id });
    if (c.id === "id3 " || c.id === "ID3 ") id3v2(u.subarray(c.start, c.start + c.len), out);
  }
}

function vorbisComments(b, out) {
  let p = 0;
  const vlen = u32le(b, p); p += 4;
  out.push({ key: "VENDOR", name: "Vendor", value: clean(decodeUtf8(b.subarray(p, p + vlen))) }); p += vlen;
  const n = u32le(b, p); p += 4;
  for (let i = 0; i < n && p < b.length; i++) {
    const l = u32le(b, p); p += 4;
    const s = decodeUtf8(b.subarray(p, p + l)); p += l;
    const e = s.indexOf("=");
    out.push({ key: s.slice(0, e).toUpperCase(), name: s.slice(0, e), value: clean(s.slice(e + 1)).slice(0, 500) });
  }
}

function mp4Tags(u, out) {
  const list = mp4Boxes(u);
  for (const b of list) {
    if (b.depth >= 2 && /^(\xa9...|desc|ldes|cprt|aART|----)$/.test(b.type)) {
      const data = list.find((d) => d.type === "data" && d.start >= b.start && d.end <= b.end);
      if (data) out.push({ key: b.type, name: { "\xa9too": "Encoder", "\xa9nam": "Title", "\xa9ART": "Artist", "\xa9cmt": "Comment", "\xa9swr": "Software", "\xa9alb": "Album", desc: "Description", cprt: "Copyright" }[b.type] || b.type.replace("\xa9", "©"), value: clean(decodeUtf8(u.subarray(data.start + 8, data.end))).slice(0, 500) });
    }
  }
}

export function tags(u, type) {
  const out = [];
  let geob = null;
  try {
    if (type === "mp3" || ascii(u, 0, 3) === "ID3") { geob = id3v2(u, out); id3v1(u, out); }
    if (type === "wav") riffTags(u, out);
    if (type === "flac") { const c = flacBlocks(u).find((b) => b.type === 4); if (c) vorbisComments(u.subarray(c.start, c.start + c.len), out); }
    if (type === "ogg") {
      // Second packet: comment header (OpusTags / \x03vorbis), assume it starts the 2nd page.
      const p2 = 27 + u[26] + [...u.subarray(27, 27 + u[26])].reduce((a, b) => a + b, 0);
      const d = p2 + 27 + u[p2 + 26];
      const magic = ascii(u, d, 8);
      if (magic === "OpusTags") vorbisComments(u.subarray(d + 8), out);
      else if (ascii(u, d + 1, 6) === "vorbis") vorbisComments(u.subarray(d + 7), out);
    }
    if (/^(m4a|mp4|mov)$/.test(type)) mp4Tags(u, out);
  } catch (e) { /* partial tags are fine */ }
  return { list: out, c2paJumbf: geob };
}

// Fields where a generator name means "this tool made the file", vs. free
// text (titles, lyrics) where it may just be a word.
export const STRONG_TAGS = /^(TSSE|TENC|TOOL|ISFT|ITCH|IENG|VENDOR|ENCODER|ENCODED_BY|ENCODEDBY|©too|©swr|PRIV|WOAS|WXXX|TPUB|bext|_PMX|iXML)$/i;
