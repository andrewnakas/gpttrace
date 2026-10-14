/* Byte-level helpers shared by every engine. Pure functions, no DOM, so they
 * run in workers and in Node (eval harness) unchanged. */

export async function readBytes(file) {
  return new Uint8Array(await file.arrayBuffer());
}

export function ascii(u8, start, len) {
  let s = "";
  const end = Math.min(u8.length, start + len);
  for (let i = start; i < end; i++) s += String.fromCharCode(u8[i]);
  return s;
}

// Printable ASCII with everything else mapped to \0, for regex scanning of
// raw files (the original GPTTrace bufferToAscii).
export function latin1(u8, limit) {
  const len = Math.min(u8.length, limit || u8.length);
  const parts = [];
  const CH = 65536;
  for (let off = 0; off < len; off += CH) {
    const n = Math.min(CH, len - off);
    const a = new Array(n);
    for (let i = 0; i < n; i++) {
      const c = u8[off + i];
      a[i] = c >= 32 && c < 127 ? c : 0;
    }
    parts.push(String.fromCharCode.apply(null, a));
  }
  return parts.join("");
}

export const u16be = (u, p) => (u[p] << 8) | u[p + 1];
export const u16le = (u, p) => u[p] | (u[p + 1] << 8);
export const u32be = (u, p) => ((u[p] << 24) >>> 0) + ((u[p + 1] << 16) | (u[p + 2] << 8) | u[p + 3]);
export const u32le = (u, p) => ((u[p + 3] << 24) >>> 0) + ((u[p + 2] << 16) | (u[p + 1] << 8) | u[p]);
export const u64be = (u, p) => u32be(u, p) * 4294967296 + u32be(u, p + 4);
// ID3v2 sizes are 4 x 7-bit.
export const synchsafe = (u, p) => ((u[p] & 127) << 21) | ((u[p + 1] & 127) << 14) | ((u[p + 2] & 127) << 7) | (u[p + 3] & 127);

export function indexOfBytes(u8, pat, from = 0, to = u8.length) {
  const first = pat[0];
  const last = Math.min(to, u8.length) - pat.length;
  outer: for (let i = from; i <= last; i++) {
    if (u8[i] !== first) continue;
    for (let j = 1; j < pat.length; j++) if (u8[i + j] !== pat[j]) continue outer;
    return i;
  }
  return -1;
}

export const bytesOf = (s) => Uint8Array.from(s, (c) => c.charCodeAt(0));

const utf8 = new TextDecoder("utf-8", { fatal: false });
const latin = new TextDecoder("latin1");
const utf16 = new TextDecoder("utf-16le");
const utf16be = new TextDecoder("utf-16be");
export const decodeUtf8 = (u8) => utf8.decode(u8);
export const decodeLatin1 = (u8) => latin.decode(u8);
export function decodeUtf16(u8) {
  if (u8[0] === 0xfe && u8[1] === 0xff) return utf16be.decode(u8.subarray(2));
  if (u8[0] === 0xff && u8[1] === 0xfe) return utf16.decode(u8.subarray(2));
  return utf16.decode(u8);
}

export const clean = (s) => String(s).replace(/[\x00-\x08\x0b\x0c\x0e-\x1f]+/g, " ").trim();

export function sniffType(u8, name = "") {
  const a = ascii(u8, 0, 12);
  if (u8[0] === 0xff && u8[1] === 0xd8) return "jpeg";
  if (u8[0] === 0x89 && a.slice(1, 4) === "PNG") return "png";
  if (a.slice(0, 4) === "RIFF" && a.slice(8, 12) === "WEBP") return "webp";
  if (a.slice(0, 4) === "RIFF" && a.slice(8, 12) === "WAVE") return "wav";
  if (a.slice(0, 4) === "RIFF" && a.slice(8, 12) === "AVI ") return "avi";
  if (a.slice(0, 4) === "GIF8") return "gif";
  if (a.slice(4, 8) === "ftyp") {
    const brand = a.slice(8, 12);
    if (/^(heic|heix|hevc|mif1|msf1|heim|heis)$/.test(brand)) return "heic";
    if (/^avi[fs]$/.test(brand)) return "avif";
    if (/^(M4A |M4B |M4P )$/.test(brand)) return "m4a";
    if (/^qt  $/.test(brand)) return "mov";
    return "mp4";
  }
  if (a.slice(0, 3) === "ID3" || (u8[0] === 0xff && (u8[1] & 0xe0) === 0xe0)) return "mp3";
  if (a.slice(0, 4) === "fLaC") return "flac";
  if (a.slice(0, 4) === "OggS") return "ogg";
  if (u8[0] === 0x1a && u8[1] === 0x45 && u8[2] === 0xdf && u8[3] === 0xa3) return "webm";
  const ext = (/\.([a-z0-9]+)$/i.exec(name) || [])[1];
  return ext ? ext.toLowerCase() : "unknown";
}
