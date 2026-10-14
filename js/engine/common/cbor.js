/* Minimal CBOR (RFC 8949) decoder — enough for C2PA claims, assertions and
 * COSE_Sign1 signatures. Byte strings stay Uint8Array; tags come back as
 * { tag, value }. Throws on truncated input. */

export function decodeCBOR(u8) {
  const dv = new DataView(u8.buffer, u8.byteOffset, u8.byteLength);
  let p = 0;
  const td = new TextDecoder();
  const BREAK = Symbol("break");

  function len(ai) {
    if (ai < 24) return ai;
    if (ai === 24) return u8[p++];
    if (ai === 25) { const v = dv.getUint16(p); p += 2; return v; }
    if (ai === 26) { const v = dv.getUint32(p); p += 4; return v; }
    if (ai === 27) { const v = dv.getUint32(p) * 4294967296 + dv.getUint32(p + 4); p += 8; return v; }
    if (ai === 31) return -1;
    throw new Error("cbor: bad length");
  }
  function half(h) {
    const e = (h >> 10) & 31, f = h & 1023, s = h & 32768 ? -1 : 1;
    if (e === 0) return s * 2 ** -14 * (f / 1024);
    if (e === 31) return f ? NaN : s * Infinity;
    return s * 2 ** (e - 15) * (1 + f / 1024);
  }
  function item() {
    if (p >= u8.length) throw new Error("cbor: truncated");
    const b = u8[p++], mt = b >> 5, ai = b & 31;
    if (mt === 7) {
      if (ai === 20) return false;
      if (ai === 21) return true;
      if (ai === 22) return null;
      if (ai === 23) return undefined;
      if (ai === 25) { const v = half(dv.getUint16(p)); p += 2; return v; }
      if (ai === 26) { const v = dv.getFloat32(p); p += 4; return v; }
      if (ai === 27) { const v = dv.getFloat64(p); p += 8; return v; }
      if (ai === 31) return BREAK;
      return ai < 24 ? ai : u8[p++];
    }
    const n = len(ai);
    switch (mt) {
      case 0: return n;
      case 1: return -1 - n;
      case 2:
      case 3: {
        if (n === -1) {
          const parts = [];
          for (;;) { const c = item(); if (c === BREAK) break; parts.push(c); }
          if (mt === 3) return parts.join("");
          const out = new Uint8Array(parts.reduce((a, c) => a + c.length, 0));
          let o = 0; parts.forEach((c) => { out.set(c, o); o += c.length; });
          return out;
        }
        if (p + n > u8.length) throw new Error("cbor: truncated string");
        const s = u8.subarray(p, p + n); p += n;
        return mt === 3 ? td.decode(s) : s;
      }
      case 4: {
        const a = [];
        if (n === -1) { for (;;) { const c = item(); if (c === BREAK) break; a.push(c); } }
        else for (let i = 0; i < n; i++) a.push(item());
        return a;
      }
      case 5: {
        const o = {};
        const put = () => { const k = item(); if (k === BREAK) return false; o[typeof k === "object" ? JSON.stringify(k) : k] = item(); return true; };
        if (n === -1) { while (put()); } else for (let i = 0; i < n; i++) put();
        return o;
      }
      case 6: return { tag: n, value: item() };
    }
    throw new Error("cbor: bad major type");
  }
  return item();
}

export const untag = (v) => (v && typeof v === "object" && "tag" in v && "value" in v ? v.value : v);
