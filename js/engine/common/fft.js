/* Radix-2 FFT (GPTTrace's fft1d/fft2d) plus the spectral summaries both the
 * image and audio engines use. The radial profile replaces the
 * Ai_writing_detector FFTAnalyzer.computeRIO, which sampled an unshifted
 * spectrum (its "centre" was not DC). */

function bitReverse(x, bits) {
  let y = 0;
  for (let i = 0; i < bits; i++) { y = (y << 1) | (x & 1); x >>= 1; }
  return y;
}

const twiddles = new Map();
function table(n) {
  let t = twiddles.get(n);
  if (!t) {
    t = { cos: new Float64Array(n / 2), sin: new Float64Array(n / 2), rev: new Uint32Array(n) };
    for (let i = 0; i < n / 2; i++) { t.cos[i] = Math.cos((-2 * Math.PI * i) / n); t.sin[i] = Math.sin((-2 * Math.PI * i) / n); }
    const bits = Math.log2(n) | 0;
    for (let i = 0; i < n; i++) t.rev[i] = bitReverse(i, bits);
    twiddles.set(n, t);
  }
  return t;
}

export function fft1d(re, im) {
  const n = re.length;
  const t = table(n);
  for (let i = 0; i < n; i++) {
    const j = t.rev[i];
    if (j > i) { let a = re[i]; re[i] = re[j]; re[j] = a; a = im[i]; im[i] = im[j]; im[j] = a; }
  }
  for (let size = 2; size <= n; size <<= 1) {
    const half = size >> 1, step = n / size;
    for (let i = 0; i < n; i += size) {
      for (let j = 0, k = 0; j < half; j++, k += step) {
        const a = i + j, b = a + half;
        const wr = t.cos[k], wi = t.sin[k];
        const tr = wr * re[b] - wi * im[b];
        const ti = wr * im[b] + wi * re[b];
        re[b] = re[a] - tr; im[b] = im[a] - ti;
        re[a] += tr; im[a] += ti;
      }
    }
  }
}

export function fft2d(re, im, w, h) {
  const rr = new Float64Array(w), ri = new Float64Array(w);
  for (let y = 0; y < h; y++) {
    const o = y * w;
    for (let x = 0; x < w; x++) { rr[x] = re[o + x]; ri[x] = im[o + x]; }
    fft1d(rr, ri);
    for (let x = 0; x < w; x++) { re[o + x] = rr[x]; im[o + x] = ri[x]; }
  }
  const cr = new Float64Array(h), ci = new Float64Array(h);
  for (let x = 0; x < w; x++) {
    for (let y = 0; y < h; y++) { cr[y] = re[y * w + x]; ci[y] = im[y * w + x]; }
    fft1d(cr, ci);
    for (let y = 0; y < h; y++) { re[y * w + x] = cr[y]; im[y * w + x] = ci[y]; }
  }
}

// log|F| with DC moved to the centre (fftshift).
export function logMagnitudeShifted(re, im, n) {
  const mag = new Float32Array(n * n);
  for (let y = 0; y < n; y++) {
    for (let x = 0; x < n; x++) {
      const idx = ((y + n / 2) % n) * n + ((x + n / 2) % n);
      mag[y * n + x] = Math.log1p(Math.hypot(re[idx], im[idx]));
    }
  }
  return mag;
}

export function radialProfile(mag, n, bins = 64) {
  const sum = new Float64Array(bins), cnt = new Float64Array(bins);
  const c = n / 2;
  for (let y = 0; y < n; y++) {
    for (let x = 0; x < n; x++) {
      const r = Math.hypot(x - c, y - c) / c;
      if (r >= 1) continue;
      const b = (r * bins) | 0;
      sum[b] += mag[y * n + x]; cnt[b]++;
    }
  }
  const out = new Float64Array(bins);
  for (let i = 0; i < bins; i++) out[i] = cnt[i] ? sum[i] / cnt[i] : 0;
  return out;
}

export const hann = (n) => Float64Array.from({ length: n }, (_, i) => 0.5 - 0.5 * Math.cos((2 * Math.PI * i) / (n - 1)));
