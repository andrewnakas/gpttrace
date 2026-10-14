/* Audio signal features for synthetic-speech / AI-music detection. Pure
 * functions over a mono Float32Array at the file's native sample rate.
 *
 *  bandwidth   where the long-term spectrum ends, and how abruptly. A neural
 *              vocoder running at 16/22.05/24 kHz has nothing above 8/11/12
 *              kHz; served in a 44.1 kHz file, that is a hard wall.
 *  silence     exact digital zeros and the noise floor between words. A
 *              microphone always hears a room; a TTS engine often doesn't.
 *  pitch       frame-to-frame f0 jitter in voiced speech; synthetic prosody
 *              is often smoother than a larynx.
 *  loudness    spread of voiced-frame level.
 *  tonal lines narrow, steady peaks in the upper band (vocoder / upsampler
 *              artifacts at fractions of the sample rate).
 */
import { fft1d, hann } from "../common/fft.js";

const db = (p) => 10 * Math.log10(p + 1e-20);

export function stft(x, sr, opts = {}) {
  const N = opts.n || (sr > 32000 ? 2048 : 1024), hop = N / 2;
  const win = hann(N);
  const frames = Math.max(0, Math.floor((x.length - N) / hop) + 1);
  const bins = N / 2 + 1;
  const power = new Float32Array(frames * bins), rms = new Float32Array(frames);
  const re = new Float64Array(N), im = new Float64Array(N);
  for (let f = 0; f < frames; f++) {
    let e = 0;
    for (let i = 0; i < N; i++) { const v = x[f * hop + i]; e += v * v; re[i] = v * win[i]; im[i] = 0; }
    rms[f] = Math.sqrt(e / N);
    fft1d(re, im);
    for (let k = 0; k < bins; k++) power[f * bins + k] = re[k] * re[k] + im[k] * im[k];
  }
  return { power, rms, frames, bins, N, hop, sr, hz: sr / N };
}

function percentile(arr, q) {
  const a = Float64Array.from(arr).sort();
  return a.length ? a[Math.min(a.length - 1, Math.floor(q * a.length))] : 0;
}

// Long-term average spectrum over active frames, smoothed, in dB.
function ltas(S, active) {
  const out = new Float64Array(S.bins);
  let n = 0;
  for (let f = 0; f < S.frames; f++) if (active[f]) { n++; for (let k = 0; k < S.bins; k++) out[k] += S.power[f * S.bins + k]; }
  for (let k = 0; k < S.bins; k++) out[k] = db(out[k] / Math.max(1, n));
  const sm = new Float64Array(S.bins), w = 3;
  for (let k = 0; k < S.bins; k++) { let s = 0, c = 0; for (let j = k - w; j <= k + w; j++) if (j >= 0 && j < S.bins) { s += out[j]; c++; } sm[k] = s / c; }
  return sm;
}

export function bandwidth(S, active) {
  const L = ltas(S, active);
  const hz = S.hz;
  // Reference level: the loudest 1–4 kHz region.
  let ref = -Infinity;
  for (let k = Math.round(300 / hz); k < Math.round(4000 / hz); k++) ref = Math.max(ref, L[k]);
  // Effective bandwidth: highest frequency still within 50 dB of the reference.
  let edge = 0;
  for (let k = S.bins - 1; k > 0; k--) if (L[k] > ref - 50) { edge = k * hz; break; }
  // Sharpest drop above 3 kHz across a ±400 Hz window (a "brick wall").
  const span = Math.max(2, Math.round(400 / hz));
  let drop = 0, at = 0;
  for (let k = Math.round(3000 / hz) + span; k < S.bins - span; k++) {
    const d = L[k - span] - L[k + span];
    if (d > drop) { drop = d; at = k * hz; }
  }
  // Energy left above the wall, relative to the reference.
  let above = -Infinity;
  if (at) for (let k = Math.round((at + 800) / hz); k < S.bins; k++) above = Math.max(above, L[k]);
  return { edgeHz: Math.round(edge), wallHz: Math.round(at), wallDb: +drop.toFixed(1), aboveWallDb: isFinite(above) ? +(above - ref).toFixed(1) : null, ltas: L };
}

export function silence(x, sr, S) {
  // Runs of exact (or near-exact) digital zero ≥ 50 ms.
  const minRun = Math.round(sr * 0.05);
  let zeroSamples = 0, run = 0, runs = 0;
  for (let i = 0; i < x.length; i++) {
    if (Math.abs(x[i]) < 3e-5) run++;
    else { if (run >= minRun) { zeroSamples += run; runs++; } run = 0; }
  }
  if (run >= minRun) { zeroSamples += run; runs++; }
  const lv = Array.from(S.rms, (r) => 20 * Math.log10(r + 1e-10));
  const floor = percentile(lv, 0.1), peak = percentile(lv, 0.95);
  return { digitalSilence: +(zeroSamples / x.length).toFixed(4), zeroRuns: runs, noiseFloorDb: +floor.toFixed(1), peakDb: +peak.toFixed(1), dynamicRangeDb: +(peak - floor).toFixed(1) };
}

// YIN-style f0 on 30 ms frames at ~8 kHz, 10 ms hop, for voiced speech statistics.
export function pitch(x, sr) {
  const ds = Math.max(1, Math.round(sr / 8000));
  const fs = sr / ds;
  const y = new Float32Array(Math.floor(x.length / ds));
  for (let i = 0; i < y.length; i++) { let s = 0; for (let j = 0; j < ds; j++) s += x[i * ds + j]; y[i] = s / ds; }
  const W = Math.round(fs * 0.03), hop = Math.round(fs * 0.01);
  const minLag = Math.floor(fs / 400), maxLag = Math.ceil(fs / 70);
  const f0 = [], lvl = [];
  const d = new Float64Array(maxLag + 1);
  for (let s = 0; s + W + maxLag < y.length && f0.length < 3000; s += hop) {
    let e = 0; for (let i = 0; i < W; i++) e += y[s + i] * y[s + i];
    const rmsDb = 10 * Math.log10(e / W + 1e-12);
    if (rmsDb < -45) { f0.push(0); lvl.push(rmsDb); continue; }
    let run = 0, best = 0;
    for (let tau = 1; tau <= maxLag; tau++) {
      let acc = 0;
      for (let i = 0; i < W; i++) { const v = y[s + i] - y[s + i + tau]; acc += v * v; }
      run += acc;
      d[tau] = run ? (acc * tau) / run : 1;
      if (tau >= minLag && !best && d[tau] < 0.15) {
        while (tau + 1 <= maxLag) { // walk to the local minimum
          let a2 = 0; for (let i = 0; i < W; i++) { const v = y[s + i] - y[s + i + tau + 1]; a2 += v * v; }
          const dn = (a2 * (tau + 1)) / (run + a2);
          if (dn >= d[tau]) break;
          run += a2; tau++; d[tau] = dn;
        }
        best = tau; break;
      }
    }
    f0.push(best ? fs / best : 0); lvl.push(rmsDb);
  }
  const voiced = f0.filter((v) => v > 0);
  const jit = [];
  for (let i = 1; i < f0.length; i++) if (f0[i] && f0[i - 1]) { const r = Math.abs(f0[i] - f0[i - 1]) / f0[i - 1]; if (r < 0.2) jit.push(r); }
  const vl = lvl.filter((_, i) => f0[i] > 0);
  const mean = (a) => a.reduce((p, c) => p + c, 0) / Math.max(1, a.length);
  const sd = (a) => { const m = mean(a); return Math.sqrt(mean(a.map((v) => (v - m) ** 2))); };
  return {
    voicedRatio: +(voiced.length / Math.max(1, f0.length)).toFixed(3),
    f0Median: voiced.length ? Math.round(percentile(voiced, 0.5)) : null,
    f0Range: voiced.length ? +(12 * Math.log2(percentile(voiced, 0.95) / percentile(voiced, 0.05))).toFixed(1) : null, // semitones
    jitter: jit.length >= 20 ? +percentile(jit, 0.5).toFixed(4) : null,
    loudnessSpreadDb: vl.length >= 20 ? +sd(vl).toFixed(2) : null,
    track: f0.length > 1500 ? f0.filter((_, i) => i % Math.ceil(f0.length / 1500) === 0) : f0
  };
}

export function tonalLines(S, active) {
  // Bins above 4 kHz whose level stands > 9 dB over the local median of the
  // LTAS and stays put in time (low frame-to-frame variance).
  const L = ltas(S, active);
  const k0 = Math.round(4000 / S.hz), lines = [];
  for (let k = k0; k < S.bins - 16; k++) {
    const nb = []; for (let j = k - 16; j <= k + 16; j++) if (Math.abs(j - k) > 2) nb.push(L[j]);
    nb.sort((a, b) => a - b);
    if (L[k] - nb[nb.length >> 1] > 9 && L[k] >= L[k - 1] && L[k] >= L[k + 1]) lines.push(Math.round(k * S.hz));
  }
  return { count: lines.length, hz: lines.slice(0, 12) };
}

// 900×256 log-spectrogram for display (Uint8, row 0 = top = Nyquist).
export function spectrogramImage(S, W = 900, H = 256) {
  const img = new Uint8ClampedArray(W * H * 4);
  let mx = -Infinity;
  const cols = new Float32Array(W * H);
  for (let x = 0; x < W; x++) {
    const f = Math.min(S.frames - 1, Math.floor((x / W) * S.frames));
    for (let y = 0; y < H; y++) {
      const k = Math.floor(((H - 1 - y) / H) * (S.bins - 1));
      const v = db(S.power[f * S.bins + k]);
      cols[y * W + x] = v; if (v > mx) mx = v;
    }
  }
  for (let i = 0; i < W * H; i++) {
    const t = Math.max(0, Math.min(1, (cols[i] - (mx - 90)) / 90));
    img[i * 4] = Math.round(255 * Math.min(1, t * 1.6)); img[i * 4 + 1] = Math.round(255 * Math.max(0, t * 1.4 - 0.4)); img[i * 4 + 2] = Math.round(255 * (0.3 + 0.7 * t * (1 - t) * 2)); img[i * 4 + 3] = 255;
  }
  return { width: W, height: H, data: img };
}

export function analyzeSignal(x, sr) {
  const S = stft(x, sr);
  const lv = Array.from(S.rms, (r) => 20 * Math.log10(r + 1e-10));
  const gate = percentile(lv, 0.95) - 35;
  const active = lv.map((v) => v > gate);
  return { sampleRate: sr, seconds: +(x.length / sr).toFixed(2), bandwidth: (({ ltas: _, ...b }) => b)(bandwidth(S, active)), silence: silence(x, sr, S), pitch: pitch(x, sr), tonal: tonalLines(S, active), spectrogram: spectrogramImage(S) };
}
