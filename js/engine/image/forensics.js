/* Pixel forensics on an ImageBitmap. Runs inside forensics.worker.js
 * (OffscreenCanvas) so the page stays responsive.
 *
 *  - ELA (GPTTrace computeELA): re-save as JPEG q0.85 and diff.
 *  - Spectrum (GPTTrace computeFFT): now on a NATIVE-resolution centre crop.
 *    Downscaling first, as both old codebases did, low-pass filters away
 *    the upsampling lattice the FFT is meant to find.
 *  - Ai_writing_detector pixel measurements: checkerboard, texture entropy,
 *    noise residual, histogram smoothness, duplicate-neighbour ratio.
 *  - NovelAI "stealth pnginfo" in the alpha channel's low bits.
 */
import { fft2d, logMagnitudeShifted, radialProfile } from "../common/fft.js";

const canvas = (w, h) => new OffscreenCanvas(w, h);

function pixels(bmp, sx, sy, sw, sh, dw, dh) {
  const c = canvas(dw, dh);
  const x = c.getContext("2d", { willReadFrequently: true });
  x.imageSmoothingQuality = "high";
  x.drawImage(bmp, sx, sy, sw, sh, 0, 0, dw, dh);
  return x.getImageData(0, 0, dw, dh);
}

export async function ela(bmp) {
  const W = Math.min(bmp.width, 1024), H = Math.round(bmp.height * (W / bmp.width));
  const c1 = canvas(W, H);
  const x1 = c1.getContext("2d", { willReadFrequently: true });
  x1.drawImage(bmp, 0, 0, W, H);
  const orig = x1.getImageData(0, 0, W, H).data;
  const blob = await c1.convertToBlob({ type: "image/jpeg", quality: 0.85 });
  const re = pixels(await createImageBitmap(blob), 0, 0, W, H, W, H).data;
  const vis = new ImageData(W, H);
  let sum = 0, sq = 0, max = 0;
  for (let i = 0; i < orig.length; i += 4) {
    const d = Math.max(Math.abs(orig[i] - re[i]), Math.abs(orig[i + 1] - re[i + 1]), Math.abs(orig[i + 2] - re[i + 2]));
    sum += d; sq += d * d; if (d > max) max = d;
    const v = Math.min(255, d * 14);
    vis.data[i] = vis.data[i + 1] = vis.data[i + 2] = v; vis.data[i + 3] = 255;
  }
  const n = W * H, mean = sum / n;
  const vc = canvas(W, H); vc.getContext("2d").putImageData(vis, 0, 0);
  return { mean: +mean.toFixed(2), variance: +(sq / n - mean * mean).toFixed(2), max, image: vc.transferToImageBitmap() };
}

function luminance(img) {
  const d = img.data, out = new Float64Array(img.width * img.height);
  for (let i = 0, j = 0; i < d.length; i += 4, j++) out[j] = 0.2126 * d[i] + 0.7152 * d[i + 1] + 0.0722 * d[i + 2];
  return out;
}

export function spectrum(bmp) {
  const side = Math.min(bmp.width, bmp.height);
  const N = side >= 512 ? 512 : side >= 256 ? 256 : 128;
  const sx = Math.floor((bmp.width - N) / 2), sy = Math.floor((bmp.height - N) / 2);
  const img = side >= 128 ? pixels(bmp, sx, sy, N, N, N, N) : pixels(bmp, 0, 0, bmp.width, bmp.height, N, N);
  const re = luminance(img), im = new Float64Array(N * N);
  let mean = 0; for (let i = 0; i < re.length; i++) mean += re[i]; mean /= re.length;
  for (let y = 0; y < N; y++) for (let x = 0; x < N; x++) {
    const w = (0.5 - 0.5 * Math.cos((2 * Math.PI * x) / (N - 1))) * (0.5 - 0.5 * Math.cos((2 * Math.PI * y) / (N - 1)));
    re[y * N + x] = (re[y * N + x] - mean) * w;
  }
  fft2d(re, im, N, N);
  const mag = logMagnitudeShifted(re, im, N);
  const radial = radialProfile(mag, N, 64);

  // Peak-to-mean in the 0.5–0.9 band (GPTTrace spike ratio).
  let mx = 0, s = 0, k = 0;
  for (let i = 32; i < 58; i++) { s += radial[i]; k++; if (radial[i] > mx) mx = radial[i]; }
  const spikeRatio = s ? mx / (s / k) : 0;
  const total = radial.reduce((a, b) => a + b, 0);
  const highFreqShare = total ? radial.slice(38).reduce((a, b) => a + b, 0) / total : 0;
  // Fall-off slope of log-magnitude vs radius: natural photos ≈ 1/f.
  let sxr = 0, sy2 = 0, sxx = 0, sxy = 0, n = 0;
  for (let i = 4; i < 60; i++) { const lx = Math.log(i); sxr += lx; sy2 += radial[i]; sxx += lx * lx; sxy += lx * radial[i]; n++; }
  const slope = (n * sxy - sxr * sy2) / (n * sxx - sxr * sxr);

  // Periodic peaks off the axes: upsampling / VAE lattices show up as bright
  // isolated points at regular positions. Score: strongest local maximum
  // (outside the low-frequency disc) vs the median of its radial ring.
  const c = N / 2;
  let peak = 0;
  for (let y = 2; y < N - 2; y++) for (let x = 2; x < N - 2; x++) {
    const r = Math.hypot(x - c, y - c) / c;
    if (r < 0.25 || r >= 0.98 || Math.abs(x - c) < 2 || Math.abs(y - c) < 2) continue;
    const v = mag[y * N + x];
    let isMax = true;
    for (let dy = -2; dy <= 2 && isMax; dy++) for (let dx = -2; dx <= 2; dx++) if ((dx || dy) && mag[(y + dy) * N + x + dx] >= v) { isMax = false; break; }
    if (!isMax) continue;
    const ring = radial[Math.min(63, (r * 64) | 0)];
    if (v - ring > peak) peak = v - ring;
  }

  let maxV = 0; for (let i = 0; i < mag.length; i++) if (mag[i] > maxV) maxV = mag[i];
  const vis = new ImageData(N, N);
  for (let i = 0; i < mag.length; i++) { const v = Math.min(255, (mag[i] / maxV) * 255); vis.data[i * 4] = vis.data[i * 4 + 1] = vis.data[i * 4 + 2] = v; vis.data[i * 4 + 3] = 255; }
  const vc = canvas(N, N); vc.getContext("2d").putImageData(vis, 0, 0);
  return { size: N, native: side >= 128, spikeRatio: +spikeRatio.toFixed(3), highFreqShare: +highFreqShare.toFixed(3), slope: +slope.toFixed(3), latticePeak: +peak.toFixed(3), image: vc.transferToImageBitmap() };
}

// Ai_writing_detector PixelAnalysisDetector measurements, on a native crop.
export function pixelStats(bmp) {
  const S = Math.min(512, bmp.width, bmp.height);
  const img = pixels(bmp, Math.floor((bmp.width - S) / 2), Math.floor((bmp.height - S) / 2), S, S, S, S);
  const d = img.data, W = S, H = S;
  const g = (x, y) => { const i = (y * W + x) * 4; return (d[i] + d[i + 1] + d[i + 2]) / 3; };
  let dup = 0, dupN = 0, cb = 0, cbN = 0, noise = 0, noiseN = 0;
  const hist = new Float64Array(256), rh = new Float64Array(256);
  for (let y = 1; y < H - 1; y++) {
    for (let x = 1; x < W - 1; x++) {
      const c = g(x, y);
      if (x % 2 === 0 && y % 2 === 0) {
        const i = (y * W + x) * 4, j = i + 4;
        if (Math.abs(d[i] - d[j]) + Math.abs(d[i + 1] - d[j + 1]) + Math.abs(d[i + 2] - d[j + 2]) < 3) dup++;
        dupN++;
      }
      const r = g(x + 1, y), b = g(x, y + 1), br = g(x + 1, y + 1);
      if (Math.abs(c - br) + Math.abs(r - b) < 10 && Math.abs(c - r) + Math.abs(c - b) > 20) cb++;
      cbN++;
      if (x % 3 === 0 && y % 3 === 0) {
        const nb = (g(x - 1, y - 1) + g(x, y - 1) + g(x + 1, y - 1) + g(x - 1, y) + r + g(x - 1, y + 1) + b + br) / 8;
        noise += Math.abs(c - nb); noiseN++;
      }
      hist[c | 0]++;
      rh[d[(y * W + x) * 4]]++;
    }
  }
  let entropy = 0; const tot = hist.reduce((a, b) => a + b, 0);
  for (const h of hist) if (h) { const p = h / tot; entropy -= p * Math.log2(p); }
  let smooth = 0; for (let i = 1; i < 256; i++) smooth += 1 / (1 + Math.abs(rh[i] - rh[i - 1])); smooth /= 255;
  // Combing in the histogram (empty bins between full ones) means the levels
  // were stretched after capture.
  let gaps = 0; for (let i = 1; i < 255; i++) if (!rh[i] && rh[i - 1] && rh[i + 1]) gaps++;
  return {
    duplicateRatio: +(dup / dupN).toFixed(4), checkerboardRatio: +(cb / cbN).toFixed(4),
    noise: +(noise / noiseN).toFixed(3), entropy: +entropy.toFixed(3), histSmoothness: +smooth.toFixed(4), histGaps: gaps, side: S
  };
}

// NovelAI writes its settings into the alpha LSBs, column-major, behind the
// ASCII magic "stealth_pnginfo" / "stealth_pngcomp".
export function stealthInfo(bmp) {
  const H = bmp.height, need = 15 * 8;
  if (bmp.width * H < need * 2) return null;
  const cols = Math.ceil(need / H) + 1;
  const img = pixels(bmp, 0, 0, Math.min(cols, bmp.width), H, Math.min(cols, bmp.width), H);
  let bits = "";
  for (let x = 0; x < img.width && bits.length < need; x++)
    for (let y = 0; y < H && bits.length < need; y++) bits += img.data[(y * img.width + x) * 4 + 3] & 1;
  let s = "";
  for (let i = 0; i < bits.length; i += 8) s += String.fromCharCode(parseInt(bits.slice(i, i + 8), 2));
  return /^stealth_png(info|comp)/.test(s) ? s : null;
}

export async function analyzePixels(bmp) {
  const [e, sp] = [await ela(bmp), spectrum(bmp)];
  return { ela: e, spectrum: sp, pixels: pixelStats(bmp), stealth: stealthInfo(bmp), width: bmp.width, height: bmp.height };
}
