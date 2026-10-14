/* Audio engine: analyze(file) → { prob, verdict, evidence, header, tags,
 * signal }. Reads the container header first (decodeAudioData resamples and
 * would hide the native rate), decodes at the native rate, and measures the
 * first 60 seconds in a worker. */
import { readBytes, sniffType } from "../common/bytes.js";
import { readC2PA } from "../common/c2pa.js";
import { matchGenerators } from "../common/generators.js";
import { combine } from "../common/score.js";
import { workerRPC } from "../common/worker-rpc.js";
import { c2paEvidence, generatorEvidence } from "../common/provenance.js";
import { header, tags, STRONG_TAGS } from "./tags.js";

const clip = (v, lo, hi) => Math.max(lo, Math.min(hi, v));
let featW = null, clsW = null;
const features = () => (featW ||= workerRPC(new URL("./features.worker.js", import.meta.url)));
const classifier = () => (clsW ||= workerRPC(new URL("./classifier.worker.js", import.meta.url)));

// Native sample rates of common TTS / voice-clone / music models. A spectrum
// that ends at one of these Nyquists inside a higher-rate file means the
// audio was synthesised (or recorded) at the lower rate and upsampled.
const MODEL_NYQUIST = [8000, 11025, 12000, 16000];
// Lowpass a lossy encoder normally applies at a given bitrate (kHz).
function expectedLowpass(h) {
  if (h.lowpass) return h.lowpass;
  const br = h.bitrate;
  if (h.codec === "mp3" && br) return br <= 64 ? 11000 : br <= 96 ? 15000 : br <= 128 ? 16000 : br <= 160 ? 17500 : br <= 192 ? 19000 : 20000;
  if (h.codec === "aac" && br) return br <= 64 ? 12000 : br <= 96 ? 14000 : br <= 128 ? 16000 : 18000;
  if (h.codec === "opus") return 20000;
  if (h.codec === "vorbis") return br && br <= 96 ? 15000 : 18000;
  return null;
}

async function decode(u, sampleRate) {
  const rate = clip(sampleRate || 48000, 8000, 96000);
  const ctx = new OfflineAudioContext(1, 1, rate);
  const buf = await ctx.decodeAudioData(u.buffer.slice(u.byteOffset, u.byteOffset + u.byteLength));
  const len = Math.min(buf.length, Math.round(buf.sampleRate * 60));
  const mono = new Float32Array(len);
  for (let c = 0; c < buf.numberOfChannels; c++) { const d = buf.getChannelData(c); for (let i = 0; i < len; i++) mono[i] += d[i] / buf.numberOfChannels; }
  return { samples: mono, sampleRate: buf.sampleRate, duration: buf.duration, channels: buf.numberOfChannels };
}

export function signalEvidence(sig, h) {
  const ev = [];
  const b = sig.bandwidth, s = sig.silence, p = sig.pitch;
  const nyq = sig.sampleRate / 2;
  const lp = expectedLowpass(h);
  // Speech has pauses between phrases (wide level range) and is voiced only part
  // of the time; music is continuous. Speech-only checks need speech.
  const isSpeech = p.voicedRatio > 0.25 && p.voicedRatio < 0.9 && p.f0Median && p.f0Median < 350 && s.dynamicRangeDb > 20;
  // Band limit well below both the container Nyquist and the codec lowpass.
  if (b.wallDb > 20 && b.wallHz < nyq * 0.8 && (!lp || b.wallHz < lp * 0.85)) {
    const match = MODEL_NYQUIST.find((f) => Math.abs(b.wallHz - f) / f < 0.06);
    const phone = Math.abs(b.wallHz - 4000) < 300 || Math.abs(b.wallHz - 3400) < 300;
    ev.push({ id: "band-limit", group: "signal", label: match ? `Spectrum stops at ${(match / 1000).toFixed(match % 1000 ? 2 : 0)} kHz` : "Hard band limit", w: 0.8,
      x: match ? (match === 8000 ? 0.6 : 1.4) : phone ? 0 : 0.6, sev: match ? "med" : "low",
      detail: `The audio falls off a ${b.wallDb} dB cliff at ${(b.wallHz / 1000).toFixed(1)} kHz in a ${sig.sampleRate / 1000} kHz file${lp ? ` whose encoder would keep up to ~${(lp / 1000).toFixed(0)} kHz` : ""}. ${match ? `That is the Nyquist frequency of a ${(match * 2 / 1000).toFixed(match === 11025 ? 2 : 0)} kHz model — the rate many text-to-speech and voice-cloning systems run at — upsampled into a bigger file. Phone calls and old voice recorders can do this too.` : "Something limited the bandwidth before the final encode."}` });
  } else if (b.edgeHz > 15000) {
    ev.push({ id: "full-band", group: "signal", label: "Full-bandwidth audio", w: 0.6, x: -0.5, sev: "info", detail: `Content reaches ${(b.edgeHz / 1000).toFixed(1)} kHz with no artificial cliff.` });
  }
  if (s.digitalSilence > 0.02 && isSpeech) ev.push({ id: "digital-silence", group: "signal", label: "Pauses of perfect digital silence", w: 0.8, x: clip(s.digitalSilence * 10, 0.3, 2), sev: "med",
    detail: `${(s.digitalSilence * 100).toFixed(1)}% of the audio is exact zero (${s.zeroRuns} gaps). A microphone always picks up some room noise between words; synthesized speech and heavily gated edits do not.` });
  if (isSpeech) {
    ev.push({ id: "noise-floor", group: "signal", label: "Background noise floor", w: 0.5, x: clip((-62 - s.noiseFloorDb) / 12, -1.5, 1.5), sev: "low",
      detail: `The quietest 10% of the audio sits at ${s.noiseFloorDb} dBFS (dynamic range ${s.dynamicRangeDb} dB). Real recordings usually keep a floor of room tone above −70 dBFS.` });
    if (p.jitter != null) ev.push({ id: "pitch-jitter", group: "signal", label: "Pitch micro-variation", w: 0.4, x: clip((0.012 - p.jitter) / 0.006, -1.5, 1.5), sev: "low",
      detail: `Median frame-to-frame pitch change is ${(p.jitter * 100).toFixed(2)}% (median f0 ${p.f0Median} Hz). Voices wobble; some synthetic voices are smoother.` });
    if (p.f0Range != null) ev.push({ id: "pitch-range", group: "signal", label: "Intonation range", w: 0, measure: true, x: clip((8 - p.f0Range) / 4, -1.5, 1.5), detail: `${p.f0Range} semitones between the 5th and 95th pitch percentiles.` });
    if (p.loudnessSpreadDb != null) ev.push({ id: "loudness-spread", group: "signal", label: "Loudness evenness", w: 0.3, x: clip((5 - p.loudnessSpreadDb) / 2, -1.5, 1.5), sev: "low",
      detail: `Voiced segments vary by ${p.loudnessSpreadDb} dB. Text-to-speech tends to hold a very even level.` });
  }
  if (sig.tonal.count >= 3) ev.push({ id: "tonal-lines", group: "signal", label: "Steady tones in the upper band", w: 0.3, x: clip((sig.tonal.count - 2) / 4, 0, 1.5), sev: "low",
    detail: `${sig.tonal.count} narrow, constant peaks above 4 kHz (${sig.tonal.hz.slice(0, 6).map((f) => (f / 1000).toFixed(1) + " kHz").join(", ")}). Neural vocoders and upsamplers can leave these; so can electrical hum harmonics and synths.` });
  return { ev, isSpeech };
}

export async function analyze(file, opts = {}) {
  const stage = opts.onStage || (() => {});
  stage("Reading tags and Content Credentials…");
  const u = await readBytes(file);
  const type = sniffType(u, file.name);
  const h = header(u, type);
  const t = tags(u, type);
  const c2pa = readC2PA(u, type, t.c2paJumbf);
  const strongStr = t.list.filter((x) => STRONG_TAGS.test(x.key)).map((x) => `${x.name}: ${x.value}`);
  const weakStr = t.list.filter((x) => !STRONG_TAGS.test(x.key)).map((x) => `${x.name}: ${x.value}`);
  if (h.encoder) strongStr.push(`Encoder: ${h.encoder}`);
  const strong = matchGenerators(strongStr, "audio").concat(c2pa.generators || []);
  const weak = matchGenerators(weakStr, "audio").map((g) => ({ ...g, weak: true }));

  stage("Decoding audio…");
  let decoded = null, signal = null, decodeError = null;
  try { decoded = await decode(u, h.sampleRate); } catch (e) { decodeError = "This browser could not decode the file: " + (e.message || e); }
  if (decoded) {
    stage("Measuring bandwidth, silence and pitch…");
    const copy = decoded.samples;
    signal = await features().call({ samples: copy, sampleRate: decoded.sampleRate }, [copy.buffer]);
  }
  const evidence = [...c2paEvidence(c2pa, "recording"), ...generatorEvidence(strong, weak, "recording")];
  let isSpeech = false;
  if (signal) { const s = signalEvidence(signal, h); evidence.push(...s.ev); isSpeech = s.isSpeech; }
  // Neural check (opt-in, 96 MB): mono 16 kHz from Web Audio's resampler, the
  // same anti-aliased path the training data went through (resample_poly).
  if (opts.deep && decoded) {
    try {
      const d16 = await decode(u, 16000);
      const n = await classifier().call({ samples: d16.samples }, [d16.samples.buffer], opts.onModelProgress);
      const p = 1 / (1 + Math.exp(-n.logit));
      evidence.push({ id: "neural-audio", group: "model", label: "Neural voice-deepfake classifier", w: 0.5, x: +clip(n.logit, -8, 8).toFixed(3),
        detail: `A speech model trained on synthetic and real voices gives ${(p * 100).toFixed(1)}% across ${n.windows} four-second window${n.windows === 1 ? "" : "s"}. It is built for speech; music and sound effects are outside what it has seen.` });
    } catch (e) {
      evidence.push({ id: "neural-audio-error", group: "model", label: "Neural audio classifier unavailable", w: 0, x: 0, sev: "info", detail: e.message });
    }
  }
  if (h.codec === "wav" && h.sampleRate && [16000, 22050, 24000].includes(h.sampleRate) && h.channels === 1)
    evidence.push({ id: "tts-wav-format", group: "metadata", label: `Mono WAV at ${h.sampleRate / 1000} kHz`, w: 0.4, x: 0.6, sev: "low", detail: "The native output format of many speech-synthesis engines. Dictation apps and speech datasets use it too." });
  // With the neural check the weights were fitted together with it ("audio-deep").
  const deep = evidence.some((e) => e.id === "neural-audio");
  let key = deep ? "audio-deep" : "audio";
  // Pro cloud model: only when the visitor switched it on for this check.
  if (opts.pro) {
    stage("Checking with the large cloud model…");
    const { cloudEvidence, proWeightsKey } = await import("../common/cloud.js");
    const row = await cloudEvidence("audio", file, "audio-deep", "neural-audio");
    evidence.push(row);
    if (row.id === "cloud-audio") key = proWeightsKey("audio", "audio-deep");
  }
  const scored = combine("audio", evidence, { weights: opts.weights, lowData: !signal, weightsKey: key });
  return { ...scored, type, header: h, tags: t.list, c2pa, signal, isSpeech, duration: decoded && decoded.duration, decodeError };
}
