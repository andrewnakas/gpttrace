import { analyzeSignal } from "./features.js";

self.onmessage = (e) => {
  const { id, samples, sampleRate } = e.data;
  try {
    const r = analyzeSignal(samples, sampleRate);
    self.postMessage({ id, ok: true, result: r }, [r.spectrogram.data.buffer]);
  } catch (err) {
    self.postMessage({ id, ok: false, error: String((err && err.message) || err) });
  }
};
