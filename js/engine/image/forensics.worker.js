import { analyzePixels } from "./forensics.js";

self.onmessage = async (e) => {
  const { id, bitmap } = e.data;
  try {
    const r = await analyzePixels(bitmap);
    self.postMessage({ id, ok: true, result: r }, [r.ela.image, r.spectrum.image]);
  } catch (err) {
    self.postMessage({ id, ok: false, error: String((err && err.message) || err) });
  } finally {
    bitmap.close && bitmap.close();
  }
};
