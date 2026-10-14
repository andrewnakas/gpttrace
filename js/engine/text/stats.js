/* Sentence/word statistics (GPTTrace's statistics/statSignals). */

export function sentences(text) {
  return text
    .replace(/\s+/g, " ")
    .split(/(?<=[.!?…])\s+(?=[A-Z"'“‘(\[])/)
    .map((s) => s.trim())
    .filter(Boolean);
}

export function words(text) {
  return text.toLowerCase().match(/\b[a-z][a-z'\-]*\b/g) || [];
}

export function stddev(arr) {
  if (!arr.length) return 0;
  const m = arr.reduce((a, b) => a + b, 0) / arr.length;
  return Math.sqrt(arr.reduce((a, b) => a + (b - m) * (b - m), 0) / arr.length);
}

// Moving-average type/token ratio: plain TTR falls with length, so it can't
// be compared between a tweet and an essay.
function mattr(ws, win) {
  if (!ws.length) return 0;
  if (ws.length <= win) return new Set(ws).size / ws.length;
  const counts = new Map();
  let types = 0, sum = 0, n = 0;
  for (let i = 0; i < ws.length; i++) {
    const c = counts.get(ws[i]) || 0;
    if (!c) types++;
    counts.set(ws[i], c + 1);
    if (i >= win) {
      const old = ws[i - win], oc = counts.get(old);
      if (oc === 1) { counts.delete(old); types--; } else counts.set(old, oc - 1);
    }
    if (i >= win - 1) { sum += types / win; n++; }
  }
  return sum / n;
}

const STOP = new Set(("the a an and or but of to in on for with at by as is are was were be been being this that it its from not no into over " +
  "than then so such also more most very up out if we you he she they them our their his her i my me do does did have has had will would can could should may").split(" "));

export function statistics(text) {
  const sents = sentences(text);
  const ws = words(text);
  const sentLens = sents.map((s) => words(s).length).filter((n) => n > 0);
  const avgLen = sentLens.length ? sentLens.reduce((a, b) => a + b, 0) / sentLens.length : 0;
  const sdLen = stddev(sentLens);
  const freq = {};
  ws.forEach((w) => { if (!STOP.has(w) && w.length > 2) freq[w] = (freq[w] || 0) + 1; });
  // Paragraph-length uniformity: AI answers come in evenly sized blocks.
  const paras = text.split(/\n\s*\n/).map((p) => words(p).length).filter((n) => n >= 15);
  const paraCv = paras.length >= 3 ? stddev(paras) / (paras.reduce((a, b) => a + b, 0) / paras.length) : null;
  return {
    charCount: text.length,
    wordCount: ws.length,
    sentCount: sents.length,
    avgLen: +avgLen.toFixed(1),
    sdLen: +sdLen.toFixed(1),
    burstiness: avgLen > 0 ? +(sdLen / avgLen).toFixed(2) : 0,
    ttr: +mattr(ws, 100).toFixed(3),
    paraCv: paraCv == null ? null : +paraCv.toFixed(2),
    paraCount: paras.length,
    topRepeats: Object.entries(freq).sort((a, b) => b[1] - a[1]).slice(0, 5)
  };
}

const clip = (v, lo, hi) => Math.max(lo, Math.min(hi, v));

// Evidence items from the statistics. x > 0 leans AI.
export function statEvidence(st) {
  const out = [];
  if (st.wordCount < 80) return out;
  out.push({
    id: "burstiness", group: "statistics", label: "Sentence-length variation (burstiness)",
    x: clip((0.45 - st.burstiness) / 0.15, -2, 2),
    detail: `σ/μ of sentence length is ${st.burstiness}. People mix short and long sentences (typically 0.4–0.8); language models write more evenly (often under 0.35).`
  });
  out.push({
    id: "sentence-length", group: "statistics", label: "Average sentence length",
    x: clip((st.avgLen - 19) / 8, -1.5, 1.5),
    detail: `${st.avgLen} words per sentence on average.`
  });
  if (st.paraCv != null) out.push({
    id: "paragraph-uniformity", group: "statistics", label: "Paragraph-size uniformity",
    x: clip((0.35 - st.paraCv) / 0.2, -1.5, 1.5),
    detail: `${st.paraCount} paragraphs with a size variation of ${st.paraCv}. Chat answers tend to arrive in evenly sized blocks.`
  });
  if (st.wordCount >= 120) out.push({
    id: "vocabulary-richness", group: "statistics", label: "Vocabulary richness (type/token)",
    x: clip((st.ttr - 0.74) / 0.05, -1.5, 1.5),
    detail: `Moving-average type-token ratio ${st.ttr} (100-word window). Models over-vary word choice to avoid repeating themselves (“elegant variation”).`
  });
  return out;
}
