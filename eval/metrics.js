/* Metrics for a list of { label, prob } rows. */

export function auc(rows) {
  const pos = rows.filter((r) => r.label === "ai").map((r) => r.prob);
  const neg = rows.filter((r) => r.label === "human").map((r) => r.prob);
  if (!pos.length || !neg.length) return null;
  const all = [...pos.map((p) => [p, 1]), ...neg.map((p) => [p, 0])].sort((a, b) => a[0] - b[0]);
  // Mann–Whitney U with tie-averaged ranks.
  let rankSum = 0;
  for (let i = 0; i < all.length;) {
    let j = i;
    while (j < all.length && all[j][0] === all[i][0]) j++;
    const avg = (i + j + 1) / 2;
    for (let k = i; k < j; k++) if (all[k][1]) rankSum += avg;
    i = j;
  }
  return (rankSum - (pos.length * (pos.length + 1)) / 2) / (pos.length * neg.length);
}

export function at(rows, threshold) {
  const pos = rows.filter((r) => r.label === "ai"), neg = rows.filter((r) => r.label === "human");
  const tp = pos.filter((r) => r.prob >= threshold).length, fp = neg.filter((r) => r.prob >= threshold).length;
  return {
    threshold,
    tpr: pos.length ? tp / pos.length : null,
    fpr: neg.length ? fp / neg.length : null,
    precision: tp + fp ? tp / (tp + fp) : null,
    accuracy: (tp + neg.length - fp) / Math.max(1, rows.length)
  };
}

// Detection rate when the threshold is set so that at most `fpr` of human
// rows are flagged.
export function tprAtFpr(rows, fpr) {
  const neg = rows.filter((r) => r.label === "human").map((r) => r.prob).sort((a, b) => b - a);
  if (!neg.length) return null;
  const k = Math.floor(neg.length * fpr);
  const thr = k < neg.length ? neg[k] + 1e-9 : 0;
  return { ...at(rows, thr), targetFpr: fpr };
}

// Equal error rate: the point where the miss rate on AI rows equals the
// false-positive rate on human rows (the standard audio deepfake metric).
export function eer(rows) {
  const pos = rows.filter((r) => r.label === "ai").map((r) => r.prob).sort((a, b) => a - b);
  const neg = rows.filter((r) => r.label === "human").map((r) => r.prob).sort((a, b) => a - b);
  if (!pos.length || !neg.length) return null;
  let best = 1;
  for (const t of [...new Set([...pos, ...neg])]) {
    const fnr = lowerCount(pos, t) / pos.length, fpr = 1 - lowerCount(neg, t) / neg.length;
    best = Math.min(best, Math.max(fnr, fpr));
  }
  return best;
}

// Number of sorted values below t.
function lowerCount(a, t) {
  let lo = 0, hi = a.length;
  while (lo < hi) { const m = (lo + hi) >> 1; if (a[m] < t) lo = m + 1; else hi = m; }
  return lo;
}

export function summary(rows) {
  const slices = {};
  for (const r of rows) (slices[r.slice] = slices[r.slice] || []).push(r);
  const sliceStats = Object.fromEntries(Object.entries(slices).map(([k, v]) => {
    const flagged = v.filter((r) => r.prob >= 0.7).length;
    return [k, { n: v.length, label: v[0].label, flaggedAI: +(flagged / v.length).toFixed(3), meanProb: +(v.reduce((a, r) => a + r.prob, 0) / v.length).toFixed(3) }];
  }));
  const r3 = (x) => (x == null ? null : +x.toFixed(3));
  const fix = (o) => Object.fromEntries(Object.entries(o).map(([k, v]) => [k, typeof v === "number" ? r3(v) : v]));
  return {
    n: rows.length,
    nAI: rows.filter((r) => r.label === "ai").length,
    nHuman: rows.filter((r) => r.label === "human").length,
    auc: r3(auc(rows)),
    eer: r3(eer(rows)),
    atLikelyAI: fix(at(rows, 0.7)),
    at1pctFpr: fix(tprAtFpr(rows, 0.01) || {}),
    at5pctFpr: fix(tprAtFpr(rows, 0.05) || {}),
    slices: sliceStats
  };
}
