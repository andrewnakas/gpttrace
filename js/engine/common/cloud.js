/* GPTTrace Pro: the one place the engines may send content off the device.
 * Called only when the visitor has a Pro plan AND has switched on "Pro:
 * large cloud model" for this check (opts.pro). The free tools never import
 * a code path that reaches it otherwise. */
import { WEIGHTS } from "../weights.js";

const LABEL = { text: "Large cloud text model (Pro)", image: "Large cloud image model (Pro)", audio: "Large cloud voice model (Pro)" };

// kind: "text" | "image" | "audio"; payload: {text} for text, a File/Blob otherwise.
export async function cloudInfer(kind, payload) {
  const blob = payload instanceof Blob;
  const r = await fetch("/api/infer/" + kind, {
    method: "POST", credentials: "same-origin",
    headers: { "content-type": blob ? payload.type || "application/octet-stream" : "application/json" },
    body: blob ? payload : JSON.stringify(payload)
  });
  const out = await r.json().catch(() => ({}));
  if (!r.ok) throw new Error((out.error && out.error.message) || `Cloud check failed (${r.status}).`);
  return out; // {logit, model}
}

/* Runs the cloud check and returns its evidence row (or an error row with
 * no weight). The row is weighted under "<kind>-pro" when those weights
 * exist; until then it stands in for the on-device neural row, using that
 * row's weight in the deep weights (deepKey / neuralId). */
export async function cloudEvidence(kind, payload, deepKey, neuralId) {
  try {
    const { logit, model } = await cloudInfer(kind, payload);
    const W = WEIGHTS[kind + "-pro"] || WEIGHTS[deepKey] || {};
    const w = W.w && neuralId in W.w ? W.w[neuralId] : W.prior && neuralId in W.prior ? W.prior[neuralId] : 0.5;
    const p = 1 / (1 + Math.exp(-logit));
    return { id: "cloud-" + kind, group: "model", label: LABEL[kind], w, x: +Math.max(-8, Math.min(8, logit)).toFixed(3),
      detail: `${model} gives ${(p * 100).toFixed(1)}%. Sent to GPTTrace for this check only, processed in memory and not kept.` };
  } catch (e) {
    return { id: "cloud-" + kind + "-error", group: "model", label: "Cloud model unavailable", w: 0, x: 0, sev: "info", detail: e.message };
  }
}

// Weights key for a result that includes the cloud row.
export const proWeightsKey = (kind, deepKey) => (WEIGHTS[kind + "-pro"] ? kind + "-pro" : deepKey);
