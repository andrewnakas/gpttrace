/* One scoring model for every modality.
 *
 * Each engine emits evidence items { id, label, x, detail, ... } where x is a
 * signed, roughly unit-scale feature (positive = points toward AI). The
 * probability is a logistic model, P(AI) = sigmoid(bias + Σ w[id]·x), with
 * the weights in weights.js — written by eval/fit-weights.js from labelled
 * data, hand-set until there is data. An item with `decisive: "ai"` (signed
 * C2PA "created by AI", IPTC trainedAlgorithmicMedia, Stable Diffusion
 * settings) settles the verdict on its own.
 *
 * This replaces the six copies of the "≥70 / 50 / 30 / 15" ladder and the
 * hand-picked 0.45/0.30/0.25 and 70/30 blends of the two old codebases.
 */
import { WEIGHTS } from "../weights.js";

export const sigmoid = (z) => 1 / (1 + Math.exp(-z));

const NOUN = {
  text: { ai: "AI-written", human: "human-written" },
  image: { ai: "AI-generated", human: "authentic" },
  video: { ai: "AI-generated", human: "authentic" },
  audio: { ai: "AI-generated", human: "authentic" }
};

export function verdictFor(prob, modality, decisive) {
  const n = NOUN[modality] || NOUN.image;
  if (decisive) return { key: "ai", label: `${cap(n.ai)} (provenance)`, tone: "ai" };
  if (prob >= 0.92) return { key: "ai", label: cap(n.ai), tone: "ai" };
  if (prob >= 0.7) return { key: "likely-ai", label: `Likely ${n.ai}`, tone: "likely-ai" };
  if (prob > 0.3) return { key: "inconclusive", label: "Inconclusive", tone: "mixed" };
  if (prob > 0.08) return { key: "likely-human", label: `Likely ${n.human}`, tone: "likely-human" };
  return modality === "text"
    ? { key: "human", label: cap(n.human), tone: "human" }
    : { key: "human", label: "No AI signs found", tone: "human" };
}
const cap = (s) => s[0].toUpperCase() + s.slice(1);

export function combine(modality, evidence, opts = {}) {
  const W = (opts.weights || WEIGHTS)[opts.weightsKey || modality] || (opts.weights || WEIGHTS)[modality] || { bias: 0, w: {} };
  let z = W.bias;
  const items = evidence.map((e) => {
    // Fitted weight, else the hand-set prior (signs too rare to fit), else the item's own.
    const w = e.id in W.w ? W.w[e.id] : W.prior && e.id in W.prior ? W.prior[e.id] : e.w != null ? e.w : 1;
    const contrib = (e.x || 0) * w;
    z += contrib;
    return { ...e, contrib };
  });
  const decisive = items.find((e) => e.decisive === "ai");
  let prob = sigmoid(z);
  if (decisive) prob = Math.max(prob, 0.995);
  const verdict = verdictFor(prob, modality, !!decisive);
  const strength = Math.abs(z);
  let confidence = decisive ? "high" : strength > 3 ? "high" : strength > 1.5 ? "medium" : "low";
  if (opts.lowData) confidence = "low";
  items.sort((a, b) => (b.decisive ? 1 : 0) - (a.decisive ? 1 : 0) || Math.abs(b.contrib) - Math.abs(a.contrib));
  return { modality, prob, score: Math.round(prob * 100), logit: z, verdict, confidence, decisive: decisive ? decisive.id : null, evidence: items };
}

// Feature vector for the eval harness / weight fitting.
export const featureVector = (evidence) => Object.fromEntries(evidence.map((e) => [e.id, e.x || 0]));
