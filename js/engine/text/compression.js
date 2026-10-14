/* Compression-based similarity, ZipPy-style.
 *
 * deflate keeps a 32 KB window, so text that shares phrasing with a "prime"
 * passage compresses better when appended to it. We measure how many bytes
 * the sample costs after an AI-style prime versus on its own:
 *
 *   gain = 1 − (|deflate(prime + sample)| − |deflate(prime)|) / |deflate(sample)|
 *
 * Text written the way chat models write reuses the prime's phrasing and
 * gains more. Replaces the old CompressionDetector.simpleCompress, which
 * was a regex dictionary (O(n²)) rather than compression.
 *
 * The prime is our own writing (no licensed corpus), kept under 24 KB so
 * prime + an 8 KB sample fit the window.
 */
import { AI_PRIME } from "./prime.js";

const enc = new TextEncoder();

async function deflatedSize(str) {
  const cs = new CompressionStream("deflate-raw");
  const stream = new Blob([enc.encode(str)]).stream().pipeThrough(cs);
  const buf = await new Response(stream).arrayBuffer();
  return buf.byteLength;
}

let primeSize = null;

export async function compressionEvidence(text, wordCount) {
  if (typeof CompressionStream === "undefined") return null;
  if (wordCount < 150) return null; // too short to compress meaningfully
  const sample = text.replace(/\s+/g, " ").trim().slice(0, 8000);
  if (primeSize == null) primeSize = await deflatedSize(AI_PRIME);
  const [alone, joined] = await Promise.all([deflatedSize(sample), deflatedSize(AI_PRIME + " " + sample)]);
  const gain = 1 - (joined - primeSize) / alone;
  return {
    id: "compression", group: "statistics", label: "Compression similarity to AI prose",
    raw: +gain.toFixed(4),
    x: +((gain - 0.085) / 0.03).toFixed(3),
    detail: `Appending the text to a passage of typical chatbot prose saves ${(gain * 100).toFixed(1)}% of its compressed size. Human prose usually shares less phrasing with it.`
  };
}
