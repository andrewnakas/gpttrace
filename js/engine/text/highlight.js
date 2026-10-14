/* Highlighted rendering of the analysed text (Ai_writing_detector's
 * highlightMatches + makeInvisibleCharsVisible, without the DOM dependency
 * and without rewriting every em dash as "[—]"). */
import { INVISIBLE_NAMES } from "./signs.js";

export function escapeHtml(s) {
  return String(s).replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" })[c]);
}

const INVIS_RX = new RegExp("[" + Object.keys(INVISIBLE_NAMES).join("") + "]", "g");

export function showInvisible(text) {
  return escapeHtml(text).replace(INVIS_RX, (c) =>
    `<span class="invisible-char" title="U+${c.codePointAt(0).toString(16).toUpperCase().padStart(4, "0")}">${INVISIBLE_NAMES[c]}</span>`);
}

export const showInvisiblePlain = (text) => text.replace(INVIS_RX, (c) => `[${INVISIBLE_NAMES[c]}]`);

export function highlight(text, signs) {
  const spans = [];
  for (const s of signs) for (const m of s.matches) spans.push({ start: m.index, end: m.index + m.length, ids: [s.id], labels: [s.label], group: s.group });
  spans.sort((a, b) => a.start - b.start || b.end - a.end);
  const merged = [];
  for (const h of spans) {
    const prev = merged[merged.length - 1];
    if (prev && h.start < prev.end) {
      prev.end = Math.max(prev.end, h.end);
      h.labels.forEach((l) => { if (!prev.labels.includes(l)) prev.labels.push(l); });
      h.ids.forEach((l) => { if (!prev.ids.includes(l)) prev.ids.push(l); });
    } else merged.push({ ...h, ids: [...h.ids], labels: [...h.labels] });
  }
  let out = "", last = 0;
  for (const h of merged) {
    out += showInvisible(text.slice(last, h.start));
    out += `<mark class="hl hl-${h.group}" data-signs="${h.ids.join(" ")}" title="${escapeHtml(h.labels.join(" · "))}">${showInvisible(text.slice(h.start, h.end))}</mark>`;
    last = h.end;
  }
  return out + showInvisible(text.slice(last));
}
