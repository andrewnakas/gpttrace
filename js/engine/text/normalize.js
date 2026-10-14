/* Undoes cheap character-level evasion before the neural check: invisible
 * format characters (zero-width spaces, joiners, soft hyphens, BOM) and
 * Cyrillic/Greek letters that look like Latin ones. On RAID these two tricks
 * alone cut a RoBERTa detector from 99.7% to 70% (zero-width) and 89%
 * (homoglyph) detection. Lookalikes are only mapped when the text has more
 * Latin letters than Cyrillic/Greek letters without a Latin twin, so genuine
 * Russian or Greek text is left alone.
 * Mirrored exactly in eval/train/text/normalize.py — change both together. */

// Lookalike → Latin. Covers RAID's homoglyph attack and the common
// confusables from Unicode's confusables.txt that render identically.
export const HOMOGLYPHS = {
  "а": "a", "е": "e", "о": "o", "р": "p", "с": "c", "у": "y", "х": "x", "і": "i", "ј": "j", "ѕ": "s", "һ": "h", "ԁ": "d", "ԛ": "q", "ԝ": "w", "ӏ": "l",
  "А": "A", "В": "B", "Е": "E", "К": "K", "М": "M", "Н": "H", "О": "O", "Р": "P", "С": "C", "Т": "T", "Х": "X", "І": "I", "Ј": "J", "Ѕ": "S", "Ү": "Y", "Ԁ": "D", "Ԛ": "Q", "Ԝ": "W",
  "Α": "A", "Β": "B", "Ε": "E", "Ζ": "Z", "Η": "H", "Ι": "I", "Κ": "K", "Μ": "M", "Ν": "N", "Ο": "O", "Ρ": "P", "Τ": "T", "Υ": "Y", "Χ": "X",
  "ο": "o", "ν": "v", "ι": "i", "κ": "k", "ρ": "p", "τ": "t", "υ": "u", "χ": "x", "α": "a"
};
const LOOKALIKE = new RegExp("[" + Object.keys(HOMOGLYPHS).join("") + "]", "g");
const GREEK_CYRILLIC = /[\u0370-\u03FF\u0400-\u052F]/g;
// Unicode "Cf" format characters (zero-width space/joiners, LRM/RLM, word
// joiner, BOM, soft hyphen, …) plus the variation selectors.
const INVISIBLE = /[\p{Cf}\u{FE00}-\u{FE0F}]/gu;

export function normalizeText(text) {
  let t = String(text).replace(INVISIBLE, "");
  const look = t.match(LOOKALIKE);
  if (!look) return t;
  // Real Russian/Greek uses letters with no Latin twin (д, л, λ, …); an
  // attacked English text has Latin letters and lookalikes only.
  const foreign = (t.match(GREEK_CYRILLIC) || []).length - look.length;
  if ((t.match(/[A-Za-z]/g) || []).length <= foreign) return t;
  // Within a mostly-Latin text, leave words that contain a real Cyrillic or
  // Greek letter (a quoted "Москва") as they are.
  const map = (w) => w.replace(LOOKALIKE, (c) => HOMOGLYPHS[c]);
  return t.replace(/\p{L}+/gu, (w) => (/[\u0370-\u03FF\u0400-\u052F]/.test(map(w)) ? w : map(w)));
}
