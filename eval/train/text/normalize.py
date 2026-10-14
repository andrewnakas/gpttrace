"""Python mirror of js/engine/text/normalize.js — change both together.
Strips invisible format characters and maps Cyrillic/Greek lookalikes to
Latin in mostly-Latin text. eval/train/text/check_normalize.js checks the
two agree."""
import re
import unicodedata

HOMOGLYPHS = {
    "а": "a", "е": "e", "о": "o", "р": "p", "с": "c", "у": "y", "х": "x", "і": "i", "ј": "j", "ѕ": "s", "һ": "h", "ԁ": "d", "ԛ": "q", "ԝ": "w", "ӏ": "l",
    "А": "A", "В": "B", "Е": "E", "К": "K", "М": "M", "Н": "H", "О": "O", "Р": "P", "С": "C", "Т": "T", "Х": "X", "І": "I", "Ј": "J", "Ѕ": "S", "Ү": "Y", "Ԁ": "D", "Ԛ": "Q", "Ԝ": "W",
    "Α": "A", "Β": "B", "Ε": "E", "Ζ": "Z", "Η": "H", "Ι": "I", "Κ": "K", "Μ": "M", "Ν": "N", "Ο": "O", "Ρ": "P", "Τ": "T", "Υ": "Y", "Χ": "X",
    "ο": "o", "ν": "v", "ι": "i", "κ": "k", "ρ": "p", "τ": "t", "υ": "u", "χ": "x", "α": "a",
}
LOOKALIKE = re.compile("[" + "".join(HOMOGLYPHS) + "]")
GREEK_CYRILLIC = re.compile("[Ͱ-ϿЀ-ԯ]")
LATIN = re.compile("[A-Za-z]")
# \p{L}+ — Python's re has no \p, so: runs of letters (word chars minus digits/_).
LETTERS = re.compile(r"[^\W\d_]+")


def _invisible(c):
    return unicodedata.category(c) == "Cf" or "︀" <= c <= "️"


def _map(w):
    return LOOKALIKE.sub(lambda m: HOMOGLYPHS[m.group(0)], w)


def normalize_text(text):
    t = "".join(c for c in str(text) if not _invisible(c))
    look = LOOKALIKE.findall(t)
    if not look:
        return t
    foreign = len(GREEK_CYRILLIC.findall(t)) - len(look)
    if len(LATIN.findall(t)) <= foreign:
        return t
    return LETTERS.sub(lambda m: m.group(0) if GREEK_CYRILLIC.search(_map(m.group(0))) else _map(m.group(0)), t)
