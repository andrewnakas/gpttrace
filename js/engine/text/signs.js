/* The merged catalogue of text signs.
 *
 * Sources: GPTTrace's era-weighted vocabulary and structural detectors, and
 * the 21 Wikipedia "Signs of AI writing" rules from Ai_writing_detector
 * (wiki-data.json). Where the two overlapped (puffery, em dashes, rule of
 * three, negative parallelism, markdown, disclaimers, title case, chat
 * markup) they are one sign here, so nothing is counted twice. The old
 * single-word keyword lists ("important", "however", "most", "research",
 * "model"…) are gone: they fired on ordinary prose.
 *
 * Every sign returns match positions, so the highlighter can mark all of
 * them, and an `x` feature for the scorer (positive = leans AI).
 */

const WIKI = "https://en.wikipedia.org/wiki/Wikipedia:Signs_of_AI_writing";

const esc = (s) => s.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
const phraseRx = (list) => new RegExp("\\b(?:" + list.map((t) => esc(t).replace(/ /g, "\\s+")).join("|") + ")\\b", "gi");

function scan(text, rx, group = 0) {
  const out = [];
  rx.lastIndex = 0;
  let m;
  while ((m = rx.exec(text)) !== null) {
    if (m[0] === "") { rx.lastIndex++; continue; }
    const value = m[group] || m[0];
    const index = m.index + (group ? m[0].indexOf(value) : 0);
    out.push({ index, length: value.length, value });
  }
  return out;
}
const scanAll = (text, rxs) => rxs.flatMap((rx) => scan(text, rx));
const per1k = (n, ctx) => (n * 1000) / Math.max(ctx.words, 120);
const soft = (v) => Math.log1p(Math.max(0, v));

// ---- vocabulary (GPTTrace, era-weighted) -------------------------------
// Words that survive into the newest models carry the most weight. Generic
// words that people use constantly ("key", "landscape", "valuable",
// "comprehensive", "navigate") were dropped from the old list.
export const VOCAB = {
  current: ["emphasizing", "emphasize", "emphasizes", "enhance", "enhances", "enhancing", "highlighting", "showcasing", "showcases", "showcase", "underscoring", "underscores", "underscore", "align with", "aligns with", "aligning with"],
  recent: ["bolstered", "bolster", "bolsters", "crucial", "enduring", "fostering", "foster", "fosters", "pivotal", "vibrant", "intricate", "intricacies", "interplay", "nuanced", "seamless", "seamlessly", "elevate", "elevates", "resonate", "resonates", "commendable"],
  classic: ["delve", "delves", "delving", "tapestry", "testament", "meticulous", "meticulously", "garner", "garners", "garnered", "boasts", "realm", "realms", "myriad", "plethora", "embark", "embarks", "embarking", "unleash", "unlock the", "ever-evolving", "ever-changing", "paradigm shift", "synergy", "synergies", "holistic", "multifaceted", "navigate the complexities", "navigating the complexities", "valuable insights", "ever-evolving landscape", "dynamic landscape", "rapidly evolving", "game-changer", "game changer", "cutting-edge", "harness the power", "unwavering", "invaluable", "noteworthy", "profound", "transformative"]
};
const ERA_W = { current: 3.0, recent: 2.2, classic: 1.6 };
const VOCAB_RX = Object.fromEntries(Object.entries(VOCAB).map(([era, list]) => [era, phraseRx(list)]));

// ---- phrase lists (Wikipedia signs) ------------------------------------
const SIGNIFICANCE = [
  /\b(?:serves|stands) as (?:a |an )?(?:testament|reminder|beacon|symbol|cornerstone|hallmark|catalyst)\b/gi,
  /\b(?:is|remains) a (?:true |powerful |lasting )?testament to\b/gi,
  /\bplays? (?:a |an )?(?:vital|crucial|pivotal|key|significant|instrumental|central|essential) role\b/gi,
  /\bunderscor(?:es|ing) (?:the|its|their) (?:importance|significance|role|need)\b/gi,
  /\b(?:lasting|enduring|indelible|profound) (?:impact|legacy|mark|influence)\b/gi,
  /\b(?:a|the) (?:cornerstone|hallmark|linchpin|bedrock) of\b/gi,
  /\bof paramount importance\b|\bintegral part of\b|\bdeeply rooted\b|\bmarking a pivotal\b|\bpivotal moment\b/gi,
  /\breflects? (?:a |the )?broader\b|\bcontinues to (?:shape|inspire|resonate)\b|\bleft an indelible\b/gi,
  /\brich (?:history|tradition|heritage|cultural heritage|tapestry)\b/gi
];
const PROMOTIONAL = [
  /\bnestled (?:in|within|among|between)\b/gi,
  /\bin the heart of\b/gi,
  /\b(?:breathtaking|awe-inspiring|picturesque|captivating|enchanting|idyllic|world-class|must-visit|hidden gem|unparalleled|state-of-the-art|groundbreaking|stunning natural beauty)\b/gi,
  /\b(?:boasts|offers) (?:a |an )?(?:diverse|wide|rich|vibrant|stunning|impressive) (?:array|range|selection|variety)\b/gi,
  /\bdiverse array\b|\bseamlessly (?:blends|integrates|combines)\b|\bcommitment to excellence\b/gi
];
const EDITORIAL = [
  /\bit(?:'|’)?s (?:important|crucial|essential|worth|vital) (?:to )?(?:note|remember|consider|recognize|understand|mention|noting)\b/gi,
  /\bit is (?:important|crucial|essential|worth|vital) (?:to )?(?:note|remember|consider|recognize|understand|mention|noting)\b/gi,
  /\b(?:worth noting|it should be noted|it bears mentioning|needless to say|one must (?:note|consider))\b/gi
];
const CONNECTIVES = /\b(?:moreover|furthermore|additionally|consequently|nevertheless|in addition|notably|importantly|ultimately)\b,?/gi;
const ING_ANALYSIS = /,\s+(?:highlighting|underscoring|emphasizing|reflecting|showcasing|demonstrating|illustrating|signaling|signalling|solidifying|cementing|contributing to|ensuring|fostering|reinforcing|paving the way|marking)\b[^.!?\n]{0,80}/gi;
const COPULA = [
  /\b(?:serves|stands|acts|functions) as (?:a |an |the )/gi,
  /\b(?:boasts|features|represents|embodies|exemplifies) (?:a |an |the )/gi
];
const PARALLEL = [
  /\bnot only\b[^.,;!?\n]{2,50}?\bbut (?:also )?/gi,
  /\b(?:it(?:'|’)?s|this is|that(?:'|’)?s) not (?:just |merely |simply |only )?(?:about )?[^.,;!?\n]{2,40}?[,;—–-]\s*(?:it(?:'|’)?s|but)\b/gi,
  /\bnot (?:just|merely|simply) [^.,;!?\n]{2,40}?,? but\b/gi,
  /\b(?:more than just|goes beyond|extends beyond|rather than simply)\b/gi,
  /\bno [a-z]+, no [a-z]+(?:, no [a-z]+)?\b/gi
];
const VAGUE = [
  /\b(?:industry reports|observers have noted|experts (?:say|argue|believe|note|suggest|agree)|critics (?:argue|have noted|say)|analysts (?:believe|say|note))\b/gi,
  /\bstudies (?:have shown|show|suggest|indicate)\b|\bresearch (?:suggests|shows|indicates) that\b/gi,
  /\bit (?:is|has been) widely (?:believed|regarded|considered|recognized|acknowledged)\b/gi,
  /\b(?:many|some|various|numerous|several) (?:scholars|experts|researchers|critics|historians|observers|studies|sources)\b/gi,
  /\baccording to (?:experts|many|some)\b/gi
];
const SUPERLATIVES = [
  /\bone of the (?:most|best|largest|greatest|leading|finest)\b/gi,
  /\b(?:widely regarded as|considered (?:to be )?one of|recognized as one of|renowned for|celebrated for|hailed as)\b/gi
];
const CLOSERS = [
  /(?:^|[.!?]\s+|\n)(?:in conclusion|in summary|to sum up|to summarize|overall|ultimately|all in all)[,:]/gi,
  /\bin today(?:'|’)?s (?:fast-paced|digital|modern|ever-changing|rapidly evolving|interconnected) (?:world|age|era|landscape)\b/gi,
  /\b(?:throughout history|at the end of the day|only time will tell|as we (?:navigate|move forward|look to the future)|the future (?:looks|is) (?:bright|promising))\b/gi,
  /\b(?:by (?:doing so|embracing|leveraging|understanding)|whether you(?:'|’)?re a)\b/gi
];
const CHATBOT = [
  // Visible chain-of-thought copied from reasoning models (DeepSeek R1 style).
  /\b(?:okay|ok|alright|hmm+),? so the user (?:is asking|wants|asked|needs)\b|\bthe user (?:is asking|wants me to|asked me to)\b|\blet me (?:double-check|recall|think about this)\b/gi,
  /\b(?:i hope this helps|hope this helps|let me know if (?:you|there)|feel free to (?:ask|reach|let)|i(?:'|’)?d be happy to|i(?:'|’)?m happy to help|would you like me to|great question|happy to help|here(?:'|’)?s (?:a|an|the) (?:breakdown|summary|revised|rewritten|polished|quick|concise|detailed|step-by-step)|here is (?:a|an|the) (?:breakdown|summary|revised|rewritten|polished)|as requested|absolutely[!,]|certainly[!,])\b/gi
];
const DISCLAIMER = [
  /\bas an ai(?: language model| assistant| model)?\b/gi,
  /\bi(?:'|’)?m (?:just )?an ai\b|\bi am (?:just )?an ai\b/gi,
  /\bi (?:cannot|can(?:'|’)?t|am unable to|don(?:'|’)?t have the ability to) (?:browse|access real-time|provide real-time|assist with that)\b/gi,
  /\bas of my (?:last |latest )?(?:knowledge )?(?:update|cutoff|training)\b|\bmy (?:training data|knowledge cutoff)\b|\bup to my last (?:training|update)\b/gi,
  /\bi don(?:'|’)?t have (?:access to|real-time|personal (?:opinions|experiences))\b/gi
];
const CHAT_MARKUP = [
  /:?contentReference\[oaicite:\d+\]\{index=\d+\}/g,
  /\boaicite(?::\d+)?\b|\boai_citation\b|\bcontentReference\b/gi,
  /\bturn\d+(?:search|news|view|fetch|image)\d+\b/gi,
  /\bcite(?:turn\d+\w+)+/gi,
  /【\d+(?::\d+)?†[^】]{0,40}】/g,
  /[?&]utm_source=(?:chatgpt\.com|openai|copilot\.com|perplexity)/gi,
  /\bgrok_card\b|\battributableIndex\b|\battached_file\b|<\/?grok:render[^>]*>/gi
];
const PLACEHOLDER = [
  /\[(?:insert|your|add|company|client|recipient|name|date|link|source)[^\]\n]{0,40}\]/gi,
  /\[TODO[^\]\n]*\]|\blorem ipsum\b|\bPASTE (?:HERE|TEXT)\b/gi
];
const MARKDOWN = [
  /\*\*[^*\n]{2,80}?\*\*/g,
  /(?:^|\n)#{1,6}\s+\S[^\n]*/g,
  /(?:^|\n)\s*[-*•]\s+\*\*[^*\n]+\*\*\s*[:：]/g,
  /\[[^\]\n]{1,80}\]\(https?:\/\/[^)\s]+\)/g,
  /```[\s\S]*?```/g
];
const EMOJI_BULLETS = /(?:^|\n)\s*(?:[✅❌✔️🚀🔍📌💡⚡🎯🔥✨📈🧠👉🛠️⚙️🌟📊💬🔑📝]|\p{Extended_Pictographic})️?\s+\S/gu;
// Invisible / format characters. The em dash is NOT one of them (the old
// sign-19 regex counted every em dash twice). A ZWJ inside an emoji
// sequence is normal and is excluded below.
const INVISIBLE = /[​‌‍⁠⁡⁢⁣⁤﻿­  ᠎‎‏‪-‮]/g;

export const INVISIBLE_NAMES = {
  "​": "ZWSP", "‌": "ZWNJ", "‍": "ZWJ", "⁠": "WJ", "⁡": "FA", "⁢": "IT", "⁣": "IS", "⁤": "IP",
  "﻿": "BOM", "­": "SHY", " ": "NNBSP", " ": "EMSP", "᠎": "MVS", "‎": "LRM", "‏": "RLM",
  "‪": "LRE", "‫": "RLE", "‬": "PDF", "‭": "LRO", "‮": "RLO"
};

// ---- the catalogue -----------------------------------------------------
// group: vocabulary | tone | structure | content | artifacts | typography
export const SIGNS = [
  {
    id: "ai-vocabulary", group: "vocabulary", name: "AI vocabulary", sev: "med", wiki: WIKI + "#Language_and_tone",
    desc: "Words that language models over-use — “delve”, “tapestry”, “pivotal”, “underscore”, “showcasing” — weighted by how recently they became tells.",
    find(text) {
      const out = [];
      for (const era of Object.keys(VOCAB_RX)) scan(text, VOCAB_RX[era]).forEach((m) => out.push({ ...m, era }));
      return out;
    },
    x(m, ctx) { return soft(per1k(m.reduce((a, b) => a + ERA_W[b.era], 0), ctx) / 6); }
  },
  {
    id: "significance", group: "tone", name: "Undue emphasis on significance", sev: "high", wiki: WIKI + "#Undue_emphasis_on_symbolism_and_importance",
    desc: "“Stands as a testament”, “plays a pivotal role”, “left an indelible mark” — inflating the importance of ordinary subjects.",
    find: (t) => scanAll(t, SIGNIFICANCE), x: (m, c) => soft(per1k(m.length, c) / 1.5)
  },
  {
    id: "promotional", group: "tone", name: "Promotional language", sev: "med", wiki: WIKI + "#Promotional_language",
    desc: "Travel-brochure adjectives: “nestled in the heart of”, “breathtaking”, “boasts a diverse array”.",
    find: (t) => scanAll(t, PROMOTIONAL), x: (m, c) => soft(per1k(m.length, c) / 2)
  },
  {
    id: "editorial", group: "tone", name: "Editorializing (“it’s important to note”)", sev: "med", wiki: WIKI + "#Editorializing",
    desc: "Telling the reader what matters instead of showing it.",
    find: (t) => scanAll(t, EDITORIAL), x: (m, c) => soft(per1k(m.length, c) / 1.5)
  },
  {
    id: "connectives", group: "tone", name: "Over-used connectives", sev: "low", wiki: WIKI + "#Overuse_of_conjunctions",
    desc: "“Moreover”, “Furthermore”, “Additionally” opening sentence after sentence.",
    find: (t) => scan(t, CONNECTIVES), x: (m, c) => soft((m.length / Math.max(c.sentences, 4) - 0.04) * 12)
  },
  {
    id: "ing-analysis", group: "structure", name: "Superficial “-ing” analysis", sev: "high", wiki: WIKI + "#Superficial_analyses",
    desc: "Sentences that end in a tacked-on participle clause: “…, highlighting its importance”, “…, underscoring the need for…”.",
    find: (t) => scan(t, ING_ANALYSIS), x: (m, c) => soft((m.length / Math.max(c.sentences, 4)) * 10)
  },
  {
    id: "rule-of-three", group: "structure", name: "Rule of three", sev: "low", wiki: WIKI + "#Rule_of_three",
    desc: "Lists of exactly three — “innovation, collaboration, and growth” — far more often than people write them.",
    find: (t) => scan(t, /\b[\w-]+(?: [\w-]+)?, [\w-]+(?: [\w-]+)?,? and [\w-]+\b/gi),
    x: (m, c) => soft((m.length / Math.max(c.sentences, 4) - 0.08) * 8)
  },
  {
    id: "negative-parallelism", group: "structure", name: "Negative parallelism (“not just X, but Y”)", sev: "high", wiki: WIKI + "#Negative_parallelisms",
    desc: "“It’s not just a tool — it’s a revolution.” A signature cadence of chat models.",
    find: (t) => scanAll(t, PARALLEL), x: (m, c) => soft(per1k(m.length, c) / 1.5)
  },
  {
    id: "copula-avoidance", group: "structure", name: "Avoiding “is” / “are”", sev: "low", wiki: WIKI + "#Avoidance_of_basic_copulatives",
    desc: "“Serves as”, “stands as”, “boasts”, “features” where a person would just write “is”.",
    find: (t) => scanAll(t, COPULA), x: (m, c) => soft(per1k(m.length, c) / 3)
  },
  {
    id: "vague-attribution", group: "content", name: "Vague attributions", sev: "med", wiki: WIKI + "#Vague_attributions_of_opinion",
    desc: "“Experts say”, “studies have shown”, “many historians” — claims attributed to nobody in particular.",
    find: (t) => scanAll(t, VAGUE), x: (m, c) => soft(per1k(m.length, c) / 2)
  },
  {
    id: "superlatives", group: "content", name: "Unsupported superlatives", sev: "low", wiki: WIKI + "#Peacock_terms",
    desc: "“One of the most…”, “widely regarded as…”, “renowned for…” without a source.",
    find: (t) => scanAll(t, SUPERLATIVES), x: (m, c) => soft(per1k(m.length, c) / 3)
  },
  {
    id: "summary-closers", group: "content", name: "Formulaic openers and closers", sev: "med", wiki: WIKI + "#Sections_that_summarize",
    desc: "“In conclusion,”, “In today’s fast-paced world”, “Whether you’re a beginner or…”, “only time will tell”.",
    find: (t) => scanAll(t, CLOSERS), x: (m) => soft(m.length * 0.8)
  },
  {
    id: "chatbot-phrases", group: "artifacts", name: "Chatbot conversation left in", sev: "high", wiki: WIKI + "#Collaborative_communication",
    desc: "“I hope this helps!”, “Let me know if…”, “Here’s a revised version:” — the assistant talking to its user.",
    find: (t) => scanAll(t, CHATBOT), x: (m) => soft(m.length * 1.5)
  },
  {
    id: "ai-disclaimer", group: "artifacts", name: "AI disclaimers and knowledge-cutoff talk", sev: "high", wiki: WIKI + "#Knowledge-cutoff_disclaimers_and_speculation_about_gaps_in_sources",
    desc: "“As an AI language model…”, “As of my last knowledge update…”.",
    find: (t) => scanAll(t, DISCLAIMER), x: (m) => (m.length ? 2 + soft(m.length) : 0)
  },
  {
    id: "chat-markup", group: "artifacts", name: "Chat-interface markup", sev: "high", wiki: WIKI + "#Markup_and_formatting_issues",
    desc: "Citation tokens that only exist in copied ChatGPT/Grok output: oaicite, contentReference, turn0search0, 【4†source】, utm_source=chatgpt.com.",
    find: (t) => scanAll(t, CHAT_MARKUP), x: (m) => (m.length ? 2.5 + soft(m.length) : 0)
  },
  {
    id: "placeholders", group: "artifacts", name: "Unfilled placeholders", sev: "high", wiki: WIKI + "#Placeholder_text",
    desc: "Template slots such as “[Insert date]”, “[Your Name]”, “[Company Name]”.",
    find: (t) => scanAll(t, PLACEHOLDER), x: (m) => soft(m.length * 1.5)
  },
  {
    id: "markdown", group: "artifacts", name: "Markdown in prose", sev: "med", wiki: WIKI + "#Markdown",
    desc: "**Bold** labels, ### headings and [links](…) pasted into places that don’t render Markdown.",
    find: (t) => scanAll(t, MARKDOWN), x: (m, c) => soft(per1k(m.length, c) / 4)
  },
  {
    id: "title-case", group: "artifacts", name: "Title-Case headings", sev: "low", wiki: WIKI + "#Title_case",
    desc: "Short headings With Every Word Capitalized, on their own lines.",
    find(text) {
      const out = [];
      const rx = /^[^\n]+$/gm;
      let m;
      while ((m = rx.exec(text))) {
        const l = m[0].trim();
        if (l.length < 4 || l.length > 80 || /[.!?:,;]$/.test(l)) continue;
        const w = l.replace(/^[#*\s\d.]+/, "").split(/\s+/);
        if (w.length < 3 || w.length > 10) continue;
        const caps = w.filter((x) => /^[A-Z][a-z]/.test(x) || /^(?:and|of|the|in|to|for|with|on|a|an|or|vs\.?)$/i.test(x)).length;
        const majors = w.filter((x) => /^[A-Z][a-z]/.test(x)).length;
        if (caps / w.length >= 0.9 && majors >= 3) out.push({ index: m.index, length: m[0].length, value: l });
      }
      return out;
    },
    x: (m) => (m.length >= 2 ? soft(m.length / 2) : 0)
  },
  {
    id: "emoji-bullets", group: "artifacts", name: "Emoji bullet points", sev: "med", wiki: WIKI + "#Emoji",
    desc: "Lines and headings that start with ✅ 🚀 💡 📌-style emoji — a chat-assistant formatting habit.",
    find: (t) => scan(t, EMOJI_BULLETS), x: (m) => (m.length >= 2 ? soft(m.length / 1.5) : 0)
  },
  {
    id: "invisible-chars", group: "typography", name: "Invisible Unicode characters", sev: "high", wiki: WIKI + "#Unusual_Unicode_characters",
    desc: "Zero-width spaces, word joiners, soft hyphens, narrow no-break spaces and direction marks. Invisible on screen, they survive copy-paste from some AI tools and “humanizers”.",
    find(text) {
      return scan(text, INVISIBLE).filter((m) => {
        if (m.value !== "‍") return true;
        // ZWJ between two emoji is how families/flags are built — not a tell.
        const around = text.slice(Math.max(0, m.index - 2), m.index) + text.slice(m.index + 1, m.index + 3);
        return !/\p{Extended_Pictographic}/u.test(around);
      });
    },
    x: (m) => (m.length ? 1.5 + soft(m.length / 2) : 0)
  },
  {
    id: "em-dash", group: "typography", name: "Em-dash density", sev: "med", wiki: WIKI + "#Em_dashes",
    desc: "Language models use the em dash (—) far more than most people type it.",
    find: (t) => scan(t, /—/g), x: (m, c) => (m.length >= 2 ? soft((per1k(m.length, c) - 2) / 3) : 0)
  },
  {
    id: "smart-punctuation", group: "typography", name: "Typographic punctuation", sev: "info", wiki: WIKI + "#Curly_quotation_marks_and_apostrophes",
    desc: "Curly quotes, en dashes and the single-character ellipsis. Also produced by word processors and phones, so it is weak evidence.",
    find: (t) => scan(t, /[“”‘’–…]/g), x: (m, c) => soft(per1k(m.length, c) / 25) * 0.5
  }
];

export function runSigns(text, ctx) {
  const results = [];
  for (const s of SIGNS) {
    let matches = [];
    try { matches = s.find(text, ctx); } catch (e) { matches = []; }
    if (!matches.length) continue;
    results.push({
      id: s.id, group: s.group, label: s.name, sev: s.sev, wiki: s.wiki, desc: s.desc,
      matches, count: matches.length, x: +s.x(matches, ctx).toFixed(3)
    });
  }
  return results;
}
