/* Text tool: binds to the shell from content/templates (tool-text). */
import { esc, verdictHTML, evidenceHTML, metricGrid, actionsHTML, bindActions, reportText, progressHTML, errorHTML, pref } from "./findings.js";
import { MODELS, mb } from "../engine/models.js";
import { bindPro } from "./pro-toggle.js";

const SAMPLES = {
  ai: `Duluth, Minnesota, is a vibrant city nestled on the shores of Lake Superior, renowned for its rich maritime heritage and breathtaking natural beauty. It serves as a vital hub for the Great Lakes shipping industry, playing a pivotal role in the regional economy.

The city's iconic Aerial Lift Bridge stands as a testament to its engineering legacy, showcasing the ingenuity of early twentieth-century design. Moreover, Duluth boasts a diverse array of outdoor activities, from hiking and biking to kayaking and skiing, ensuring that visitors of all ages can find something to enjoy.

Furthermore, the city's commitment to sustainability and community engagement underscores its enduring appeal. It's not just a destination — it's an experience. Whether you're a history enthusiast, an outdoor adventurer, or simply seeking a peaceful retreat, Duluth offers a unique blend of culture, nature, and charm.

In conclusion, Duluth remains one of the most captivating cities in the Midwest, highlighting the profound connection between people and the natural world. I hope this helps!`,
  // Public domain: Jane Austen, Pride and Prejudice (1813), ch. 3. The previous
  // sample read as AI-written to every model we tested.
  human: `In a few days Mr. Bingley returned Mr. Bennet’s visit, and sat about ten minutes with him in his library. He had entertained hopes of being admitted to a sight of the young ladies, of whose beauty he had heard much; but he saw only the father. The ladies were somewhat more fortunate, for they had the advantage of ascertaining, from an upper window, that he wore a blue coat and rode a black horse. An invitation to dinner was soon afterwards despatched; and already had Mrs. Bennet planned the courses that were to do credit to her housekeeping, when an answer arrived which deferred it all. Mr. Bingley was obliged to be in town the following day, and consequently unable to accept the honour of their invitation, etc. Mrs. Bennet was quite disconcerted. She could not imagine what business he could have in town so soon after his arrival in Hertfordshire; and she began to fear that he might always be flying about from one place to another, and never settled at Netherfield as he ought to be. Lady Lucas quieted her fears a little by starting the idea of his`
};

let engine = null;
const loadEngine = () => (engine ||= import("../engine/text/index.js"));

export function mount(root) {
  const $ = (r) => root.querySelector(`[data-role="${r}"]`);
  const input = $("input"), result = $("result"), hl = $("highlight"), hlBody = $("highlight-body");
  const counts = { w: $("words"), c: $("chars") };
  const deep = $("deep");
  if (deep) {
    deep.checked = pref.get("deep-text", false);
    deep.addEventListener("change", () => { pref.set("deep-text", deep.checked); if (input.value.trim()) run(); });
    root.querySelector("[data-role=deep-size]").textContent = mb(MODELS.text.bytes);
  }
  const pro = bindPro(root);
  let last = null, seq = 0;

  const update = () => {
    const t = input.value;
    counts.c.textContent = t.length.toLocaleString();
    counts.w.textContent = ((t.match(/\b[\w'’-]+\b/g) || []).length).toLocaleString();
  };
  input.addEventListener("input", update);
  input.addEventListener("focus", loadEngine, { once: true });
  input.addEventListener("keydown", (e) => { if ((e.metaKey || e.ctrlKey) && e.key === "Enter") run(); });

  root.querySelectorAll("[data-sample]").forEach((b) => b.addEventListener("click", () => { input.value = SAMPLES[b.dataset.sample]; update(); run(); }));
  $("clear").addEventListener("click", () => { input.value = ""; update(); result.innerHTML = result.dataset.empty || ""; hl.hidden = true; input.focus(); });
  $("analyze").addEventListener("click", run);
  if (!result.dataset.empty) result.dataset.empty = result.innerHTML;

  async function run() {
    const my = ++seq; // only the latest run may render
    const text = input.value;
    if (!text.trim()) { result.innerHTML = result.dataset.empty; hl.hidden = true; return; }
    result.innerHTML = progressHTML("Analysing…");
    try {
      const { analyze, highlight } = await loadEngine();
      const r = await analyze(text, {
        deep: deep && deep.checked,
        pro: pro.on,
        onModelProgress: (p) => (result.innerHTML = progressHTML(`Downloading the neural text model (${mb(MODELS.text.bytes)}, once)…`, p))
      });
      if (my !== seq) return;
      last = r;
      const s = r.stats;
      const short = s.wordCount < 150 ? `<p class="note warn">Only ${s.wordCount} words. Every detector is unreliable on short text; 150+ words is the minimum for a meaningful estimate.</p>` : "";
      result.innerHTML = verdictHTML(r, `${s.wordCount.toLocaleString()} words`) + short +
        metricGrid([["Sentences", s.sentCount], ["Avg sentence", s.avgLen + " w"], ["Burstiness", s.burstiness], ["Vocabulary", s.ttr]]) +
        (deep && !deep.checked ? `<p class="note">Standard check. Turn on <strong>Deep scan</strong> for a neural model that catches current chatbots far more often (on our test set it lifts accuracy from AUC 0.80 to 0.96).</p>` : "") +
        evidenceHTML(r) + actionsHTML() +
        `<p class="note">An estimate from writing patterns, not proof. People write like this too — especially in formal or translated English. Never use a detector score alone to accuse someone.</p>`;
      bindActions(result, () => reportText(r, "text analysis"), () => ({ ...r, signs: r.signs.map((x) => ({ ...x, matches: x.matches.slice(0, 50) })) }), "text");
      hlBody.innerHTML = highlight(text, r.signs);
      hl.hidden = !r.signs.length;
    } catch (e) {
      if (my === seq) result.innerHTML = errorHTML("Analysis failed", e.message);
    }
  }
  update();
  if (input.value.trim()) run();
  return { handle: null, setText(t) { input.value = t; update(); run(); } };
}
