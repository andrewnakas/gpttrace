/* The one registry of AI generators, editors and provenance markers.
 *
 * Used twice: the engines match metadata strings against `match`, and the
 * page builder renders each landing page's "metadata signatures" table from
 * `signatures`. Keep claims honest — every signature carries `certainty`:
 *   documented  the vendor has said publicly that it writes this
 *   observed    commonly seen in files from this tool, but not promised
 *   possible    some exports / versions only; absence means nothing
 * Merged from GPTTrace's GENERATOR_SIGS and the Ai_writing_detector
 * MetadataDetector marker lists.
 */

export const GENERATORS = [
  // ---- image -------------------------------------------------------
  { id: "midjourney", label: "Midjourney", maker: "Midjourney, Inc.", kind: "ai", modalities: ["image", "video"],
    match: /\bmidjourney\b/i, page: "midjourney-detector",
    signatures: [
      { where: "PNG iTXt / XMP Description", what: "The prompt text followed by a “Job ID” on files downloaded from the web app", certainty: "observed" },
      { where: "XMP / IPTC", what: "Author or credit fields naming Midjourney on some exports", certainty: "possible" },
      { where: "Pixels", what: "No camera EXIF; Discord and social re-uploads strip all metadata", certainty: "observed" }
    ] },
  { id: "openai", label: "OpenAI (DALL·E / ChatGPT images / Sora)", maker: "OpenAI", kind: "ai", modalities: ["image", "video", "audio"],
    match: /\bopenai\b|\bdall[-\s·]?e\b|\bgpt-image\b|\bchatgpt\b/i, page: "chatgpt-image-detector",
    signatures: [
      { where: "C2PA manifest", what: "Content Credentials signed by OpenAI, with a c2pa.created action and digitalSourceType trainedAlgorithmicMedia", certainty: "documented" },
      { where: "Pixels", what: "Square or 3:2 sizes such as 1024×1024, 1536×1024 and 1024×1536", certainty: "observed" }
    ] },
  { id: "sora", label: "Sora", maker: "OpenAI", kind: "ai", modalities: ["video", "image"],
    match: /\bsora\b/i, page: "sora-detector",
    signatures: [
      { where: "C2PA manifest", what: "Content Credentials signed by OpenAI on downloaded videos", certainty: "documented" },
      { where: "Frames", what: "A visible, moving Sora watermark on videos from the Sora app", certainty: "documented" }
    ] },
  { id: "firefly", label: "Adobe Firefly", maker: "Adobe", kind: "ai", modalities: ["image", "video", "audio"],
    match: /\bfirefly\b|com\.adobe\.generative-ai/i, page: "adobe-firefly-detector",
    signatures: [
      { where: "C2PA manifest", what: "Content Credentials signed by Adobe, naming Firefly or Photoshop Generative Fill as the generator", certainty: "documented" },
      { where: "XMP", what: "xmpMM history and photoshop:Credit entries when exported through Photoshop", certainty: "possible" }
    ] },
  { id: "google", label: "Google Imagen / Gemini (Nano Banana)", maker: "Google", kind: "ai", modalities: ["image", "video", "audio"],
    match: /\bimagen\b|\bgemini\b|\bnano[\s-]?banana\b|made with google ai|google ai/i, page: "gemini-image-detector",
    signatures: [
      { where: "Pixels", what: "SynthID invisible watermark — only Google’s own tools can read it", certainty: "documented" },
      { where: "IPTC / XMP", what: "DigitalSourceType = trainedAlgorithmicMedia (“Made with Google AI”)", certainty: "documented" },
      { where: "C2PA manifest", what: "Content Credentials on images from recent Gemini and Pixel versions", certainty: "possible" },
      { where: "Pixels", what: "A small visible sparkle watermark in the corner on free-tier Gemini images", certainty: "observed" }
    ] },
  { id: "veo", label: "Google Veo", maker: "Google", kind: "ai", modalities: ["video"],
    match: /\bveo\b/i, page: "veo-detector",
    signatures: [
      { where: "Frames", what: "SynthID watermark in every frame (not readable outside Google)", certainty: "documented" },
      { where: "Frames", what: "Visible “Veo” watermark on most consumer exports", certainty: "observed" },
      { where: "C2PA / XMP", what: "Content Credentials or IPTC DigitalSourceType on some exports", certainty: "possible" }
    ] },
  { id: "stable-diffusion", label: "Stable Diffusion", maker: "Stability AI and open-source tools", kind: "ai", modalities: ["image"],
    match: /stable[\s-]?diffusion|\bsdxl\b|stable\.diffusion\.xl|\bsd\s?3(\.5)?\b|stability\.ai|dreamstudio/i, page: "stable-diffusion-detector",
    signatures: [
      { where: "PNG tEXt “parameters”", what: "Prompt, negative prompt, Steps, Sampler, CFG scale, Seed and Model hash (AUTOMATIC1111, Forge, SD.Next)", certainty: "documented" },
      { where: "PNG tEXt “prompt” / “workflow”", what: "The full node graph as JSON (ComfyUI)", certainty: "documented" },
      { where: "PNG tEXt “invokeai_metadata”", what: "Generation settings (InvokeAI)", certainty: "documented" },
      { where: "Pixels", what: "Sizes that are multiples of 64, e.g. 512×512, 768×768, 1024×1024, 832×1216", certainty: "observed" }
    ] },
  { id: "comfyui", label: "ComfyUI", maker: "ComfyUI", kind: "ai", modalities: ["image", "video"],
    match: /\bcomfyui\b|"class_type"\s*:/i, page: "stable-diffusion-detector", signatures: [] },
  { id: "a1111", label: "AUTOMATIC1111 / Forge", maker: "open source", kind: "ai", modalities: ["image"],
    match: /automatic1111|\ba1111\b|stable-diffusion-webui|\bforge\b.*\bsampler/i, page: "stable-diffusion-detector", signatures: [] },
  { id: "invokeai", label: "InvokeAI", maker: "Invoke", kind: "ai", modalities: ["image"],
    match: /\binvoke ?ai\b|invokeai_metadata/i, page: "stable-diffusion-detector", signatures: [] },
  { id: "fooocus", label: "Fooocus", maker: "open source", kind: "ai", modalities: ["image"],
    match: /\bfooocus\b/i, page: "stable-diffusion-detector", signatures: [] },
  { id: "novelai", label: "NovelAI", maker: "Anlatan", kind: "ai", modalities: ["image"],
    match: /\bnovelai\b/i, page: "stable-diffusion-detector",
    signatures: [
      { where: "PNG tEXt", what: "Software = NovelAI and a Comment JSON with the generation settings", certainty: "observed" },
      { where: "Alpha channel", what: "A “stealth pnginfo” copy of the settings hidden in the alpha channel’s low bits", certainty: "observed" }
    ] },
  { id: "flux", label: "FLUX", maker: "Black Forest Labs", kind: "ai", modalities: ["image"],
    match: /\bflux(?:\.1|\.2|-pro|-dev|-schnell|-kontext)?\b|black forest labs/i, page: "flux-detector",
    signatures: [
      { where: "PNG tEXt", what: "ComfyUI or Forge settings when run locally (model name contains “flux”)", certainty: "observed" },
      { where: "C2PA manifest", what: "Content Credentials on some hosted API outputs", certainty: "possible" },
      { where: "Pixels", what: "Often no metadata at all from hosted apps — the classifier carries the decision", certainty: "observed" }
    ] },
  { id: "ideogram", label: "Ideogram", maker: "Ideogram AI", kind: "ai", modalities: ["image"],
    match: /\bideogram\b/i, page: "ideogram-detector",
    signatures: [
      { where: "Pixels", what: "Unusually clean, legible typography — Ideogram’s strength — with no camera EXIF", certainty: "observed" },
      { where: "Metadata", what: "Little or no generator metadata on downloads", certainty: "observed" }
    ] },
  { id: "leonardo", label: "Leonardo.ai", maker: "Leonardo Interactive (Canva)", kind: "ai", modalities: ["image"],
    match: /\bleonardo(?:\.ai)?\b/i, page: "leonardo-ai-detector",
    signatures: [ { where: "Metadata", what: "Usually stripped; occasionally an XMP or C2PA entry naming Leonardo", certainty: "possible" } ] },
  { id: "grok", label: "Grok / Aurora (xAI)", maker: "xAI", kind: "ai", modalities: ["image", "video"],
    match: /\bgrok\b|\baurora\b.*\bxai\b|\bx\.ai\b|\bxai\b/i, page: "grok-image-detector",
    signatures: [ { where: "Metadata", what: "Usually none on images saved from X; the pixel classifier carries the decision", certainty: "observed" } ] },
  { id: "meta", label: "Meta AI (Imagine)", maker: "Meta", kind: "ai", modalities: ["image", "video"],
    match: /\bmeta ai\b|imagined with ai|\bimagine with meta\b/i, page: null,
    signatures: [
      { where: "IPTC / C2PA", what: "IPTC DigitalSourceType and C2PA labels on generated images", certainty: "documented" },
      { where: "Pixels", what: "A visible “Imagined with AI” watermark on many exports", certainty: "observed" }
    ] },
  { id: "microsoft", label: "Microsoft Designer / Bing Image Creator", maker: "Microsoft", kind: "ai", modalities: ["image"],
    match: /bing image creator|microsoft designer|\bcopilot\b/i, page: null,
    signatures: [ { where: "C2PA manifest", what: "Content Credentials signed by Microsoft", certainty: "documented" } ] },
  { id: "recraft", label: "Recraft", maker: "Recraft", kind: "ai", modalities: ["image"], match: /\brecraft\b/i, page: null, signatures: [] },
  { id: "seedream", label: "Seedream / Doubao (ByteDance)", maker: "ByteDance", kind: "ai", modalities: ["image", "video"],
    match: /\bseedream\b|\bseedance\b|\bdoubao\b|\bjimeng\b/i, page: null, signatures: [] },
  { id: "qwen", label: "Qwen-Image (Alibaba)", maker: "Alibaba", kind: "ai", modalities: ["image"], match: /\bqwen[-\s]?image\b|\bwan\s?2\.\d\b/i, page: null, signatures: [] },
  { id: "playground", label: "Playground", maker: "Playground", kind: "ai", modalities: ["image"], match: /\bplayground\s?(ai|v\d)\b/i, page: null, signatures: [] },
  { id: "nightcafe", label: "NightCafe", maker: "NightCafe", kind: "ai", modalities: ["image"], match: /\bnightcafe\b/i, page: null, signatures: [] },
  { id: "craiyon", label: "Craiyon", maker: "Craiyon", kind: "ai", modalities: ["image"], match: /\bcraiyon\b/i, page: null, signatures: [] },
  { id: "other-image-ai", label: "Other AI image tool", maker: "", kind: "ai", modalities: ["image"],
    match: /\bartbreeder\b|\bdeepai\b|\bwombo\b|\bstarryai\b|\bpixray\b|\bnmkd\b|\bsd\.next\b|\bdreamlike\b|\btensor\.art\b|\bcivitai\b|\bmage\.space\b/i, page: null, signatures: [] },
  { id: "synthid", label: "SynthID (named in metadata)", maker: "Google", kind: "ai", modalities: ["image", "video", "audio", "text"],
    match: /\bsynthid\b/i, page: "synthid-detector", signatures: [] },

  // ---- video ---------------------------------------------------------
  { id: "runway", label: "Runway", maker: "Runway", kind: "ai", modalities: ["video", "image"], match: /\brunway(ml)?\b|\bgen-?[34]\b/i, page: "runway-detector",
    signatures: [ { where: "C2PA manifest", what: "Content Credentials on Gen-3 and later exports", certainty: "possible" } ] },
  { id: "kling", label: "Kling", maker: "Kuaishou", kind: "ai", modalities: ["video"], match: /\bkling\b|\bkuaishou\b/i, page: "kling-detector",
    signatures: [ { where: "Frames", what: "A visible Kling watermark on free-tier videos", certainty: "observed" } ] },
  { id: "pika", label: "Pika", maker: "Pika Labs", kind: "ai", modalities: ["video"], match: /\bpika(\s?labs|\.art)?\b/i, page: null, signatures: [] },
  { id: "luma", label: "Luma Dream Machine", maker: "Luma AI", kind: "ai", modalities: ["video"], match: /\bluma\b|dream machine/i, page: null, signatures: [] },
  { id: "hailuo", label: "Hailuo / MiniMax", maker: "MiniMax", kind: "ai", modalities: ["video", "audio"], match: /\bhailuo\b|\bminimax\b/i, page: null, signatures: [] },
  { id: "heygen", label: "HeyGen / Synthesia (avatar video)", maker: "", kind: "ai", modalities: ["video"], match: /\bheygen\b|\bsynthesia\b|\bd-id\b/i, page: null, signatures: [] },

  // ---- audio ---------------------------------------------------------
  // Audio tag strings are matched only in encoder/software/comment fields,
  // never titles. Treat as "the file says so", not "the vendor always writes it".
  { id: "elevenlabs", label: "ElevenLabs", maker: "ElevenLabs", kind: "ai", modalities: ["audio"], match: /\beleven\s?labs\b/i, page: "elevenlabs-detector",
    signatures: [
      { where: "Audio", what: "Studio-clean speech with very low noise floor and few breaths", certainty: "observed" },
      { where: "Container", what: "MP3 at 44.1 kHz from the website; tags rarely name ElevenLabs", certainty: "observed" }
    ] },
  { id: "suno", label: "Suno", maker: "Suno", kind: "ai", modalities: ["audio", "video"], match: /\bsuno(\.ai|\.com)?\b/i, page: "suno-detector",
    signatures: [
      { where: "ID3 tags", what: "Comment or URL frames that mention suno.com on some downloads", certainty: "possible" },
      { where: "Audio", what: "Shimmering high-frequency artifacts and a soft upper band", certainty: "observed" }
    ] },
  { id: "udio", label: "Udio", maker: "Udio", kind: "ai", modalities: ["audio"], match: /\budio(\.com)?\b/i, page: "udio-detector",
    signatures: [ { where: "ID3 tags", what: "Occasionally a comment or URL naming udio.com", certainty: "possible" } ] },
  { id: "tts-other", label: "Other TTS / voice-clone tool", maker: "", kind: "ai", modalities: ["audio"],
    match: /\bplay\.?ht\b|\bmurf(\.ai)?\b|\bresemble\b|\bspeechify\b|\bwellsaid\b|\bdescript overdub\b|\bkokoro\b|\bpiper\b|\bcoqui\b|\bxtts\b|\bbark\b.*\bsuno\b|\bf5-tts\b|\bchatterbox\b|\bvall-?e\b|\btortoise-tts\b|\brvc\b|\bso-vits\b|\bsynthetic speech\b|\btext[- ]to[- ]speech\b/i,
    page: "ai-voice-detector", signatures: [] },
  { id: "lyria", label: "Google Lyria / MusicFX", maker: "Google", kind: "ai", modalities: ["audio"], match: /\blyria\b|\bmusicfx\b/i, page: null,
    signatures: [ { where: "Audio", what: "SynthID audio watermark — only Google’s tools can read it", certainty: "documented" } ] },

  // ---- editors: context, never proof ----------------------------------
  { id: "photoshop", label: "Adobe Photoshop", kind: "edit", modalities: ["image"], match: /\badobe photoshop\b/i, signatures: [] },
  { id: "lightroom", label: "Adobe Lightroom", kind: "edit", modalities: ["image"], match: /\blightroom\b/i, signatures: [] },
  { id: "gimp", label: "GIMP", kind: "edit", modalities: ["image"], match: /\bgimp\b/i, signatures: [] },
  { id: "affinity", label: "Affinity", kind: "edit", modalities: ["image"], match: /\baffinity (?:photo|designer)\b/i, signatures: [] },
  { id: "captureone", label: "Capture One", kind: "edit", modalities: ["image"], match: /\bcapture one\b/i, signatures: [] },
  { id: "canva", label: "Canva", kind: "edit", modalities: ["image", "video"], match: /\bcanva\b/i, signatures: [] },
  { id: "snapseed", label: "Snapseed", kind: "edit", modalities: ["image"], match: /\bsnapseed\b/i, signatures: [] },
  { id: "ffmpeg", label: "FFmpeg (re-encoded)", kind: "edit", modalities: ["video", "audio"], match: /\blavf\d|\blavc\d|\bffmpeg\b/i, signatures: [] },
  { id: "daw", label: "Audio editor / DAW", kind: "edit", modalities: ["audio"], match: /\baudacity\b|\badobe audition\b|\blogic pro\b|\bgarageband\b|\bableton\b|\bfl studio\b|\breaper\b|\bpro tools\b/i, signatures: [] },
  { id: "lame", label: "LAME MP3 encoder", kind: "edit", modalities: ["audio"], match: /\blame\s?\d\.\d+/i, signatures: [] }
];

// IPTC "Digital Source Type" vocabulary. The single most useful metadata
// field for AI provenance: Google, Meta, Adobe, OpenAI and Midjourney write it.
export const DIGITAL_SOURCE_TYPES = {
  trainedAlgorithmicMedia: { ai: true, label: "Created by generative AI (trainedAlgorithmicMedia)" },
  compositeWithTrainedAlgorithmicMedia: { ai: true, partial: true, label: "Edited with generative AI (compositeWithTrainedAlgorithmicMedia)" },
  algorithmicMedia: { ai: true, weak: true, label: "Algorithmically created (algorithmicMedia)" },
  compositeSynthetic: { ai: true, partial: true, label: "Composite including synthetic elements (compositeSynthetic)" },
  digitalCapture: { ai: false, label: "Digital capture (camera)" },
  negativeFilm: { ai: false, label: "Scanned negative film" },
  positiveFilm: { ai: false, label: "Scanned slide" },
  print: { ai: false, label: "Scanned print" },
  minorHumanEdits: { ai: false, label: "Camera capture with minor human edits" },
  compositeCapture: { ai: false, label: "Composite of captured images" },
  screenCapture: { ai: false, label: "Screenshot" },
  digitalCreation: { ai: false, label: "Created digitally by a person" },
  dataDrivenMedia: { ai: false, label: "Data-driven media" }
};

export function digitalSourceType(s) {
  const m = /digitalsourcetype\/([A-Za-z]+)/i.exec(s || "") || /^([A-Za-z]+)$/.exec(s || "");
  if (!m) return null;
  const key = Object.keys(DIGITAL_SOURCE_TYPES).find((k) => k.toLowerCase() === m[1].toLowerCase());
  return key ? { key, ...DIGITAL_SOURCE_TYPES[key] } : null;
}

export function matchGenerators(haystacks, modality) {
  const hits = new Map();
  for (const g of GENERATORS) {
    if (modality && !g.modalities.includes(modality)) continue;
    for (const h of haystacks) {
      if (!h || !g.match.test(h)) continue;
      if (!hits.has(g.id)) hits.set(g.id, { id: g.id, label: g.label, kind: g.kind, contexts: [] });
      const ctx = h.length > 140 ? h.slice(0, 137) + "…" : h;
      const hit = hits.get(g.id);
      if (!hit.contexts.includes(ctx)) hit.contexts.push(ctx);
    }
  }
  return [...hits.values()];
}

export const generatorById = (id) => GENERATORS.find((g) => g.id === id);
