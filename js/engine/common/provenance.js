/* Provenance evidence shared by the image, audio and video engines: C2PA
 * manifests and generator names found in metadata. */

const CAMERA_SIGNERS = /leica|sony|nikon|canon|pixel camera|google pixel|samsung|fujifilm|truepic|proofmode/i;

export function c2paEvidence(c, noun = "file") {
  const ev = [];
  if (c.present && c.active) {
    const a = c.active;
    const who = c.signedBy ? ` Signed by ${c.signedBy}.` : "";
    const gen = [a.claimGenerator, ...a.generators].filter(Boolean)[0] || "unknown tool";
    if (c.aiGenerated) ev.push({ id: "c2pa-ai", group: "provenance", label: "Content Credentials: created by AI", decisive: "ai", x: 5, w: 1, sev: "high",
      detail: `The C2PA manifest records that this ${noun} was created with digital source type trainedAlgorithmicMedia (claim generator: ${gen}).${who}`, list: a.sourceTypes });
    else if (c.aiEdited) ev.push({ id: "c2pa-ai-edit", group: "provenance", label: "Content Credentials: edited with generative AI", x: 2.5, w: 1, sev: "high",
      detail: `The manifest history includes generative-AI steps (composite or AI-edited). Claim generator: ${gen}.${who}`, list: a.sourceTypes });
    else if (a.actions.some((x) => /digitalCapture/i.test(x.digitalSourceType)) || CAMERA_SIGNERS.test(c.signedBy + " " + gen))
      ev.push({ id: "c2pa-capture", group: "provenance", label: "Content Credentials: captured by a device", x: -3, w: 1, sev: "info",
        detail: `The signed manifest says this ${noun} was captured by a camera or recorder (${gen}).${who}` });
    else ev.push({ id: "c2pa-present", group: "provenance", label: "Content Credentials present", x: 0.3, w: 1, sev: "low",
      detail: `A C2PA manifest is embedded (claim generator: ${gen}).${who} Actions: ${a.actions.map((x) => x.action).join(", ") || "none listed"}.`, list: a.assertions.slice(0, 8) });
  } else if (c.markers && c.markers.length) {
    ev.push({ id: "c2pa-markers", group: "provenance", label: "C2PA traces (manifest not readable)", x: 1, w: 1, sev: "med", detail: "Byte markers of a Content Credentials manifest were found, but no complete manifest store could be parsed.", list: c.markers });
  }
  return ev;
}

export function generatorEvidence(strong, weak, noun = "file") {
  const ev = [];
  const sAI = strong.filter((g) => g.kind === "ai"), wAI = weak.filter((g) => g.kind === "ai" && !sAI.some((s) => s.id === g.id));
  if (sAI.length) ev.push({ id: "generator-software", group: "provenance", label: `Generator named in metadata: ${sAI.map((g) => g.label).join(", ")}`, decisive: "ai", x: 4, w: 1, sev: "high",
    detail: `A software, encoder or creator-tool field of this ${noun} names an AI generator.`, list: sAI.flatMap((g) => g.contexts).slice(0, 4) });
  if (wAI.length) ev.push({ id: "generator-mention", group: "provenance", label: `Metadata mentions ${wAI.map((g) => g.label).join(", ")}`, x: 2, w: 1, sev: "med",
    detail: "A description, comment or title field mentions an AI tool. That can be a credit line — or just a title about AI.", list: wAI.flatMap((g) => g.contexts).slice(0, 4) });
  const edit = [...strong, ...weak].filter((g) => g.kind === "edit");
  if (edit.length) ev.push({ id: "editor", group: "metadata", label: `Processed with ${[...new Set(edit.map((g) => g.label))].join(", ")}`, x: 0, w: 0, sev: "info", detail: "Editing and re-encoding tools are used on real and AI media alike." });
  return ev;
}
