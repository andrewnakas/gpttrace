/* C2PA viewer: any file → full manifest store, readable. */
import { esc, metricGrid, errorHTML, progressHTML, bindDropzone } from "./findings.js";

export function mount(root) {
  const $ = (r) => root.querySelector(`[data-role="${r}"]`);
  const dz = $("dropzone"), input = $("file"), result = $("result");
  bindDropzone(dz, input, handle);

  async function handle(file) {
    result.innerHTML = progressHTML("Reading the manifest store…");
    try {
      const [{ readBytes, sniffType }, { readC2PA }, { tags }, { digitalSourceType }] = await Promise.all([
        import("../engine/common/bytes.js"), import("../engine/common/c2pa.js"), import("../engine/audio/tags.js"), import("../engine/common/generators.js")]);
      const u = await readBytes(file);
      const type = sniffType(u, file.name);
      const geob = type === "mp3" ? tags(u, type).c2paJumbf : null;
      const c = readC2PA(u, type, geob);
      if (!c.present) {
        result.innerHTML = `<div class="verdict tone-mixed" style="--meter:0"><div class="verdict-body"><h3>No Content Credentials found</h3><p>${esc(file.name)} · ${type.toUpperCase()}</p></div></div>
          <p class="note">Most files have none — absence proves nothing. Social networks and messaging apps usually strip manifests on upload; download the original from its source and try again.${c.markers.length ? ` Traces found: ${esc(c.markers.join(", "))} — the manifest may be remote or damaged.` : ""}</p>`;
        return;
      }
      const a = c.active;
      const tone = c.aiGenerated ? "ai" : c.aiEdited ? "likely-ai" : "human";
      const title = c.aiGenerated ? "Content Credentials: made with generative AI" : c.aiEdited ? "Content Credentials: edited with generative AI" : "Content Credentials found";
      let h = `<div class="verdict tone-${tone}"><div class="verdict-body"><h3>${title}</h3><p>${esc(file.name)} · ${c.manifests.length} manifest${c.manifests.length > 1 ? "s" : ""} · found in ${esc(c.source)}</p></div></div>`;
      h += metricGrid([["Signed by", c.signedBy || "—"], ["Claim generator", a.claimGenerator || a.generators.join(", ") || "—"], ["Signature algorithm", (a.signer && a.signer.alg) || "—"], ["Certificate valid", a.signer && a.signer.cert ? `${a.signer.cert.notBefore} → ${a.signer.cert.notAfter}` : "—"]]);
      c.manifests.slice().reverse().forEach((m, i) => {
        h += `<div class="section-title">${i === 0 ? "Active manifest" : "Earlier manifest (ingredient)"}${m.title ? " — " + esc(m.title) : ""}</div><div class="findings">`;
        if (m.actions.length) h += m.actions.map((x) => {
          const d = digitalSourceType(x.digitalSourceType);
          return `<div class="finding"><div class="finding-head"><span class="sev ${d && d.ai ? "high" : "info"}">${esc(x.action.replace(/^c2pa\./, ""))}</span>${esc(x.softwareAgent || "")}</div>${d ? `<p class="finding-detail">${esc(d.label)}</p>` : ""}${x.when ? `<p class="finding-detail">${esc(x.when)}</p>` : ""}</div>`;
        }).join("");
        else h += `<div class="finding"><p class="finding-detail">No actions recorded.</p></div>`;
        h += `</div><ul class="finding-list">${m.assertions.map((x) => `<li>${esc(x)}</li>`).join("")}</ul>`;
        if (m.ingredients.length) h += `<p class="note">Ingredients: ${esc(m.ingredients.join(", "))}</p>`;
      });
      h += `<p class="note">GPTTrace reads the manifest structure and the signer’s certificate on your device. It does not verify the cryptographic signature or check the file’s hash against it; for a verified report use the official <a href="https://contentcredentials.org/verify" target="_blank" rel="noopener">Content Credentials Verify</a> tool.</p>`;
      result.innerHTML = h;
    } catch (e) {
      result.innerHTML = errorHTML("Could not read this file", e.message);
    }
  }
  return { handle };
}
