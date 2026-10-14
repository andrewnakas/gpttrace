/* "Pro: large cloud model" switch next to the deep-scan box.
 * It stays hidden and disabled unless this browser has signed in to Pro
 * (the non-secret gt_signed_in cookie) AND /api/me confirms an active plan,
 * so free visitors never even make the /api/me request. Off on every page
 * load; never remembered. */
let me = null;
const signedIn = () => /(?:^|;\s*)gt_signed_in=1(?:;|$)/.test(document.cookie);
const account = () => (me ||= fetch("/api/me", { credentials: "same-origin" }).then((r) => (r.ok ? r.json() : null)).catch(() => null));

// Toggling never starts a check by itself: the upload happens only on the
// next "Check" the visitor asks for.
export function bindPro(root) {
  const box = root.querySelector("[data-role=pro]"), wrap = root.querySelector("[data-role=pro-wrap]"), note = root.querySelector("[data-role=pro-note]");
  const state = { get on() { return !!(box && !box.disabled && box.checked); } };
  if (!box || !signedIn()) return state;
  account().then((a) => {
    if (!a || !a.pro) return;
    box.checked = false; box.disabled = false; wrap.hidden = false;
    box.addEventListener("change", () => { note.hidden = !box.checked; });
  });
  return state;
}
