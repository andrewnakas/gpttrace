/* /pro/ page: waitlist form, and (once sales are on, or for a signed-in
 * visitor) email sign-in, plan status and Stripe checkout. Uses fetch, never
 * a form submission. The Worker sets data-sitekey (Turnstile) and
 * data-sales on the [data-pro] element. */
const post = async (url, body) => {
  const r = await fetch(url, { method: "POST", credentials: "same-origin", headers: { "content-type": "application/json" }, body: JSON.stringify(body || {}) });
  const out = await r.json().catch(() => ({}));
  if (!r.ok) throw new Error((out.error && out.error.message) || `Request failed (${r.status}).`);
  return out;
};

// Turnstile (only when the Worker supplied a site key).
let ts = null;
function turnstile(sitekey) {
  return (ts ||= new Promise((resolve, reject) => {
    window.onGtTurnstile = () => resolve(window.turnstile);
    const s = document.createElement("script");
    s.src = "https://challenges.cloudflare.com/turnstile/v0/api.js?render=explicit&onload=onGtTurnstile";
    s.async = true; s.onerror = () => reject(new Error("The bot check failed to load."));
    document.head.appendChild(s);
  }));
}

export function mount(root) {
  const $ = (r) => root.querySelector(`[data-role="${r}"]`);
  const sitekey = root.dataset.sitekey, sales = root.dataset.sales === "on";
  const signedIn = /(?:^|;\s*)gt_signed_in=1(?:;|$)/.test(document.cookie);

  // Each form gets its own widget; tokens are single-use, so reset after use.
  const widgets = new Map();
  const token = async (form) => {
    if (!sitekey) return undefined;
    const t = await turnstile(sitekey);
    if (!widgets.has(form)) widgets.set(form, t.render(form.querySelector("[data-role=turnstile]"), { sitekey }));
    const id = widgets.get(form);
    for (let i = 0; i < 100 && !t.getResponse(id); i++) await new Promise((r) => setTimeout(r, 100));
    return t.getResponse(id) || undefined;
  };
  const reset = (form) => { if (widgets.has(form)) window.turnstile.reset(widgets.get(form)); };
  if (sitekey) root.querySelectorAll("form").forEach((f) => f.addEventListener("focusin", () => token(f).catch(() => {}), { once: true }));

  function bind(form, statusRole, send) {
    const status = $(statusRole);
    form.addEventListener("submit", async (e) => {
      e.preventDefault();
      const btn = form.querySelector("button[type=submit]");
      btn.disabled = true; status.textContent = "Sending…"; status.classList.remove("warn");
      try { status.textContent = await send(form, await token(form)); }
      catch (err) { status.textContent = err.message; status.classList.add("warn"); }
      finally { btn.disabled = false; reset(form); }
    });
  }

  bind($("waitlist"), "wl-status", async (f, turnstile) => {
    await post("/api/waitlist", { email: f.email.value, source: "pro-page", turnstile });
    f.reset();
    return "Thanks — you’re on the list. We’ll email you once when Pro launches, and that’s all.";
  });

  const params = new URLSearchParams(location.search);
  if (!sales && !signedIn) return {};
  $("account").hidden = false;
  const msg = $("account-status");
  if (params.get("signin") === "expired") msg.textContent = "That sign-in link has expired or was already used. Ask for a new one below.";
  if (params.get("checkout") === "success") msg.textContent = "Payment received. Pro switches on as soon as Stripe confirms it, usually within a few seconds.";
  if (params.has("signin") || params.has("checkout")) history.replaceState(null, "", location.pathname);

  bind($("signin"), "signin-status", async (f, turnstile) => {
    await post("/api/auth/start", { email: f.email.value, turnstile });
    return "Check your inbox for a sign-in link. It works once and expires in 15 minutes.";
  });

  const go = (url, body) => async () => { try { location.href = (await post(url, body)).url; } catch (e) { msg.textContent = e.message; } };
  $("buy-monthly").addEventListener("click", go("/api/checkout", { plan: "monthly" }));
  $("buy-yearly").addEventListener("click", go("/api/checkout", { plan: "yearly" }));
  $("portal").addEventListener("click", go("/api/billing-portal"));
  $("logout").addEventListener("click", async () => { await post("/api/auth/logout").catch(() => {}); location.reload(); });

  if (signedIn) fetch("/api/me", { credentials: "same-origin" }).then((r) => r.json()).then((me) => {
    if (!me.email) return;
    $("signin").hidden = true; $("me").hidden = false;
    $("me-email").textContent = me.email;
    $("me-plan").textContent = me.pro ? `Pro (${me.plan || "active"})${me.cancelAtPeriodEnd ? ", ends at the end of this period" : ""}` : "Free";
    $("me-usage").textContent = me.usage ? `${me.usage.today} of ${me.usage.limit} cloud checks used today` : "";
    $("buy").hidden = !(me.sales && !me.pro);
    $("portal").hidden = !(me.sales && me.pro);
  }).catch(() => {});
  return {};
}
