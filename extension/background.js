/* GPTTrace extension service worker: two context-menu items and the toolbar
 * button. It only routes. The check runs in result.html, an extension page
 * that can host the engine's module workers and WASM (a service worker
 * can't start workers). The selection or image URL is handed over in
 * chrome.storage.session, which lives in memory and is never synced. */
const RESULT = "result.html";
const MENU = { image: "gt-image", text: "gt-text" };

chrome.runtime.onInstalled.addListener(() => {
  chrome.contextMenus.removeAll(() => {
    chrome.contextMenus.create({ id: MENU.image, title: "Check image with GPTTrace", contexts: ["image"] });
    chrome.contextMenus.create({ id: MENU.text, title: "Check text with GPTTrace", contexts: ["selection"] });
  });
});

chrome.contextMenus.onClicked.addListener((info, tab) => { handleMenu(info, tab); });
chrome.action.onClicked.addListener(() => { openResult(null); });

async function handleMenu(info, tab) {
  if (info.menuItemId === MENU.image && info.srcUrl) return openResult({ kind: "image", src: info.srcUrl });
  if (info.menuItemId === MENU.text) return openResult({ kind: "text", text: await selectedText(info, tab) });
}
// The same path as a right-click, for tests driving the service worker.
self.handleMenu = handleMenu;

// info.selectionText has its line breaks collapsed, and paragraph structure
// feeds several text signals, so read the selection itself. activeTab grants
// this for the clicked tab only, at the moment of the click.
async function selectedText(info, tab) {
  if (tab && tab.id != null && tab.id >= 0) {
    try {
      const [r] = await chrome.scripting.executeScript({
        target: { tabId: tab.id, frameIds: [info.frameId || 0] },
        func: () => {
          const el = document.activeElement;
          if (el && (el.tagName === "TEXTAREA" || el.tagName === "INPUT") && el.selectionEnd > el.selectionStart)
            return el.value.slice(el.selectionStart, el.selectionEnd);
          return String(getSelection());
        }
      });
      if (r && typeof r.result === "string" && r.result.trim()) return r.result;
    } catch (_) { /* chrome:// pages, the Web Store and the PDF viewer can't be scripted */ }
  }
  return info.selectionText || "";
}

// One result window: reuse it if it is open, else open it.
async function openResult(job) {
  if (job) await chrome.storage.session.set({ job: { ...job, id: Date.now() + Math.random() } });
  const { win } = await chrome.storage.session.get("win");
  if (win != null) {
    try { await chrome.windows.update(win, { focused: true }); return; } catch (_) { /* closed */ }
  }
  const w = await chrome.windows.create({ url: RESULT, type: "popup", width: 520, height: 780, focused: true });
  await chrome.storage.session.set({ win: w.id });
}

chrome.windows.onRemoved.addListener(async (id) => {
  const { win } = await chrome.storage.session.get("win");
  if (win === id) await chrome.storage.session.remove(["win", "job"]);
});
