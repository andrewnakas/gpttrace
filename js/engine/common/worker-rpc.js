/* Promise wrapper around a module worker: call(msg, transfer) → result.
 * Messages with { type: "progress" } go to the onProgress callback. */
export function workerRPC(url) {
  let w = null, seq = 0;
  const pending = new Map();
  let onProgress = null;
  function ensure() {
    if (w) return w;
    w = new Worker(url, { type: "module" });
    w.onmessage = (e) => {
      const d = e.data;
      if (d.type === "progress") { onProgress && onProgress(d.p); return; }
      const p = pending.get(d.id);
      if (!p) return;
      pending.delete(d.id);
      d.ok ? p.resolve(d.result) : p.reject(new Error(d.error));
    };
    w.onerror = (e) => { for (const p of pending.values()) p.reject(new Error(e.message || "worker failed")); pending.clear(); w = null; };
    return w;
  }
  return {
    call(msg, transfer = [], progress) {
      onProgress = progress || null;
      const id = ++seq;
      return new Promise((resolve, reject) => { pending.set(id, { resolve, reject }); ensure().postMessage({ ...msg, id }, transfer); });
    },
    terminate() { if (w) w.terminate(); w = null; }
  };
}
