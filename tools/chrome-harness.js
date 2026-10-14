/*
 * Headless Chrome for the checks that need Web Audio.
 *
 * The same few dozen lines as the top of check-fx.js: serve the repo, start
 * Chrome with a DevTools port, evaluate an expression in the page and hand
 * back its value. check-fx.js and check-project-link.js carry their own copies;
 * new checks use this one.
 *
 *   import { findChrome, withPage } from './chrome-harness.js';
 *   await withPage({ routes: { '/__x': html } }, async (page) => {
 *     await page.goto('/__x');
 *     const v = await page.eval('window.__run()');
 *     page.listen('Fetch.requestPaused', (p) => ...);   // any CDP event
 *   });
 */
import http from 'node:http';
import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import { spawn } from 'node:child_process';
import { ROOT, parseHeaders, resolve } from './serve.js';
export { ROOT };
const TYPES = {
  '.js': 'application/javascript', '.html': 'text/html', '.css': 'text/css', '.json': 'application/json',
  '.wav': 'audio/wav', '.flac': 'audio/flac', '.mp3': 'audio/mpeg', '.wasm': 'application/wasm', '.svg': 'image/svg+xml',
  '.mjs': 'application/javascript', '.onnx': 'application/octet-stream', '.jpg': 'image/jpeg', '.png': 'image/png', '.webp': 'image/webp', '.mp4': 'video/mp4'
};

export function findChrome() {
  return [
    process.env.CHROME,
    '/Applications/Google Chrome.app/Contents/MacOS/Google Chrome',
    '/usr/bin/google-chrome', '/usr/bin/chromium', '/usr/bin/chromium-browser'
  ].filter(Boolean).find((p) => fs.existsSync(p));
}

// routes: { '/path': string | Buffer | () => Buffer }. Everything else is
// served from the repo; an extensionless path gets '.html', like Pages does.
// fakeMedia: false leaves out the fake microphone/camera and the auto-accepted
// permission prompt, so a capture flag can pick a real source.
// args: extra Chrome switches (check-record-computer passes the ones that let
// getDisplayMedia pick a tab by its title without showing the picker).
// headers: apply _headers to repo files, as Pages does (needed for the
// cross-origin isolated pages). profile: a Chrome profile directory kept
// between runs (model caches), instead of a fresh temporary one; pair it
// with port.
export async function withPage(opts, fn) {
  const routes = opts.routes || {};
  const rules = opts.headers === false ? [] : parseHeaders();
  const server = http.createServer((req, res) => {
    const url = decodeURIComponent(req.url.split('?')[0]);
    if (Object.prototype.hasOwnProperty.call(routes, url)) {
      let body = routes[url];
      if (typeof body === 'function') body = body();
      const type = typeof body === 'string' ? 'text/html' : (TYPES[path.extname(url)] || 'application/octet-stream');
      // Open to any origin: a route can stand in for a CDN file the page
      // fetches with CORS (see check-fidelity's ffmpeg core).
      res.writeHead(200, { 'content-type': type, 'access-control-allow-origin': '*' });
      res.end(body);
      return;
    }
    const file = resolve(url);
    if (!file) { res.writeHead(404); res.end(); return; }
    const headers = { 'content-type': TYPES[path.extname(file)] || 'application/octet-stream' };
    rules.forEach((r) => { if (r.re.test(url)) r.headers.forEach(([k, v]) => { headers[k.toLowerCase()] = v; }); });
    headers['cache-control'] = 'no-store';
    res.writeHead(200, headers);
    fs.createReadStream(file).pipe(res);
  });
  // opts.port: a fixed port, so a kept profile's Cache Storage (per origin,
  // and the origin includes the port) still holds the models next run.
  await new Promise((r) => server.listen(opts.port || 0, '127.0.0.1', r));
  const port = server.address().port;
  const dir = opts.profile || fs.mkdtempSync(path.join(os.tmpdir(), 'gt-chk-'));
  // A killed Chrome leaves its Singleton* lock behind, and the next launch on
  // the same profile exits at once.
  if (opts.profile) {
    fs.mkdirSync(dir, { recursive: true });
    ['DevToolsActivePort', 'SingletonLock', 'SingletonSocket', 'SingletonCookie'].forEach((f) => { try { fs.unlinkSync(path.join(dir, f)); } catch (e) {} });
  }
  const chrome = spawn(findChrome(), ['--headless=new', '--enable-automation', '--remote-debugging-port=0', '--user-data-dir=' + dir,
    '--no-first-run', '--no-default-browser-check', '--autoplay-policy=no-user-gesture-required',
  ].concat(opts.fakeMedia === false ? [] : ['--use-fake-ui-for-media-stream', '--use-fake-device-for-media-stream'], opts.args || [], ['about:blank']), { stdio: 'ignore' });
  let ws;
  try {
    ws = await connect(dir);
    let id = 0;
    const pending = new Map();
    const logs = [];
    const listeners = {};
    ws.onmessage = (m) => {
      const d = JSON.parse(m.data);
      if (d.method && listeners[d.method]) listeners[d.method].forEach((fn) => fn(d.params));
      if (d.method === 'Runtime.exceptionThrown') logs.push('exception: ' + JSON.stringify(d.params.exceptionDetails).slice(0, 400));
      if (d.method === 'Runtime.consoleAPICalled' && d.params.type === 'error') logs.push('console.error: ' + d.params.args.map((a) => a.value || a.description).join(' '));
      if (d.method === 'Log.entryAdded' && d.params.entry.level === 'error') logs.push('log: ' + d.params.entry.text + ' ' + (d.params.entry.url || ''));
      if (pending.has(d.id)) { pending.get(d.id)(d); pending.delete(d.id); }
    };
    const send = (method, params) => new Promise((r) => { const n = ++id; pending.set(n, r); ws.send(JSON.stringify({ id: n, method, params })); });
    await send('Page.enable');
    await send('Runtime.enable');
    await send('Log.enable');
    const page = {
      port, logs, send,
      listen(method, fn) { (listeners[method] = listeners[method] || []).push(fn); },
      url: (p) => 'http://127.0.0.1:' + port + p,
      async goto(p, wait) {
        await send('Page.navigate', { url: page.url(p) });
        await new Promise((r) => setTimeout(r, wait || 800));
        // On a loaded machine the fixed wait is not enough for the scripts at
        // the bottom of the page; wait for the load as well (up to a minute).
        for (let i = 0; i < 300; i++) {
          const r = await send('Runtime.evaluate', { expression: 'document.readyState', returnByValue: true });
          if (r.result && r.result.result && r.result.result.value === 'complete') break;
          await new Promise((res) => setTimeout(res, 200));
        }
      },
      async eval(expression, timeout) {
        const res = await send('Runtime.evaluate', { expression, awaitPromise: true, returnByValue: true, timeout: timeout || 300000 });
        if (res.result && res.result.exceptionDetails) {
          const ex = res.result.exceptionDetails;
          throw new Error((ex.exception && ex.exception.description) || JSON.stringify(ex).slice(0, 800));
        }
        return res.result.result.value;
      }
    };
    return await fn(page);
  } finally {
    try { ws && ws.close(); } catch (e) {}
    chrome.kill();
    server.close();
    if (!opts.profile) try { fs.rmSync(dir, { recursive: true, force: true }); } catch (e) {}
  }
}

async function connect(dir) {
  const f = path.join(dir, 'DevToolsActivePort');
  // Two minutes by default; CHROME_WAIT_MS overrides. On a loaded machine
  // (load average 40-70) a headless start was measured at 69-72 s.
  const tries = Math.ceil((+process.env.CHROME_WAIT_MS || 120000) / 100);
  for (let i = 0; i < tries && !fs.existsSync(f); i++) await new Promise((r) => setTimeout(r, 100));
  const port = fs.readFileSync(f, 'utf8').split('\n')[0];
  let list = [];
  for (let i = 0; i < 50 && !list.some((t) => t.type === 'page'); i++) {
    list = await (await fetch('http://127.0.0.1:' + port + '/json/list')).json();
    if (!list.some((t) => t.type === 'page')) await new Promise((r) => setTimeout(r, 100));
  }
  const ws = new WebSocket(list.find((t) => t.type === 'page').webSocketDebuggerUrl);
  await new Promise((r, j) => { ws.onopen = r; ws.onerror = j; });
  return ws;
}


