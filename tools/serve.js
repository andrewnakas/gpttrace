#!/usr/bin/env node
/*
 * Serve the repo the way Cloudflare Pages does: /slug/ maps to
 * /slug/index.html, extensionless URLs map to .html, and every rule in
 * _headers is applied — so the CSP is tested locally, not first in
 * production. (Adapted from AudioSaw's tools/serve.js.)
 *
 *   node tools/serve.js [port]          # default 8788
 *   import { start } from './serve.js'; const s = await start(0);  // s.port, s.close()
 *
 * Only the subset of _headers this repo uses: exact paths, a trailing '*',
 * and '*' inside a segment ('/*.wasm'). Later rules add to earlier ones, as
 * on Pages.
 */
import http from 'node:http';
import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

export const ROOT = path.join(path.dirname(fileURLToPath(import.meta.url)), '..');
const TYPES = {
  '.js': 'application/javascript', '.mjs': 'application/javascript', '.html': 'text/html; charset=utf-8',
  '.css': 'text/css', '.json': 'application/json', '.wasm': 'application/wasm', '.svg': 'image/svg+xml',
  '.png': 'image/png', '.ico': 'image/x-icon', '.woff2': 'font/woff2', '.txt': 'text/plain; charset=utf-8',
  '.xml': 'application/xml', '.webmanifest': 'application/manifest+json',
  '.wav': 'audio/wav', '.mp3': 'audio/mpeg', '.flac': 'audio/flac', '.onnx': 'application/octet-stream',
  '.mjs.map': 'application/json', '.jpg': 'image/jpeg', '.jpeg': 'image/jpeg', '.webp': 'image/webp', '.mp4': 'video/mp4'
};

export function parseHeaders() {
  const rules = [];
  let cur = null;
  fs.readFileSync(path.join(ROOT, '_headers'), 'utf8').split('\n').forEach((line) => {
    if (!line.trim() || /^\s*#/.test(line)) return;
    if (!/^\s/.test(line)) {
      const pat = line.trim();
      const re = new RegExp('^' + pat.replace(/[.+?^${}()|[\]\\]/g, '\\$&').replace(/\*/g, '.*') + '$');
      cur = { re, headers: [] };
      rules.push(cur);
    } else if (cur) {
      const i = line.indexOf(':');
      if (i > 0) cur.headers.push([line.slice(0, i).trim(), line.slice(i + 1).trim()]);
    }
  });
  return rules;
}

export function start(port) {
  const rules = parseHeaders();
  const server = http.createServer((req, res) => {
    const url = decodeURIComponent(req.url.split('?')[0]);
    const file = resolve(url);
    if (!file) {
      const nf = path.join(ROOT, '404.html');
      res.writeHead(404, { 'content-type': 'text/html; charset=utf-8' });
      res.end(fs.existsSync(nf) ? fs.readFileSync(nf) : 'not found'); return;
    }
    const headers = { 'content-type': TYPES[path.extname(file)] || 'application/octet-stream' };
    rules.forEach((r) => { if (r.re.test(url)) r.headers.forEach(([k, v]) => { headers[k.toLowerCase()] = v; }); });
    // Pages' long cache would pin a stale file during development.
    headers['cache-control'] = 'no-store';
    res.writeHead(200, headers);
    fs.createReadStream(file).pipe(res);
  });
  return new Promise((resolve) => server.listen(port, '127.0.0.1', () => resolve({
    port: server.address().port,
    close: () => new Promise((r) => server.close(r))
  })));
}

// Dev only: /__models/<model id>/<path> → the local Hugging Face cache, so
// optional models load without a network download while testing.
import os from 'node:os';
import { MODELS } from '../js/engine/models.js';
export function resolveModel(url) {
  const m = /^\/__models\/([^/]+)\/(.+)$/.exec(url);
  if (!m) return null;
  const model = Object.values(MODELS).find((x) => x.id === m[1]);
  if (!model || !model.localCache) return null;
  const f = path.join(os.homedir(), '.cache', 'gpttrace', 'hf', model.localCache, m[2]);
  return fs.existsSync(f) ? f : null;
}

export function resolve(url) {
  const model = resolveModel(url);
  if (model) return model;
  let file = path.join(ROOT, url);
  if (url.endsWith('/')) file = path.join(file, 'index.html');
  else if (!path.extname(file) && fs.existsSync(file + '.html')) file += '.html';
  if (!file.startsWith(ROOT) || !fs.existsSync(file) || fs.statSync(file).isDirectory()) return null;
  return file;
}

if (process.argv[1] === fileURLToPath(import.meta.url)) {
  start(+process.argv[2] || 8788).then((s) => console.log('serving http://localhost:' + s.port));
}
