// HTTP bridge to the real WebGPU page, so Python tests can drive the browser build.
//   POST /run     body = harness task (see web/app/harness.js)  -> harness result
//   GET  /memory  -> page JS heap + KV capacity
//   GET  /info    -> variant, GPU, load times
// usage: node tests/browser_bridge.mjs [--port 8765] [--variant q4f16]   (needs: npm run serve)
import puppeteer from "puppeteer-core";
import { findChrome } from "../tools/chrome.mjs";
import { createServer } from "node:http";
import { parseArgs } from "node:util";

const { values: a } = parseArgs({ options: {
  port: { type: "string", default: "8765" }, variant: { type: "string", default: "q4f16" }, kv: { type: "string", default: "static" },
  url: { type: "string", default: "http://localhost:8000/" },
  chrome: { type: "string" },
} });

const browser = await puppeteer.launch({
  executablePath: a.chrome || findChrome(), headless: true, userDataDir: ".chrome-bench", protocolTimeout: 0,
  args: ["--enable-unsafe-webgpu", "--force_high_performance_gpu", "--no-first-run", "--js-flags=--expose-gc"],
});
const page = await browser.newPage();
const errors = [];
page.on("pageerror", (e) => { errors.push(e.message); console.log("[pageerror]", e.message); });
page.on("console", (m) => { if (m.type() === "error") { errors.push(m.text()); console.log("[page]", m.text().slice(0, 300)); } });
browser.on("disconnected", () => { console.log("browser disconnected"); process.exit(2); });
await page.goto(`${a.url}?harness=1&variant=${a.variant}&kv=${a.kv}`);
await page.waitForFunction(() => window.__harness, { timeout: 0, polling: 200 });

let busy = Promise.resolve();
createServer(async (req, res) => {
  const reply = (code, obj) => { res.writeHead(code, { "content-type": "application/json" }); res.end(JSON.stringify(obj)); };
  try {
    if (req.url === "/info") return reply(200, { ...(await page.evaluate(() => ({ variant: __harness.variant, gpu: __harness.gpu, staticKV: __harness.staticKV, loadMs: __harness.loadMs }))), errors });
    if (req.url === "/memory") return reply(200, { ...(await page.evaluate(() => { window.gc?.(); return __harness.memory(); })), errors: errors.length });
    if (req.url === "/run" && req.method === "POST") {
      let body = "";
      for await (const c of req) body += c;
      const task = JSON.parse(body);
      const job = busy.then(() => page.evaluate((t) => __harness.run(t), task));
      busy = job.catch(() => {});
      return reply(200, await job);
    }
    reply(404, {});
  } catch (e) {
    reply(500, { error: String(e.message || e) });
  }
}).listen(Number(a.port), () => console.log(`bridge ready on http://localhost:${a.port}`));
