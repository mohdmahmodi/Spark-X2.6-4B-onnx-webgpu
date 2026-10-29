// Run a suite through the real WebGPU page (Chrome + puppeteer-core), greedy.
// The page tokenizes messages itself (web/lib/chat.js); its prompt ids are checked against the suite's.
//
// usage: node tests/run_browser.mjs SUITE [--variant q4f16] [--workers 1] [--kv static|dynamic] [--name browser_q4f16]
//                                         [--max-binding BYTES] [--max-buffer BYTES]
//   SUITE: parity | gsm8k | bfcl | niah | logits | features
//   --max-binding / --max-buffer make the page see (and get) a GPU with lower maxStorageBufferBindingSize /
//   maxBufferSize, to check devices with smaller limits on this one.
//   (server: npm run serve)
import puppeteer from "puppeteer-core";
import { findChrome } from "../tools/chrome.mjs";
import { parseArgs } from "node:util";
import { appendFileSync, existsSync, mkdirSync, readFileSync, writeFileSync } from "node:fs";

const { values: a, positionals } = parseArgs({ allowPositionals: true, options: {
  variant: { type: "string", default: "q4f16" }, workers: { type: "string", default: "1" },
  kv: { type: "string", default: "static" }, name: { type: "string" }, limit: { type: "string" },
  url: { type: "string", default: "http://localhost:8000/" },
  chrome: { type: "string" },
  flags: { type: "string", default: "" },
  "max-binding": { type: "string" }, "max-buffer": { type: "string" },
} });
const suite = positionals[0];

// Runs in the page before its scripts: requestAdapter returns the real adapter behind a wrapper whose limits are
// lowered, and requestDevice never asks for more than those, so the device really has the smaller limits.
function lowerLimits(binding, buffer) {
  const real = navigator.gpu.requestAdapter.bind(navigator.gpu);
  navigator.gpu.requestAdapter = async (opts) => {
    const adapter = await real(opts);
    if (!adapter) return adapter;
    const limits = {};
    for (const k in adapter.limits) limits[k] = adapter.limits[k];
    if (buffer) limits.maxBufferSize = Math.min(limits.maxBufferSize, buffer);
    limits.maxStorageBufferBindingSize = Math.min(limits.maxStorageBufferBindingSize, binding || Infinity, limits.maxBufferSize);
    return {
      __proto__: GPUAdapter.prototype,
      features: adapter.features, info: adapter.info, limits, isFallbackAdapter: adapter.isFallbackAdapter,
      requestDevice: (desc = {}) => {
        const req = { ...desc.requiredLimits };
        for (const k of ["maxBufferSize", "maxStorageBufferBindingSize"]) req[k] = Math.min(req[k] ?? limits[k], limits[k]);
        return adapter.requestDevice({ ...desc, requiredLimits: req });
      },
    };
  };
}
const name = a.name || `browser_${a.variant}${a.kv === "dynamic" ? "_dynkv" : ""}`;
const outDir = `export/tests/results/${name}`;
mkdirSync(outDir, { recursive: true });

// One Chrome instance per worker: two WebGPU pages in one Chrome GPU process ran out of D3D12
// descriptor heap space (device lost, E_OUTOFMEMORY) on an RTX 3090 Ti.
export async function openPages(n, { variant = a.variant, kv = a.kv, flags = a.flags } = {}) {
  const browsers = [];
  const pages = [];
  for (let i = 0; i < n; i++) {
    const browser = await puppeteer.launch({
      executablePath: a.chrome || findChrome(), headless: true, userDataDir: i ? `.chrome-bench-${i}` : ".chrome-bench", protocolTimeout: 0,
      args: ["--enable-unsafe-webgpu", "--force_high_performance_gpu", "--no-first-run", ...flags.split(" ").filter(Boolean)],
    });
    browsers.push(browser);
    const page = await browser.newPage();
    page.on("pageerror", (e) => console.log(`[page ${i} error]`, e.message));
    page.on("console", (m) => { if (m.type() === "error") console.log(`[page ${i}]`, m.text().slice(0, 300)); });
    if (a["max-binding"] || a["max-buffer"]) await page.evaluateOnNewDocument(lowerLimits, Number(a["max-binding"]) || 0, Number(a["max-buffer"]) || 0);
    await page.goto(`${a.url}?harness=1&variant=${variant}&kv=${kv}`);
    await page.waitForFunction(() => window.__harness, { timeout: 0, polling: 200 });
    pages.push(page);
  }
  const info = await pages[0].evaluate(async () => {
    let limits = null;
    try {
      const d = await (await import("onnxruntime-web/webgpu")).env.webgpu.device;
      limits = { maxStorageBufferBindingSize: d.limits.maxStorageBufferBindingSize, maxBufferSize: d.limits.maxBufferSize };
    } catch {}
    return { variant: __harness.variant, gpu: __harness.gpu, staticKV: __harness.staticKV, loadMs: __harness.loadMs, limits };
  });
  const browser = { close: () => Promise.all(browsers.map((b) => b.close())) };
  return { browser, pages, info };
}

async function main() {
  const { browser, pages, info } = await openPages(suite === "logits" ? 1 : Number(a.workers));
  console.log(name, JSON.stringify(info));
  writeFileSync(`${outDir}/info.json`, JSON.stringify(info, null, 1));

  if (suite === "logits") {
    const tasks = JSON.parse(readFileSync("export/tests/suites/parity.json", "utf8")).slice(0, 20);
    const forced = Object.fromEntries(JSON.parse(readFileSync("export/tests/results/hf_bf16/logits_forced.json", "utf8")).map((f) => [f.id, f.forced]));
    const chunks = [];
    for (const t of tasks) {
      // teacher-force all but the last forced token: logits after the prompt + after each of them
      const res = await pages[0].evaluate((task) => __harness.run(task), { type: "logits", prompt_ids: t.prompt_ids, forced: forced[t.id].slice(0, -1) });
      for (const b of res.logits) chunks.push(Buffer.from(b, "base64"));
      process.stdout.write(".");
    }
    writeFileSync(`${outDir}/logits.f32`, Buffer.concat(chunks));
    console.log("\nsaved", `${outDir}/logits.f32`);
    await browser.close();
    return;
  }

  let tasks = JSON.parse(readFileSync(`export/tests/suites/${suite}.json`, "utf8"));
  if (a.limit) tasks = tasks.slice(0, Number(a.limit));
  const outFile = `${outDir}/${suite}.jsonl`;
  const done = new Set(existsSync(outFile) ? readFileSync(outFile, "utf8").split("\n").filter(Boolean).map((l) => JSON.parse(l).id) : []);
  const queue = tasks.filter((t) => !done.has(t.id));
  console.log(`${suite}: ${tasks.length} tasks, ${done.size} already done, ${pages.length} worker(s)`);
  const t0 = Date.now();
  let n = 0;
  await Promise.all(pages.map(async (page) => {
    while (queue.length) {
      const t = queue.shift();
      const { prompt_ids, ...task } = t;
      const res = await page.evaluate((task) => __harness.run(task), { ...task, type: "generate" });
      const sameIds = res.prompt_ids.length === prompt_ids.length && res.prompt_ids.every((v, i) => v === prompt_ids[i]);
      appendFileSync(outFile, JSON.stringify({ id: t.id, ids: res.ids, text: res.text, tool_calls: res.tool_calls, prompt_ids_match: sameIds, stats: res.stats }) + "\n");
      n++;
      if (n % 10 === 0 || !queue.length) console.log(`${suite}: ${done.size + n}/${tasks.length} (${((Date.now() - t0) / 1000).toFixed(0)}s)${sameIds ? "" : " PROMPT IDS DIFFER"}`);
    }
  }));
  await browser.close();
}

if (!process.env.NO_MAIN) await main();
