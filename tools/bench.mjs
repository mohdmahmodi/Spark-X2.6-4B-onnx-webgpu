// Drives the test page in Chrome (WebGPU) and prints load time, TTFT and decode tok/s.
// usage: node tools/bench.mjs [--variant q4f16] [--tokens 128] [--prompts short,long] [--runs 2] [--headful]
import puppeteer from "puppeteer-core";
import { findChrome } from "./chrome.mjs";
import { parseArgs } from "node:util";

const { values: a } = parseArgs({ options: {
  variant: { type: "string", default: "q4f16" }, tokens: { type: "string", default: "128" },
  prompts: { type: "string", default: "short,long" }, runs: { type: "string", default: "2" },
  url: { type: "string", default: "http://localhost:8000/" }, headful: { type: "boolean", default: false },
  chrome: { type: "string" },
  profile: { type: "string", default: ".chrome-bench" }, kv: { type: "string", default: "static" },
  gpu: { type: "string", default: "high" }, // "low": integrated GPU on hybrid machines
} });

const browser = await puppeteer.launch({
  executablePath: a.chrome || findChrome(), headless: !a.headful, userDataDir: a.profile, protocolTimeout: 0,
  args: ["--enable-unsafe-webgpu", "--enable-features=WebGPUDeveloperFeatures", "--no-first-run",
         ...(a.gpu === "low" ? ["--use-webgpu-power-preference=force-low-power"] : ["--force_high_performance_gpu"])],
});
const page = await browser.newPage();
page.on("console", (m) => { const t = m.text(); if (!t.startsWith("BENCH")) console.log("[page]", t.slice(0, 400)); });
page.on("pageerror", async (e) => {
  console.log("[pageerror]", e.message);
  await browser.close();
  process.exit(1);
});
const url = `${a.url}?variant=${a.variant}&bench=${a.tokens}&prompts=${a.prompts}&runs=${a.runs}&kv=${a.kv}`;
await page.goto(url);
const res = await page.waitForFunction(() => window.__bench, { timeout: 0, polling: 500 }).then((h) => h.jsonValue());
console.log(JSON.stringify(res, null, 1));
await browser.close();
