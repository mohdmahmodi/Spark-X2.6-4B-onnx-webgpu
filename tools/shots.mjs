// Renders the model-card images into model-card/assets/ (and the chat preview into web/assets/): benchmark charts from web/benchmarks.html and a
// screenshot of the chat UI answering real prompts (headless Chrome, WebGPU, model from ./spark).
// usage: node tools/shots.mjs [--only charts|chat]   (needs: npm run serve)
import puppeteer from "puppeteer-core";
import { findChrome } from "./chrome.mjs";
import { copyFileSync, mkdirSync } from "node:fs";
import { parseArgs } from "node:util";

const { values: a } = parseArgs({ options: {
  url: { type: "string", default: "http://localhost:8000/" }, only: { type: "string", default: "" },
  out: { type: "string", default: "model-card/assets" }, theme: { type: "string", default: "dark" },
  chrome: { type: "string" },
} });
mkdirSync(a.out, { recursive: true });

const browser = await puppeteer.launch({
  executablePath: a.chrome || findChrome(), headless: true, userDataDir: ".chrome-bench", protocolTimeout: 0,
  args: ["--enable-unsafe-webgpu", "--force_high_performance_gpu", "--no-first-run"],
});
const page = await browser.newPage();
page.on("pageerror", (e) => console.log("[pageerror]", e.message));
await page.emulateMediaFeatures([{ name: "prefers-color-scheme", value: a.theme }]);

if (a.only !== "chat") {
  // element id in benchmarks.html -> file name in model-card/assets/
  // the banner keeps the page's black theme; the charts use a light background, like most model cards
  const shots = [["social", "hero", "dark"], ["quality", "quality", "light"], ["results", "results", "light"], ["parity", "accuracy", "light"],
    ["history", "speed", "light"], ["summary", "summary", "light"]];
  for (const [id, file, theme] of shots) {
    await page.setViewport({ width: 1200, height: 800, deviceScaleFactor: 2 });
    await page.goto(`${a.url}benchmarks.html?shot=${id}&theme=${theme}`);
    await page.waitForSelector("body[data-ready]");
    const node = await page.$(`#${id}`);
    await node.screenshot({ path: `${a.out}/${file}.png` });
    console.log("wrote", `${a.out}/${file}.png`);
  }
}

if (a.only !== "charts") {
  await page.setViewport({ width: 1440, height: 900, deviceScaleFactor: 2 });
  await page.goto(a.url);
  await page.evaluate(() => localStorage.clear());
  await page.reload();
  await page.waitForFunction(() => !document.getElementById("go").disabled, { timeout: 0, polling: 250 });
  const ask = async (text, { thinking = false } = {}) => {
    await page.$eval("#temp", (e) => (e.value = "0")); // greedy, so the screenshot is reproducible
    await page.$eval("#thinking", (e, on) => { if (e.checked !== on) e.click(); }, thinking);
    await page.type("#prompt", text);
    await page.click("#go");
    await page.waitForFunction(() => document.getElementById("stop").hidden && !document.getElementById("go").disabled, { timeout: 0, polling: 250 });
  };
  await ask("Write a haiku about GPUs.");
  await page.click("#new");
  await ask("What is 17 × 24?");
  await page.click("#new");
  await ask("Write a Python function that checks whether a string is a palindrome, ignoring case and punctuation. Keep the explanation short.",
    { thinking: true });
  await page.evaluate(() => document.getElementById("scroll").scrollTo(0, 0));
  await page.screenshot({ path: `${a.out}/chat.png` });
  copyFileSync(`${a.out}/chat.png`, "web/assets/chat-preview.png"); // shown on the benchmarks page
  console.log("wrote", `${a.out}/chat.png`, "and web/assets/chat-preview.png");
}
await browser.close();
