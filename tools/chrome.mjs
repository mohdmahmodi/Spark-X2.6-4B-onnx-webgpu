// Locates a Chrome / Chromium binary for the puppeteer-driven tools. Set CHROME_PATH to override.
import { existsSync } from "node:fs";

const CANDIDATES = {
  win32: [
    `${process.env.PROGRAMFILES}/Google/Chrome/Application/chrome.exe`,
    `${process.env["PROGRAMFILES(X86)"]}/Google/Chrome/Application/chrome.exe`,
    `${process.env.LOCALAPPDATA}/Google/Chrome/Application/chrome.exe`,
  ],
  darwin: ["/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"],
  linux: ["/usr/bin/google-chrome", "/usr/bin/google-chrome-stable", "/usr/bin/chromium", "/usr/bin/chromium-browser"],
};

export function findChrome() {
  if (process.env.CHROME_PATH) return process.env.CHROME_PATH;
  const found = (CANDIDATES[process.platform] || []).find((p) => existsSync(p));
  if (!found) throw new Error("Chrome not found; set CHROME_PATH to the browser executable");
  return found;
}
