// Chat app: saved chats in localStorage, streaming replies with optional reasoning, runtime stats.
//
// URL options
//   ?model=<url>    folder with spark.json (default: ./spark, else the Hugging Face repository)
//   ?variant=q4     force the fp32-activation build
//   ?kv=dynamic     disable the static KV cache
//   ?harness        expose window.__harness for tests/run_browser.mjs
//   ?bench=N        run the speed benchmark (tools/bench.mjs)
//
// The page has no looping CSS animations on purpose: a repaint every frame competes with WebGPU and
// cost about 8% of decode speed.
import { loadSpark } from "../lib/spark.js";
import { decorateCode, icon } from "./code.js";
import { renderMarkdown } from "./markdown.js";

const $ = (id) => document.getElementById(id);
const params = new URLSearchParams(location.search);
const testRun = params.has("bench") || params.has("harness"); // test runs never touch saved chats

const formatMs = (ms) => (ms >= 1000 ? `${(ms / 1000).toFixed(1)} s` : `${Math.round(ms)} ms`);
// "nvidia ampere" -> "NVIDIA Ampere"
const ACRONYMS = new Set(["nvidia", "amd", "ati", "arm", "gpu", "rtx", "gtx"]);
const formatGpu = (name) => name.split(" ").filter(Boolean)
  .map((w) => (ACRONYMS.has(w.toLowerCase()) ? w.toUpperCase() : w[0].toUpperCase() + w.slice(1))).join(" ");

// ------------------------------------------------------------------ saved chats (this browser only)

const STORAGE_KEY = "spark-chats-v1";
const MAX_SAVED = 100;

function loadChats() {
  try {
    return JSON.parse(localStorage.getItem(STORAGE_KEY)) || [];
  } catch {
    return [];
  }
}

function saveChats() {
  if (testRun) return;
  try {
    localStorage.setItem(STORAGE_KEY, JSON.stringify(chats.slice(0, MAX_SAVED)));
  } catch {
    // storage full or blocked: chats stay in memory for this session
  }
}

let chats = testRun ? [] : loadChats();
let current = null; // chat being shown; created on the first message
let spark = null;
let abort = null;
let busy = false;

function renderList() {
  const list = $("list");
  list.replaceChildren();
  if (!chats.length) return;
  list.append(Object.assign(document.createElement("div"), { className: "side-label", textContent: "Chats" }));
  for (const chat of chats) {
    const row = document.createElement("div");
    row.className = `chat-item${chat === current ? " active" : ""}`;
    const open = Object.assign(document.createElement("button"), { className: "open", type: "button", textContent: chat.title, title: chat.title });
    open.onclick = () => {
      if (busy) return;
      show(chat);
      $("side").classList.remove("open");
    };
    const remove = Object.assign(document.createElement("button"), { className: "delete", type: "button", title: "Delete chat" });
    remove.setAttribute("aria-label", "Delete chat");
    remove.innerHTML = icon("trash-2");
    remove.onclick = () => {
      if (busy && chat === current) return;
      chats = chats.filter((c) => c !== chat);
      saveChats();
      if (chat === current) show(null);
      else renderList();
    };
    row.append(open, remove);
    list.append(row);
  }
}

// ------------------------------------------------------------------ thread

const SUGGESTIONS = [
  "Write a Python function that merges two sorted lists",
  "A train travels 180 km in 2.5 hours. What is its average speed?",
  "How does WebGPU differ from WebGL?",
  "请用三句话介绍长城",
];

function welcome() {
  const el = Object.assign(document.createElement("section"), { className: "welcome" });
  el.innerHTML = "<h1>What can I help with?</h1><p>Spark-X2.6-4B runs entirely on your GPU. Nothing you type leaves this device.</p>";
  const row = Object.assign(document.createElement("div"), { className: "suggestions" });
  for (const prompt of SUGGESTIONS) {
    const button = Object.assign(document.createElement("button"), { className: "suggestion", type: "button", textContent: prompt });
    button.onclick = () => {
      $("prompt").value = prompt;
      send();
    };
    row.append(button);
  }
  el.append(row);
  return el;
}

function show(chat) {
  current = chat;
  $("thread").replaceChildren();
  if (!chat?.messages.length) $("thread").append(welcome());
  else for (const m of chat.messages) m.role === "user" ? addUserMessage(m.content) : addBotMessage(m);
  renderList();
  scrollDown(true);
}

function addUserMessage(text) {
  $("thread").querySelector(".welcome")?.remove();
  $("thread").append(Object.assign(document.createElement("div"), { className: "msg-user", textContent: text }));
}

function statsLine(stats) {
  return [`${stats.newTokens ?? 0} tokens`, stats.decodeTokS ? `${stats.decodeTokS.toFixed(1)} tok/s` : null,
    stats.ttftMs ? `${formatMs(stats.ttftMs)} to first token` : null].filter(Boolean);
}

// Adds an assistant message and returns its update function. While reasoning streams it is shown
// expanded under "Thinking"; once the answer starts it collapses to "Thought for N s".
function addBotMessage(message = {}) {
  $("thread").querySelector(".welcome")?.remove();
  const el = Object.assign(document.createElement("div"), { className: "msg-bot" });
  el.innerHTML = `<div class="thought" hidden><button class="thought-toggle" type="button" aria-expanded="false"><span></span>${icon("chevron-right")}</button>
    <div class="thought-text" hidden></div></div><div class="body"></div><div class="msg-actions" hidden></div>`;
  $("thread").append(el);
  const thought = el.querySelector(".thought");
  const toggle = thought.querySelector(".thought-toggle");
  const thoughtText = thought.querySelector(".thought-text");
  const body = el.querySelector(".body");
  const actions = el.querySelector(".msg-actions");
  let userToggled = false;
  toggle.onclick = () => {
    userToggled = true;
    const open = toggle.getAttribute("aria-expanded") !== "true";
    toggle.setAttribute("aria-expanded", String(open));
    thoughtText.hidden = !open;
  };

  const update = ({ reasoning = "", content = "", stats = null, live = false, thoughtMs = 0, stopped = false }) => {
    thought.hidden = !reasoning && !(live && !content && message.thinking);
    toggle.firstChild.textContent = live && !content ? "Thinking" : thoughtMs ? `Thought for ${formatMs(thoughtMs)}` : "Reasoning";
    thoughtText.textContent = reasoning;
    if (!userToggled) {
      const open = live && !content;
      toggle.setAttribute("aria-expanded", String(open));
      thoughtText.hidden = !open;
    }
    body.innerHTML = content ? renderMarkdown(content) : "";
    decorateCode(body, { highlight: !live });
    actions.hidden = live || !stats;
    if (!live && stats) {
      const copy = Object.assign(document.createElement("button"), { className: "copy-btn", type: "button", title: "Copy reply" });
      copy.setAttribute("aria-label", "Copy reply");
      copy.innerHTML = icon("copy");
      copy.onclick = async () => {
        await navigator.clipboard.writeText(content).catch(() => {});
        copy.innerHTML = icon("check");
        setTimeout(() => (copy.innerHTML = icon("copy")), 1600);
      };
      const line = Object.assign(document.createElement("span"), { className: "msg-stats" });
      for (const part of [...statsLine(stats), stopped ? "stopped" : null].filter(Boolean)) line.append(Object.assign(document.createElement("span"), { textContent: part }));
      actions.replaceChildren(copy, line);
    }
  };
  update({ reasoning: message.reasoning_content || "", content: message.content || "", stats: message.stats, thoughtMs: message.thought_ms });
  return update;
}

const scroller = $("scroll");
let followOutput = true;
scroller.addEventListener("scroll", () => {
  followOutput = scroller.scrollHeight - scroller.scrollTop - scroller.clientHeight < 80;
});
const scrollDown = (force) => {
  if (force || followOutput) scroller.scrollTop = scroller.scrollHeight;
};

// ------------------------------------------------------------------ generation

function showStats(stats) {
  $("s-ttft").textContent = stats.ttftMs ? formatMs(stats.ttftMs) : "–";
  $("s-tps").textContent = stats.decodeTokS ? `${stats.decodeTokS.toFixed(1)} tok/s` : "–";
  $("s-tokens").textContent = `${stats.promptTokens ?? 0} in, ${stats.newTokens ?? 0} out`;
  $("s-tokens").title = `${stats.reusedTokens ?? 0} prompt tokens reused from the previous turn`;
}

// With thinking on, the reply is "<reasoning></think><answer>"; with it off the template has already
// closed the think block, but the model may still echo "</think>".
function splitReasoning(text, thinking) {
  if (!thinking) return ["", text.replace(/^<\/think>/, "")];
  const end = text.indexOf("</think>");
  return end < 0 ? [text, ""] : [text.slice(0, end), text.slice(end + "</think>".length).trimStart()];
}

async function runTurn({ messages, thinking, effort, temperature, maxNewTokens, reuse = true }) {
  abort = new AbortController();
  const stats = {};
  const update = addBotMessage({ thinking });
  const started = performance.now();
  let thoughtMs = 0;
  let text = "";
  let lastPaint = 0;
  try {
    for await (const piece of spark.generate(messages, { thinking, effort, temperature, maxNewTokens, signal: abort.signal, stats, reuse })) {
      text += piece;
      if (thinking && !thoughtMs && text.includes("</think>")) thoughtMs = performance.now() - started;
      if (performance.now() - lastPaint > 50) {
        const [reasoning, content] = splitReasoning(text, thinking);
        update({ reasoning: reasoning.trim(), content, live: true, thoughtMs });
        showStats(stats);
        scrollDown();
        lastPaint = performance.now();
      }
    }
  } finally {
    if (thinking && !thoughtMs) thoughtMs = performance.now() - started;
    showStats(stats);
    const [reasoning, content] = splitReasoning(text, thinking);
    update({ reasoning: reasoning.trim(), content, stats, thoughtMs: thinking ? thoughtMs : 0, stopped: abort.signal.aborted });
    scrollDown();
  }
  const [reasoning, content] = splitReasoning(text, thinking);
  return { ...stats, text, reasoning: reasoning.trim(), content: content.trim(), thoughtMs: thinking ? thoughtMs : 0 };
}

function setBusy(value) {
  busy = value;
  $("go").hidden = value;
  $("stop").hidden = !value;
  $("go").disabled = value || !spark;
}

async function send() {
  const content = $("prompt").value.trim();
  if (!content || busy || !spark) return;
  $("prompt").value = "";
  resizePrompt();
  if (!current) {
    current = { id: crypto.randomUUID?.() || String(Date.now()), title: content.slice(0, 60), messages: [], updated: Date.now() };
    chats.unshift(current);
  } else {
    chats = [current, ...chats.filter((c) => c !== current)];
  }
  current.messages.push({ role: "user", content });
  addUserMessage(content);
  renderList();
  scrollDown(true);
  setBusy(true);
  const chat = current;
  try {
    const reply = await runTurn({
      // past reasoning goes back as reasoning_content, which the chat template renders inside <think>
      messages: chat.messages.map(({ role, content, reasoning_content }) => ({ role, content, ...(reasoning_content && { reasoning_content }) })),
      thinking: $("thinking").checked,
      effort: $("effort").value,
      temperature: Number($("temp").value),
      maxNewTokens: Number($("max").value),
    });
    chat.messages.push({
      role: "assistant",
      content: reply.content,
      ...(reply.reasoning && { reasoning_content: reply.reasoning, thought_ms: reply.thoughtMs }),
      stats: { newTokens: reply.newTokens, decodeTokS: reply.decodeTokS, ttftMs: reply.ttftMs },
    });
  } catch (error) {
    addBotMessage({ content: `**Error:** ${error.message}` });
    chat.messages.pop();
    console.error(error);
  }
  chat.updated = Date.now();
  saveChats();
  setBusy(false);
}

function resizePrompt() {
  const box = $("prompt");
  box.style.height = "auto";
  box.style.height = `${Math.min(box.scrollHeight, 220)}px`;
}

function toggleSettings(open = $("settings").hidden) {
  $("settings").hidden = !open;
  $("settings-btn").setAttribute("aria-expanded", String(open));
}

$("prompt").addEventListener("input", resizePrompt);
$("prompt").addEventListener("keydown", (event) => {
  if (event.key === "Enter" && !event.shiftKey && !event.isComposing) {
    event.preventDefault();
    send();
  }
});
$("thinking").addEventListener("change", () => ($("effort").hidden = !$("thinking").checked));
$("settings-btn").onclick = () => toggleSettings();
document.addEventListener("click", (event) => {
  if (!$("settings").hidden && !event.target.closest("#settings, #settings-btn")) toggleSettings(false);
});
$("go").onclick = send;
$("stop").onclick = () => abort?.abort();
$("new").onclick = () => {
  if (busy) return;
  show(null);
  $("side").classList.remove("open");
  $("prompt").focus();
};
$("menu").onclick = () => $("side").classList.toggle("open");

show(null);

// ------------------------------------------------------------------ model

async function resolveModelBase() {
  if (params.has("model")) return params.get("model");
  const meta = document.querySelector('meta[name="spark-model"]');
  if (!meta.dataset.fallback) return meta.content;
  const local = await fetch(`${meta.content}/spark.json`, { method: "HEAD", cache: "no-cache" }).then((r) => r.ok, () => false);
  return local ? meta.content : meta.dataset.fallback;
}

function onDeviceLost(info) {
  $("status").textContent = "GPU device lost. Reload the page.";
  $("device").textContent = `Device lost (${info.reason || "unknown"}): ${info.message}`;
  $("go").disabled = true;
}

function onProgress({ loaded, total }) {
  $("bar").firstElementChild.style.width = `${(100 * loaded) / total}%`;
  $("status").textContent = loaded < total ? `Downloading model, ${(loaded / 1e9).toFixed(2)} of ${(total / 1e9).toFixed(2)} GB` : "Preparing GPU";
}

try {
  $("bar").hidden = false;
  spark = await loadSpark({
    base: await resolveModelBase(),
    variant: params.get("variant") || undefined,
    staticKV: params.get("kv") !== "dynamic",
    onDeviceLost,
    onProgress,
  });
  $("bar").hidden = true;
  $("status").textContent = "";
  $("s-load").textContent = formatMs(spark.loadMs.total);
  $("device").textContent = `${formatGpu(spark.gpu)}, ${spark.variant}`;
  $("device").title = `Download ${formatMs(spark.loadMs.download)}, session ${formatMs(spark.loadMs.init)}`;
  if (spark.model) {
    document.title = spark.model;
    $("model-name").textContent = spark.model;
  }
  setBusy(false);
  $("prompt").focus();
  window.__spark = spark;
  if (params.has("harness")) (await import("./harness.js")).installHarness(spark);
  if (params.has("bench")) (await import("./bench.js")).runBenchmark(spark, runTurn, params);
} catch (error) {
  $("bar").hidden = true;
  $("status").textContent = "Could not start";
  $("device").textContent = error.message;
  $("thread").replaceChildren();
  addBotMessage({ content: `**Could not start the model.**\n\n${error.message}` });
  console.error(error);
}
