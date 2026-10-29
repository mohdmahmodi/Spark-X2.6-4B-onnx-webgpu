// Test hooks for tests/run_browser.mjs: window.__harness.run(task) drives the real WebGPU build.
//   {type: "generate", messages | prompt_ids, thinking, effort, tools, max_new_tokens}  -> greedy ids + text
//   {type: "logits", prompt_ids, forced}  -> next-token logits after the prompt and after each forced token
import { parseToolCalls } from "../lib/toolcall.js";

function b64(f32) {
  const bytes = new Uint8Array(f32.buffer, f32.byteOffset, f32.byteLength);
  let s = "";
  for (let i = 0; i < bytes.length; i += 0x8000) s += String.fromCharCode.apply(null, bytes.subarray(i, i + 0x8000));
  return btoa(s);
}

export function installHarness(spark) {
  const run = async (task) => {
    const prompt = task.prompt_ids ?? spark.chat.encodeChat(task.messages, { thinking: task.thinking ?? false, effort: task.effort ?? "high", tools: task.tools });
    const stats = {};
    if (task.type === "logits") {
      const logits = [];
      for await (const _ of spark.generateIds(prompt, {
        temperature: 0, forced: task.forced, maxNewTokens: task.forced.length + 1, stats, reuse: task.reuse ?? false,
        onLogits: (_, l) => logits.push(b64(new Float32Array(l))),
      }));
      return { prompt_len: prompt.length, logits };
    }
    const ids = [];
    // benchmark tasks start from an empty cache (reuse: false) so results never depend on task order
    for await (const id of spark.generateIds(prompt, { temperature: 0, maxNewTokens: task.max_new_tokens ?? 128, stats, reuse: task.reuse ?? false })) ids.push(id);
    const text = spark.chat.decode(ids);
    return {
      prompt_ids: prompt,
      ids,
      text,
      tool_calls: task.tools ? parseToolCalls(text, task.tools).calls : undefined,
      stats: { reusedTokens: stats.reusedTokens, promptTokens: stats.promptTokens, newTokens: stats.newTokens, ttftMs: stats.ttftMs, decodeTokS: stats.decodeTokS, totalMs: stats.totalMs },
    };
  };
  const memory = () => ({ jsHeapMB: performance.memory ? performance.memory.usedJSHeapSize / 1e6 : null, kvCapacity: spark.kvCapacity() });
  window.__harness = { run, memory, variant: spark.variant, gpu: spark.gpu, staticKV: spark.staticKV, loadMs: spark.loadMs };
}
