// Speed benchmark for tools/bench.mjs: greedy runs on fixed prompts, result on window.__bench.
// Every run starts from an empty KV cache (reuse: false), so time to first token is a full prefill.
//   ?bench=N          new tokens per run (default 128)
//   ?prompts=a,b      short (28 tokens) and/or long (2,518 tokens)
//   ?runs=N           runs per prompt (default 2; the first includes shader compilation)
export async function runBenchmark(spark, runTurn, params) {
  const prompts = {
    short: "Write a detailed essay about the history of the printing press.",
    long: await (await fetch("app/bench-prompt.txt")).text(),
  };
  const runs = [];
  for (const name of (params.get("prompts") || "short").split(",")) {
    for (let run = 0; run < Number(params.get("runs") || 2); run++) {
      const s = await runTurn({
        messages: [{ role: "user", content: prompts[name] }],
        thinking: false,
        effort: "high",
        temperature: 0,
        maxNewTokens: Number(params.get("bench") || 128),
        reuse: false,
      });
      runs.push({ prompt: name, run, promptTokens: s.promptTokens, newTokens: s.newTokens, ttftMs: +s.ttftMs.toFixed(1),
        prefillMs: +s.prefillMs.toFixed(1), decodeTokS: +s.decodeTokS.toFixed(2), text: s.text.slice(0, 160) });
    }
  }
  window.__bench = { model: spark.model, variant: spark.variant, gpu: spark.gpu, staticKV: spark.staticKV,
    kvCapacity: spark.kvCapacity(), loadMs: spark.loadMs, runs };
  console.log(`BENCH ${JSON.stringify(window.__bench)}`);
}
