// Numbers shown on the benchmarks page.
//
// MODELS / QUALITY / ALL_BENCHMARKS: reported by XHToken for the original Spark-X2.6-4B, from the model card
// (huggingface.co/XHToken/Spark-X2.6-4B, images/benchmark.svg and images/all_benchmark.svg, revision d1b5f260).
// Thinking mode, high effort. "*" marks scores XHToken took from other models' published cards; null means not reported.
//
// Everything else was measured on this build: docs/VALIDATION.md and export/tests/results/decode_vs_context.json.

export const MODELS = ["Spark-X2.6-4B", "Spark-X2.5-4B", "Qwen3.5-9B", "Qwen3.5-4B", "Gemma4-12B", "Gemma4-E4B"];

export const QUALITY = [
  { name: "BFCL-V4", task: "Function calling", scores: [71.8, 65.1, 66.1, 50.3, 37.4, 47.0] },
  { name: "τ³-bench", task: "General agent", scores: [34.7, 30.4, 9.3, 6.7, 13.3, 10.1] },
  { name: "MCP-Atlas", task: "MCP tool use", scores: [62.4, 54.6, 47.4, 40.8, 30.5, 15.0] },
  { name: "BrowseComp", task: "Web browsing", scores: [53.1, 40.9, 8.3, 14.3, 10.0, 8.3] },
  { name: "PAgentBench", task: "Personal assistant", scores: [74.1, 59.4, 53.8, 51.4, 51.8, 37.0] },
  { name: "SWE Pro", task: "Software engineering", scores: [54.3, 44.4, 33.8, 29.4, 21.9, 4.0] },
  { name: "HLE with tools", task: "Expert-level knowledge", scores: [36.9, 21.8, 17.9, 20.2, 19.0, 10.1] },
  { name: "AA-LCR v1.1", task: "Long-context reasoning", scores: [67.3, 64.7, 70.0, 63.0, 63.7, 32.0] },
];

export const ALL_BENCHMARKS = [
  ["Average", [["Average", "63.3", "52.1", "49.1", "43.2", "47.0", "29.7"]]],
  ["Agent", [
    ["BFCL-V4", "71.8", "65.1", "66.1*", "50.3*", "37.4", "47.0*"],
    ["τ²-bench", "86.9", "75.1", "79.1*", "79.9*", "69.0*", "42.2*"],
    ["τ³-bench", "34.7", "30.4", "9.3", "6.7", "13.3", "10.1"],
    ["VitaBench 2.0", "30.8", "25.2", "15.6", "18.2", "12.4", "4.8"],
    ["MCP-Atlas", "62.4", "54.6", "47.4*", "40.8*", "30.5*", "15.0*"],
    ["EnterpriseOps-Gym", "33.6", "27.3", "24.7", "22.9", "24.5", "8.6"],
    ["ClawEval-EN", "69.9", "66.2", "66.5*", "62.3*", "58.3", "58.0*"],
    ["BrowseComp", "53.1", "40.9", "8.3", "14.3", "10.0", "8.3"],
    ["SWebBench", "71.7", "46.3", "33.2", "36.3", "32.0", "19.3"],
    ["PAgentBench", "74.1", "59.4", "53.8", "51.4", "51.8", "37.0"],
  ]],
  ["Code", [
    ["SWE Pro", "54.3", "44.4", "33.8*", "29.4*", "21.9*", "4.0*"],
    ["SWE Verified", "62.8", "41.6", "53.1*", "38.8*", "44.2*", "14.0*"],
    ["SWE Multilingual", "64.7", "53.3", "43.3", "27.7", "32.5*", null],
    ["SciCode", "36.8", "34.7", "32.7*", "24.0", "39.8", "27.5"],
  ]],
  ["Math", [
    ["Gaokao 2026 (of 150)", "134.9", "133.4", "135.5", "130.3", "130.6", "102.4"],
    ["AIME 2026", "91.2", "90.7", "88.2", "83.0", "82.1*", "42.5*"],
    ["HMMT Feb 26", "81.3", "81.2", "70.8", "69.7", "65.6", "34.2"],
    ["IMO-AnswerBench", "75.2", "74.2", "69.8", "68.5", "57.2", "26.9"],
  ]],
  ["Long context", [
    ["AA-LCR v1.1", "67.3", "64.7", "70.0*", "63.0*", "63.7*", "32.0*"],
    ["LongBench v2", "53.3", "48.7", "55.2*", "50.0*", "45.3", "27.8"],
    ["Graphwalks BFS 256k F1", "92.2", "47.1", "88.4", "79.0", "68.6", "54.8"],
    ["Graphwalks BFS 1M F1", "34.2", "2.5", "25.3", "15.8", null, null],
    ["OpenAI MRCR v2 256k–512k", "65.6", "9.7", "58.6", "48.3", null, null],
    ["OpenAI MRCR v2 512k–1M", "26.0", "2.9", "34.3", "12.2", null, null],
  ]],
  ["General and knowledge", [
    ["IFEval", "93.6", "93.0", "91.5*", "89.8*", "94.8", "45.3"],
    ["IFBench", "75.8", "75.0", "64.5", "59.2", "73.5*", "44.0*"],
    ["HLE without tools", "17.9", "12.3", "14.3", "8.6", "13.1", "3.9"],
    ["HLE with tools", "36.9", "21.8", "17.9", "20.2", "19.0", "10.1"],
    ["GPQA", "69.9", "67.4", "77.2", "67.2", "72.8", "54.5"],
    ["GPQA Diamond", "77.1", "73.7", "81.7*", "76.2*", "78.8*", "58.6*"],
  ]],
];

export const SYSTEMS = [
  { key: "bf16", name: "Original model, bf16" },
  { key: "int4", name: "Official INT4 weights" },
  { key: "web", name: "This build", ours: true },
];

export const ACCURACY = [
  { name: "GSM8K", detail: "1,319 problems", n: 1319, bf16: 1198, int4: 1194, web: 1194 },
  { name: "BFCL v3 function calling", detail: "200 cases", n: 200, bf16: 178, int4: 179, web: 179 },
  { name: "Needle in a haystack", detail: "12 cases, 1k to 16k tokens", n: 12, bf16: 12, int4: 12, web: 12, count: true },
];

// check, transformers bf16, transformers INT4, this build
export const SUMMARY = [
  ["GSM8K", "90.8%", "90.5%", "90.5%"],
  ["BFCL v3 function calling", "89.0%", "89.5%", "89.5%"],
  ["Needle in a haystack", "12/12", "12/12", "12/12"],
  ["Next-token KL divergence to bf16", "0", "0.0235", "0.0238"],
  ["Greedy parity, first token, 50 prompts", "50/50", "46/50", "46/50"],
  ["Thinking, multi-turn and tool calls", "6/6", "", "8/8"],
  ["30-minute soak test", "", "", "0 errors"],
];

export const DECODE = [[28, 91.9], [1017, 91.0], [4089, 85.0], [8185, 80.1], [16377, 69.4]]; // context tokens, tok/s (mean of 2 runs)
export const FIRST_TOKEN = [[28, 0.092], [1017, 0.69], [2518, 1.72], [4089, 2.89], [8185, 6.37], [16377, 15.15]]; // prompt tokens, s (median)

// decode tok/s at 2,500 tokens of context after each change
export const HISTORY = [
  { step: "Starting point", detail: "onnxruntime-web 1.30", value: 21 },
  { step: "Newer runtime", detail: "onnxruntime-web 1.31", value: 21 },
  { step: "Static KV cache", detail: "updated in place on the GPU", value: 77 },
  { step: "Float matmul inputs", detail: "shipped build", value: 88, ours: true },
];

export const FIGURES = [
  ["Decode speed", "90", "tok/s", "69 tok/s at 16k tokens of context"],
  ["First token", "92", "ms", "1.7 s for a 2,500-token prompt"],
  ["GSM8K", "90.5", "%", "Official INT4 model: 90.5%"],
  ["Download", "2.77", "GB", "Cached after the first visit"],
];
