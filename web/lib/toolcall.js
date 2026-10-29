// Spark2_5 tool-call parser (port of SGLang's spark25 detector, python/sglang/srt/function_call/spark25_detector.py).
// Wire format: <tool_call>name<arg_key>k</arg_key><arg_value>v</arg_value>...</tool_call>
// Values are converted using the declared JSON Schema type of the parameter.

const CALL = /<tool_call>([\s\S]*?)<\/tool_call>/g;
const PAIR = /<arg_key>([\s\S]*?)<\/arg_key><arg_value>([\s\S]*?)<\/arg_value>/g;

function paramType(tools, fn, key) {
  for (const t of tools || []) {
    const f = t.function || t;
    if (f.name !== fn) continue;
    const def = f.parameters?.properties?.[key];
    if (def && typeof def.type === "string") return def.type;
  }
  return "string";
}

function tryJson(v) {
  try {
    return JSON.parse(v);
  } catch {
    return v;
  }
}

function convert(value, type) {
  if (value.toLowerCase() === "null") return null;
  const t = type.toLowerCase();
  if (["string", "str", "text"].includes(t)) return value;
  if (["integer", "int"].includes(t)) return /^\s*[+-]?\d+\s*$/.test(value) ? parseInt(value, 10) : tryJson(value);
  if (["number", "float"].includes(t)) return value.trim() !== "" && !isNaN(Number(value)) ? Number(value) : tryJson(value);
  if (["boolean", "bool"].includes(t)) {
    const v = value.trim().toLowerCase();
    return ["true", "1", "false", "0"].includes(v) ? ["true", "1"].includes(v) : tryJson(value);
  }
  return tryJson(value);
}

export function parseToolCalls(text, tools) {
  const calls = [];
  for (const m of text.matchAll(CALL)) {
    const body = m[1];
    const first = body.indexOf("<arg_key>");
    const name = (first < 0 ? body : body.slice(0, first)).trim();
    if (!name) continue;
    const args = {};
    for (const [, k, v] of body.matchAll(PAIR)) if (k) args[k] = convert(v, paramType(tools, name, k));
    calls.push({ name, arguments: args });
  }
  const i = text.indexOf("<tool_call>");
  return { content: i < 0 ? text : text.slice(0, i), calls };
}
