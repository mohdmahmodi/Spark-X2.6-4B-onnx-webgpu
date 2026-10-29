// Checks web/lib/chat.js (tokenizers.js + jinja) against transformers' apply_chat_template + tokenizer.
import { readFileSync } from "node:fs";
import { loadChat } from "../web/lib/chat.js";

const chat = await loadChat("web/spark", async (p) => JSON.parse(readFileSync(p, "utf8")));
const cases = JSON.parse(readFileSync("export/tokenizer_cases.json", "utf8"));
let ok = true;
for (const [i, c] of cases.entries()) {
  const text = chat.render(c.messages, { thinking: c.kwargs.enable_thinking, effort: c.kwargs.reasoning_effort, tools: c.kwargs.tools });
  const ids = chat.encodeChat(c.messages, { thinking: c.kwargs.enable_thinking, effort: c.kwargs.reasoning_effort, tools: c.kwargs.tools });
  const sameText = text === c.text;
  const firstDiff = ids.findIndex((v, j) => v !== c.ids[j]);
  const sameIds = ids.length === c.ids.length && firstDiff === -1;
  const roundTrip = chat.decode(c.ids) === c.text;
  ok &&= sameText && sameIds && roundTrip;
  console.log(`case ${i}: template ${sameText ? "same" : "DIFFERENT"}, ids ${sameIds ? "same" : `DIFFERENT at ${firstDiff}`} (${ids.length} vs ${c.ids.length}), decode round-trip ${roundTrip ? "ok" : "DIFFERENT"}`);
  if (!sameText) console.log(JSON.stringify(text.slice(0, 300)), "\n", JSON.stringify(c.text.slice(0, 300)));
}
process.exit(ok ? 0 : 1);
