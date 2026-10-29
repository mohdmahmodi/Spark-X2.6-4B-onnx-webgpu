// Tokenizer + chat template for Spark-X2.6, shared by the page (via import map) and Node tests.
import { Tokenizer } from "@huggingface/tokenizers";
import { Template } from "@huggingface/jinja";

export async function loadChat(base, fetchJson = (u) => fetch(u).then((r) => r.json())) {
  const [tokenizerJson, config] = await Promise.all([
    fetchJson(`${base}/tokenizer.json`),
    fetchJson(`${base}/tokenizer_config.json`),
  ]);
  const tokenizer = new Tokenizer(tokenizerJson, config);
  const template = new Template(config.chat_template);

  // enable_thinking: the template defaults it to true and ends the prompt with "<think>";
  // false ends it with "</think>" so the model answers directly.
  function render(messages, { thinking = true, effort = "high", tools } = {}) {
    return template.render({
      messages,
      tools,
      add_generation_prompt: true,
      enable_thinking: thinking,
      reasoning_effort: effort,
      bos_token: config.bos_token,
      eos_token: config.eos_token,
    });
  }

  return {
    tokenizer,
    render,
    // the template already contains every special token, so no BOS/EOS is added here
    encodeChat: (messages, opts) => tokenizer.encode(render(messages, opts), { add_special_tokens: false }).ids,
    decode: (ids) => tokenizer.decode(ids, { skip_special_tokens: false }),
  };
}
