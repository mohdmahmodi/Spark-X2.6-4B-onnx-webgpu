"""Write chat-template test cases (messages, kwargs, expected ids from transformers) for tools/check_tokenizer.mjs."""
import json
from pathlib import Path

from transformers import AutoTokenizer

SRC = Path("hf/Spark-X2.6-4B")
tok = AutoTokenizer.from_pretrained(SRC)
readme = (SRC / "README.md").read_text(encoding="utf-8")
cases = [
    ([{"role": "user", "content": "What is the capital of France? Answer in one sentence."}], {"enable_thinking": False}),
    ([{"role": "user", "content": "What is the capital of France? Answer in one sentence."}], {"enable_thinking": True}),
    ([{"role": "system", "content": "Reply in French."}, {"role": "user", "content": "Hi! 你好, 1234567 emojis 🤖🚀"},
      {"role": "assistant", "content": "Bonjour !", "reasoning_content": "User greets."}, {"role": "user", "content": "Again?"}],
     {"enable_thinking": True, "reasoning_effort": "low"}),
    ([{"role": "user", "content": "Weather in Paris and 東京 tomorrow?"}],
     {"enable_thinking": False, "tools": [{"type": "function", "function": {"name": "get_weather", "description": "Get the forecast (°C) for a city.",
       "parameters": {"type": "object", "properties": {"city": {"type": "string", "description": "City name, e.g. \"Paris\""}, "days": {"type": "integer"}},
                      "required": ["city"]}}}, {"type": "function", "function": {"name": "noop", "parameters": {"type": "object", "properties": {}}}}]}),
    ([{"role": "user", "content": "Weather in Paris?"},
      {"role": "assistant", "content": "", "tool_calls": [{"type": "function", "function": {"name": "get_weather", "arguments": {"city": "Paris", "days": 1}}}]},
      {"role": "tool", "content": "{\"temp\": 21}"}, {"role": "tool", "content": "second"}],
     {"enable_thinking": True, "tools": [{"type": "function", "function": {"name": "get_weather", "parameters": {"type": "object", "properties": {"city": {"type": "string"}}}}}]}),
    ([{"role": "user", "content": "Summarize this model card in three bullet points.\n\n" + readme}], {"enable_thinking": True}),
]
out = []
for messages, kw in cases:
    text = tok.apply_chat_template(messages, add_generation_prompt=True, tokenize=False, **kw)
    kw = {k: v for k, v in kw.items()}
    out.append({"messages": messages, "kwargs": kw, "text": text, "ids": tok(text, add_special_tokens=False).input_ids})
Path("export/tokenizer_cases.json").write_text(json.dumps(out, ensure_ascii=False), encoding="utf-8")
print(len(out), "cases", [len(c["ids"]) for c in out])
