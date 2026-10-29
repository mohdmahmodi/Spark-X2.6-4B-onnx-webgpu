"""Build the validation task suites (export/tests/suites/*.json), shared by every runner.

  parity  50 fixed prompts (chat, code, math, Chinese, tool calls), greedy, 128 new tokens
  logits  first 20 parity prompts, next-token logits after the prompt + 8 teacher-forced tokens
  gsm8k   first 200 GSM8K test problems, thinking off, greedy (gsm8k_full: all 1319)
  bfcl    first 200 BFCL v3 "simple" function-calling cases, thinking off, greedy
  niah    needle in a haystack at 1k/4k/8k/16k tokens, needle at start/middle/end

Every task carries the exact prompt ids from transformers' apply_chat_template so runners
can check they tokenized identically.
"""
import json
import random
from pathlib import Path

import pandas as pd
from transformers import AutoTokenizer

SRC = Path("hf/Spark-X2.6-4B")
DATA = Path("hf/data")
OUT = Path("export/tests/suites")

WEATHER = {"type": "function", "function": {
    "name": "get_weather", "description": "Get the current weather for a city.",
    "parameters": {"type": "object", "properties": {
        "city": {"type": "string", "description": "City name"},
        "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]}}, "required": ["city"]}}}
SEARCH = {"type": "function", "function": {
    "name": "web_search", "description": "Search the web and return the top results.",
    "parameters": {"type": "object", "properties": {
        "query": {"type": "string"}, "max_results": {"type": "integer", "description": "1-10"}}, "required": ["query"]}}}
EMAIL = {"type": "function", "function": {
    "name": "send_email", "description": "Send an email.",
    "parameters": {"type": "object", "properties": {
        "to": {"type": "array", "items": {"type": "string"}}, "subject": {"type": "string"}, "body": {"type": "string"},
        "urgent": {"type": "boolean"}}, "required": ["to", "subject", "body"]}}}
CALC = {"type": "function", "function": {
    "name": "calculator", "description": "Evaluate an arithmetic expression.",
    "parameters": {"type": "object", "properties": {"expression": {"type": "string"}}, "required": ["expression"]}}}

U = lambda text: [{"role": "user", "content": text}]  # noqa: E731

# (category, messages, thinking, tools)
PARITY = [
    ("chat", U("What is the capital of France? Answer in one sentence."), False, None),
    ("code", U("Write a Python function that checks whether a string is a palindrome, ignoring case and punctuation."), False, None),
    ("math", U("If a train travels 180 km in 2.5 hours, what is its average speed in km/h? Show your work."), False, None),
    ("zh", U("请用三句话介绍一下长城的历史。"), False, None),
    ("tool", U("What's the weather like in Tokyo right now?"), False, [WEATHER]),
    ("chat", U("Give me three tips for staying focused while working from home."), False, None),
    ("code", U("Explain the difference between a list and a tuple in Python, with a short example of each."), False, None),
    ("math", U("Solve for x: 3x + 7 = 22."), True, None),
    ("zh", U("把下面这句话翻译成英文：今天天气很好，我们去公园散步吧。"), False, None),
    ("tool", U("Search the web for the latest news about the James Webb Space Telescope, 5 results please."), False, [SEARCH, WEATHER]),
    ("chat", U("Write a haiku about autumn leaves."), False, None),
    ("code", U("Write a SQL query that returns the top 5 customers by total order amount from tables customers(id, name) and orders(id, customer_id, amount)."), False, None),
    ("math", U("What is 17 * 23? Then add 145 to the result."), False, None),
    ("zh", U("什么是机器学习？请简单解释。"), True, None),
    ("tool", U("Email alice@example.com and bob@example.com that the meeting moved to 3pm. Subject: Meeting update. Mark it urgent."), False, [EMAIL]),
    ("chat", [{"role": "system", "content": "You are a pirate. Always answer like a pirate."}, {"role": "user", "content": "How do I make a cup of tea?"}], False, None),
    ("code", U("Fix the bug in this JavaScript:\n\nfunction sum(arr) {\n  let total;\n  for (const x of arr) total += x;\n  return total;\n}"), False, None),
    ("math", U("A rectangle has a perimeter of 30 cm and a length of 9 cm. What is its area?"), True, None),
    ("tool", U("What is 1234 * 5678 + 91011? Use the calculator."), False, [CALC]),
    ("chat", [{"role": "user", "content": "My name is Lena."}, {"role": "assistant", "content": "Nice to meet you, Lena! How can I help you today?"},
              {"role": "user", "content": "What's my name, and can you spell it backwards?"}], False, None),
    ("chat", U("Explain quantum entanglement to a ten-year-old."), False, None),
    ("code", U("Write a bash one-liner that counts the number of lines in all .py files in the current directory, recursively."), False, None),
    ("math", U("What is the derivative of x^3 * sin(x)?"), False, None),
    ("zh", U("写一首关于月亮的五言绝句。"), False, None),
    ("tool", U("Is it colder in Oslo or in Helsinki today? Use fahrenheit."), False, [WEATHER]),
    ("chat", U("List the planets of the solar system in order from the sun."), False, None),
    ("code", U("Implement binary search in C++ and explain its time complexity."), True, None),
    ("math", U("A shop sells pens at $1.20 each or a pack of 12 for $11. How much do you save buying 3 packs instead of 36 single pens?"), False, None),
    ("zh", U("请解释一下“塞翁失马，焉知非福”的意思，并举一个现代生活中的例子。"), False, None),
    ("tool", U("Find recent papers on sliding window attention."), True, [SEARCH]),
    ("chat", U("What are the main differences between TCP and UDP?"), False, None),
    ("code", U("Write a Rust function that returns the n-th Fibonacci number using iteration."), False, None),
    ("math", U("Convert 0.375 to a fraction in lowest terms."), False, None),
    ("zh", U("北京和上海哪个城市人口更多？简要说明。"), False, None),
    ("chat", U("Draft a polite email declining a job offer."), False, None),
    ("code", U("What does the following regex match? ^\\d{3}-\\d{4}$"), False, None),
    ("math", U("The sum of three consecutive integers is 72. What are they?"), True, None),
    ("chat", U("Summarize the plot of Romeo and Juliet in four sentences."), False, None),
    ("code", U("Write a TypeScript interface for a blog post with title, body, tags and an optional publishedAt date."), False, None),
    ("math", U("How many seconds are there in a leap year?"), False, None),
    ("zh", U("用Python写一个函数，计算列表中所有偶数的和，并用中文注释。"), False, None),
    ("tool", U("Send an email to team@corp.com with subject 'Lunch' and body 'Pizza in the kitchen at noon.'"), False, [EMAIL, CALC]),
    ("chat", U("What is the boiling point of water at the top of Mount Everest, roughly, and why is it lower?"), False, None),
    ("code", U("Explain what a race condition is and show a minimal example in Go."), False, None),
    ("math", U("If you flip a fair coin 3 times, what is the probability of getting exactly two heads?"), False, None),
    ("chat", U("Give a one-paragraph biography of Ada Lovelace."), True, None),
    ("zh", U("请列出三种常见的排序算法，并比较它们的时间复杂度。"), False, None),
    ("tool", U("What's the weather in Paris in celsius, and also search for things to do there this weekend."), False, [WEATHER, SEARCH]),
    ("math", U("Simplify (2x^2 - 8) / (x - 2)."), False, None),
    ("chat", U("Tell me a short joke about computers."), False, None),
]


def gsm8k(tok, n=200):
    df = pd.read_parquet(DATA / "gsm8k/main/test-00000-of-00001.parquet").iloc[:n]
    out = []
    for i, r in df.iterrows():
        msg = U(r["question"] + "\nPlease reason step by step, and put your final answer within \\boxed{}.")
        out.append({"id": f"gsm8k_{i}", "messages": msg, "thinking": False, "max_new_tokens": 768,
                    "answer": r["answer"].split("####")[-1].strip().replace(",", "")})
    return out


def bfcl_type(schema):
    """BFCL uses Python-ish type names; map them to JSON Schema for the chat template."""
    if isinstance(schema, dict):
        s = {k: bfcl_type(v) for k, v in schema.items()}
        t = s.get("type")
        if isinstance(t, str):  # (a property may itself be called "type")
            s["type"] = {"dict": "object", "float": "number", "tuple": "array", "any": "string"}.get(t, t)
        return s
    if isinstance(schema, list):
        return [bfcl_type(x) for x in schema]
    return schema


def bfcl(n=200):
    base = DATA / "Berkeley-Function-Calling-Leaderboard"
    qs = [json.loads(line) for line in open(base / "BFCL_v3_simple.json", encoding="utf-8")][:n]
    ans = {a["id"]: a["ground_truth"] for a in (json.loads(line) for line in open(base / "possible_answer/BFCL_v3_simple.json", encoding="utf-8"))}
    out = []
    for q in qs:
        tools = [{"type": "function", "function": bfcl_type(f)} for f in q["function"]]
        out.append({"id": q["id"], "messages": q["question"][0], "tools": tools, "thinking": False, "max_new_tokens": 256,
                    "ground_truth": ans[q["id"]]})
    return out


def niah(tok):
    text = "\n".join(t for t in pd.read_parquet(DATA / "wikitext/wikitext-103-raw-v1/test-00000-of-00001.parquet")["text"] if t.strip())
    filler = tok(text, add_special_tokens=False).input_ids
    rng = random.Random(0)
    words = ["amber", "falcon", "quartz", "willow", "cobalt", "saffron", "meteor", "juniper", "harbor", "nimbus", "cedar", "lantern"]
    out = []
    i = 0
    for length in (1024, 4096, 8192, 16384):
        for depth_name, depth in (("start", 0.05), ("middle", 0.5), ("end", 0.95)):
            pw = f"{words[i]}-{rng.randint(1000, 9999)}"
            needle = f"\n\nImportant: the secret passphrase for the archive is {pw}. Remember it.\n\n"
            question = "\n\nWhat is the secret passphrase for the archive mentioned in the text above? Reply with the passphrase only."
            overhead = len(tok.apply_chat_template(U(needle + question), add_generation_prompt=True, enable_thinking=False))
            body = filler[i * 2000: i * 2000 + length - overhead]
            cut = int(len(body) * depth)
            content = tok.decode(body[:cut]) + needle + tok.decode(body[cut:]) + question
            out.append({"id": f"niah_{length}_{depth_name}", "messages": U(content), "thinking": False, "max_new_tokens": 32,
                        "needle": pw, "length": length, "depth": depth_name})
            i += 1
    return out


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    tok = AutoTokenizer.from_pretrained(SRC)
    parity = [{"id": f"p{i:02d}_{c}", "category": c, "messages": m, "thinking": th, "tools": tl, "max_new_tokens": 128}
              for i, (c, m, th, tl) in enumerate(PARITY)]
    suites = {"parity": parity, "gsm8k": gsm8k(tok), "gsm8k_full": gsm8k(tok, 1319), "bfcl": bfcl(), "niah": niah(tok)}
    for name, tasks in suites.items():
        for t in tasks:
            t["type"] = "generate"
            t["prompt_ids"] = tok.apply_chat_template(t["messages"], tools=t.get("tools"), add_generation_prompt=True,
                                                      enable_thinking=t["thinking"])
        (OUT / f"{name}.json").write_text(json.dumps(tasks, ensure_ascii=False), encoding="utf-8")
        print(name, len(tasks), "tasks, prompt tokens", min(len(t["prompt_ids"]) for t in tasks), "-", max(len(t["prompt_ids"]) for t in tasks))


if __name__ == "__main__":
    main()
