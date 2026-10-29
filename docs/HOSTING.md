# Hosting

The app is a static site and the model is a folder of static files, so both can be hosted anywhere that serves files
over HTTPS with CORS enabled. This guide uses Hugging Face, where both are free.

## Model repository

`web/spark/` is the complete model repository: ONNX graphs, tokenizer, chat template, config, license and model card.
Build it with `convert/export_onnx.py` and `convert/build_web.py` (see the README), then:

```bash
hf auth login
```

```bash
hf repo create <user>/Spark-X2.6-4B-ONNX-WebGPU --repo-type model
```

```bash
hf upload <user>/Spark-X2.6-4B-ONNX-WebGPU web/spark . --repo-type model
```

The upload is about 5.8 GB with both builds and resumes when run again after an interruption. `.gitattributes` keeps
the ONNX files, external data, tokenizer and images in large-file storage. The model card's `base_model` metadata lists
the repository under the original model's quantizations.

## App as a static Space

```bash
python tools/make_space.py --model <user>/Spark-X2.6-4B-ONNX-WebGPU
```

```bash
hf repo create <user>/Spark-X2.6-WebGPU --repo-type space --space_sdk static
```

```bash
hf upload <user>/Spark-X2.6-WebGPU release/space . --repo-type space
```

`make_space.py` copies the page and its modules, points the page at the model repository and writes the Space card.
The model repository must be public, because visitors' browsers download it without signing in. A static Space runs
no server code; inference happens on each visitor's GPU.

## Your own website

```bash
python tools/make_space.py --out release/site --site-url https://example.com/spark --no-space-card
```

Copy `release/site/` to that path on the server. Every link inside the folder is relative, the model still loads from
the Hugging Face repository, and `--site-url` sets the canonical and share links of the pages.

## Other hosts

Serve `web/` as the site root and either place the model folder at `web/spark/` or pass its URL with
`?model=https://host/path/to/folder`. The model host must send `Access-Control-Allow-Origin` when it is on a different
origin, and must not cap single files below 1.9 GB.

## Checks after publishing

- First visit: the model downloads once (2.77 GB). A reload should load from the browser cache in a few seconds.
- A short reply, a reply with Thinking on, and the benchmarks page all work.
- Try at least one other machine. Results so far come from one Windows PC with an RTX 3090 Ti.
