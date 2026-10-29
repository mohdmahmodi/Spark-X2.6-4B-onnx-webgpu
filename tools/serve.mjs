// Static server for web/ (python -m http.server works too; this one streams big files and sets MIME types).
import { createServer } from "node:http";
import { createReadStream, statSync } from "node:fs";
import { extname, join, normalize } from "node:path";

const root = normalize(join(import.meta.dirname, "..", "web"));
const port = Number(process.env.PORT || 8000);
const types = {
  ".html": "text/html; charset=utf-8", ".css": "text/css; charset=utf-8", ".js": "text/javascript", ".mjs": "text/javascript",
  ".json": "application/json", ".svg": "image/svg+xml", ".png": "image/png", ".txt": "text/plain; charset=utf-8",
  ".wasm": "application/wasm", ".onnx": "application/octet-stream",
};

createServer((req, res) => {
  const path = normalize(join(root, decodeURIComponent(new URL(req.url, "http://x").pathname)));
  if (!path.startsWith(root)) return res.writeHead(403).end();
  let file = path;
  try {
    if (statSync(file).isDirectory()) file = join(file, "index.html");
    const { size } = statSync(file);
    res.writeHead(200, { "content-type": types[extname(file)] || "application/octet-stream", "content-length": size, "cache-control": "no-cache" });
    createReadStream(file).pipe(res);
  } catch {
    res.writeHead(404).end();
  }
}).listen(port, () => console.log(`serving ${root} on http://localhost:${port}`));
