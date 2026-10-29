// Icons and code blocks shared by the chat app and the project page: Prism highlighting (languages load
// on demand from the CDN), a language label and a copy button.

export const icon = (name, cls = "icon") => `<svg class="${cls}" aria-hidden="true"><use href="assets/icons.svg#${name}"></use></svg>`;

if (window.Prism?.plugins?.autoloader) {
  window.Prism.plugins.autoloader.languages_path = "https://cdnjs.cloudflare.com/ajax/libs/prism/1.29.0/components/";
}

const LANGUAGE_NAMES = { js: "JavaScript", javascript: "JavaScript", ts: "TypeScript", typescript: "TypeScript", py: "Python",
  python: "Python", sh: "Shell", bash: "Shell", shell: "Shell", html: "HTML", css: "CSS", json: "JSON", cpp: "C++", c: "C",
  rust: "Rust", go: "Go", java: "Java", sql: "SQL", yaml: "YAML", markup: "HTML" };

function copyButton(code) {
  const button = document.createElement("button");
  button.className = "copy-btn";
  button.type = "button";
  button.innerHTML = `${icon("copy")}<span>Copy</span>`;
  button.onclick = async () => {
    try {
      await navigator.clipboard.writeText(code.textContent);
      button.innerHTML = `${icon("check")}<span>Copied</span>`;
      setTimeout(() => (button.innerHTML = `${icon("copy")}<span>Copy</span>`), 1600);
    } catch {
      button.querySelector("span").textContent = "Press Ctrl+C";
    }
  };
  return button;
}

// Wraps every <pre><code> under `root` with a header and highlights it. `highlight: false` keeps the
// frame but skips Prism, which the chat uses while a reply is still streaming.
export function decorateCode(root, { highlight = true } = {}) {
  for (const pre of root.querySelectorAll("pre")) {
    const code = pre.querySelector("code");
    if (!code || pre.parentElement.classList.contains("code-block")) continue;
    const lang = (code.className.match(/language-([\w+-]+)/) || [])[1]?.toLowerCase() || "";
    const frame = document.createElement("div");
    frame.className = "code-block";
    const head = document.createElement("div");
    head.className = "code-head";
    head.append(Object.assign(document.createElement("span"), { textContent: LANGUAGE_NAMES[lang] || lang || "Code" }));
    head.append(copyButton(code));
    pre.replaceWith(frame);
    frame.append(head, pre);
    if (highlight && lang && window.Prism) window.Prism.highlightElement(code);
  }
}
