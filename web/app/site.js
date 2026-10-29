// Benchmarks page: figures, model comparison, accuracy against the original, validation table, speed charts.
// The numbers live in data.js with their sources.
import { columnChart, lineChart } from "./charts.js";
import { decorateCode, icon } from "./code.js";
import { ACCURACY, ALL_BENCHMARKS, DECODE, FIGURES, FIRST_TOKEN, HISTORY, MODELS, QUALITY, SUMMARY, SYSTEMS } from "./data.js";

const $ = (id) => document.getElementById(id);
const el = (tag, className, text) => Object.assign(document.createElement(tag), className ? { className } : {}, text != null ? { textContent: text } : {});
const formatTokens = (n) => n.toLocaleString("en-US");
const formatSeconds = (s) => (s < 1 ? `${Math.round(s * 1000)} ms` : `${s.toFixed(1)} s`);
const params = new URLSearchParams(location.search);
if (params.get("theme") === "light") document.body.classList.add("light");

// ------------------------------------------------------------------ figures

for (const node of document.querySelectorAll("[data-figures]")) {
  node.innerHTML = FIGURES.map(([label, value, unit, note]) =>
    `<div class="figure"><div class="label">${label}</div><div class="value">${value}<small>${unit}</small></div><div class="note">${note}</div></div>`).join("");
}

// ------------------------------------------------------------------ model quality (reported by XHToken)

// Horizontal bars, one row per model, sorted as given; Spark-X2.6-4B leads, Spark-X2.5-4B is its predecessor.
function modelBars(scores, { max = 100, large = false } = {}) {
  const list = el("div", `hbars${large ? " large" : ""}`);
  MODELS.forEach((model, i) => {
    const score = scores[i];
    const row = el("div", `hbar${i === 0 ? " lead" : i === 1 ? " prev" : ""}`);
    const track = el("div", "track");
    const fill = el("div", "fill");
    fill.style.width = `${(100 * score) / max}%`;
    track.append(fill);
    row.append(el("span", "who", model), track, el("span", "val", score.toFixed(1)));
    row.title = `${model}: ${score.toFixed(1)}`;
    list.append(row);
  });
  return list;
}

const average = ALL_BENCHMARKS[0][1][0].slice(1).map(Number);
$("quality-average").replaceWith(Object.assign(modelBars(average, { large: true }), { id: "quality-average" }));

for (const bench of QUALITY) {
  const cell = el("div");
  cell.append(el("h3", null, bench.name), el("p", "task", bench.task), modelBars(bench.scores));
  $("quality-grid").append(cell);
}

{
  const head = `<thead><tr><th>Benchmark</th>${MODELS.map((m) => `<th class="num">${m}</th>`).join("")}</tr></thead>`;
  const body = ALL_BENCHMARKS.map(([group, rows]) => {
    const groupRow = group === "Average" ? "" : `<tr class="group"><td colspan="${MODELS.length + 1}">${group}</td></tr>`;
    return groupRow + rows.map(([name, ...cells]) => {
      const best = Math.max(...cells.map((c) => (c ? parseFloat(c) : -Infinity)));
      return `<tr><td>${name}</td>${cells.map((c) => `<td class="num${c && parseFloat(c) === best ? " best" : ""}">${c ?? ""}</td>`).join("")}</tr>`;
    }).join("");
  }).join("");
  $("quality-table").innerHTML = `${head}<tbody>${body}</tbody>`;
}

// ------------------------------------------------------------------ this build against the original

const CATEGORY_COLORS = { bf16: "var(--cat-1)", int4: "var(--cat-2)", web: "var(--cat-3)" };
$("acc-legend").innerHTML = SYSTEMS.map((s) => `<span><i style="background: ${CATEGORY_COLORS[s.key]}"></i>${s.name}</span>`).join("");

$("summary-table").innerHTML = "<thead><tr><th>Check</th><th>bf16</th><th>INT4</th><th>This build</th><th></th></tr></thead><tbody>" +
  SUMMARY.map(([check, bf16, int4, ours]) => `<tr><td>${check}</td><td class="tab">${bf16}</td>` +
    `<td class="tab">${int4}</td><td class="tab"><strong>${ours}</strong></td>` +
    `<td><span class="pass">${icon("check")}Pass</span></td></tr>`).join("") + "</tbody>";

decorateCode(document);

// ------------------------------------------------------------------ charts (redrawn on resize)

function draw() {
  columnChart($("acc-chart"), ACCURACY.map((bench) => ({
    label: bench.name,
    sublabel: bench.detail,
    bars: SYSTEMS.map((s) => {
      const pct = (100 * bench[s.key]) / bench.n;
      return { value: pct, color: CATEGORY_COLORS[s.key], strong: s.ours, text: bench.count ? `${bench[s.key]}/${bench.n}` : `${pct.toFixed(1)}%`,
        tip: `${s.name}: ${bench[s.key]} of ${bench.n}` };
    }),
  })), { yMax: 100, ticks: [0, 25, 50, 75, 100], tickLabel: (v) => `${v}%` });

  const historyColors = ["var(--series-2)", "var(--series-2)", "var(--series-2)", "var(--series-1)"];
  columnChart($("history-chart"), HISTORY.map((h, i) => ({
    label: h.step,
    sublabel: h.detail,
    bars: [{ value: h.value, color: historyColors[i], strong: h.ours, text: `${h.value} tok/s`, tip: `${h.step}: ${h.value} tok/s` }],
  })), { yMax: 100, ticks: [0, 25, 50, 75, 100], height: 340, barWidth: 88 });

  lineChart($("tps-chart"), DECODE, { yMax: 100, ticks: [0, 25, 50, 75, 100], pointLabel: (v) => `${v.toFixed(0)} tok/s`,
    tip: (a, b) => `${formatTokens(a)} tokens of context: ${b.toFixed(1)} tok/s` });
  lineChart($("ttft-chart"), FIRST_TOKEN, { yMax: 16, ticks: [0, 4, 8, 12, 16], pointLabel: formatSeconds, firstLabelLift: 12,
    tip: (a, b) => `${formatTokens(a)}-token prompt: ${formatSeconds(b)}` });
}

// ------------------------------------------------------------------ screenshot mode and layout

if (params.has("shot")) {
  document.body.classList.add("shot");
  for (const id of params.get("shot").split(",")) {
    const node = $(id);
    node?.classList.add("shoot");
    // keep the ancestors visible so the element can be screenshotted on its own
    for (let p = node?.parentElement; p && p !== document.body; p = p.parentElement) p.classList.add("shoot-path");
  }
}

let lastWidth = 0;
new ResizeObserver(() => {
  const w = document.querySelector(".page").clientWidth;
  if (w !== lastWidth) {
    lastWidth = w;
    draw();
  }
}).observe(document.querySelector(".page"));
document.fonts.ready.then(() => {
  draw();
  document.body.dataset.ready = "1";
});
