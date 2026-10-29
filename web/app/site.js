// Benchmarks page: key numbers, XHToken's comparison with other models, this build against the original,
// validation table and speed charts. The numbers live in data.js with their sources.
import { columnChart, lineChart } from "./charts.js";
import { decorateCode, icon } from "./code.js";
import { ACCURACY, ALL_BENCHMARKS, DECODE, FIGURES, FIRST_TOKEN, HISTORY, MODEL_COLORS, MODELS, QUALITY, SUMMARY, SYSTEMS } from "./data.js";

const $ = (id) => document.getElementById(id);
const el = (tag, className, text) => Object.assign(document.createElement(tag), className ? { className } : {}, text != null ? { textContent: text } : {});
const formatTokens = (n) => n.toLocaleString("en-US");
const formatSeconds = (s) => (s < 1 ? `${Math.round(s * 1000)} ms` : `${s.toFixed(1)} s`);
// the model-card images are rendered on a light background (tools/shots.mjs passes ?theme=light)
if (new URLSearchParams(location.search).get("theme") === "light") document.body.classList.add("light");

const legend = (items) => items.map(([name, color]) => `<span><i style="background: ${color}"></i>${name}</span>`).join("");

// smallest round axis maximum above the data, with four or five ticks
function niceAxis(max) {
  const step = [5, 10, 20, 25].find((s) => max / s <= 4.2) ?? 25;
  const top = Math.ceil((max * 1.08) / step) * step;
  return { yMax: top, ticks: Array.from({ length: top / step + 1 }, (_, i) => i * step) };
}

// ------------------------------------------------------------------ static content

for (const node of document.querySelectorAll("[data-figures]")) {
  node.innerHTML = FIGURES.map(([label, value, unit, note]) =>
    `<div><div class="label">${label}</div><div class="value">${value}<small>${unit}</small></div><div class="note">${note}</div></div>`).join("");
}

const multiples = QUALITY.map((bench) => {
  const cell = el("div");
  cell.append(el("h3", null, bench.name), el("p", "task", bench.task), el("div", "chart"));
  $("quality-grid").append(cell);
  return { bench, chart: cell.querySelector(".chart") };
});
$("quality-legend").innerHTML = legend(MODELS.map((m, i) => [m, MODEL_COLORS[i]]));

{
  const split = (name) => name.replace("-", "<br />"); // "Spark-X2.6-4B" -> "Spark / X2.6-4B", as in XHToken's table
  const head = `<thead><tr><th>Benchmark</th>${MODELS.map((m, i) => `<th class="${i ? "" : "ours"}">${split(m)}</th>`).join("")}</tr></thead>`;
  const body = ALL_BENCHMARKS.map(([group, rows]) => {
    const groupRow = group === "Average" ? "" : `<tr class="group"><td>${group}</td>${MODELS.map((_, i) => `<td class="${i ? "" : "ours"}"></td>`).join("")}</tr>`;
    return groupRow + rows.map(([name, ...cells]) => {
      const best = Math.max(...cells.map((c) => (c ? parseFloat(c) : -Infinity)));
      const tds = cells.map((c, i) => {
        const cls = [i === 0 ? "ours" : "", i > 0 && c && parseFloat(c) === best ? "best" : ""].filter(Boolean).join(" ");
        return `<td${cls ? ` class="${cls}"` : ""}>${c ?? "–"}</td>`;
      }).join("");
      return `<tr${group === "Average" ? ' class="average"' : ""}><td>${name}</td>${tds}</tr>`;
    }).join("");
  }).join("");
  $("quality-table").innerHTML = `${head}<tbody>${body}</tbody>`;
}

// bf16 reference, official INT4 weights, this build
const PARITY_COLORS = { bf16: "#2959d3", int4: "#bedcf7", web: "#f87d19" };
$("acc-legend").innerHTML = legend(SYSTEMS.map((s) => [s.name, PARITY_COLORS[s.key]]));

$("summary-table").innerHTML = "<thead><tr><th>Check</th><th>bf16</th><th>INT4</th><th>This build</th><th></th></tr></thead><tbody>" +
  SUMMARY.map(([check, bf16, int4, ours]) => `<tr><td>${check}</td><td>${bf16}</td><td>${int4}</td><td><strong>${ours}</strong></td>` +
    `<td><span class="pass">${icon("check")}Pass</span></td></tr>`).join("") + "</tbody>";

decorateCode(document);

// ------------------------------------------------------------------ charts (redrawn on resize)

function draw() {
  for (const { bench, chart } of multiples) {
    columnChart(chart, [{ bars: bench.scores.map((v, i) => ({ value: v, color: MODEL_COLORS[i], strong: i === 0, text: v.toFixed(1),
      tip: `${MODELS[i]}: ${v.toFixed(1)}` })) }], { ...niceAxis(Math.max(...bench.scores)), height: 230, gap: 3, labelSize: 11 });
  }

  columnChart($("acc-chart"), ACCURACY.map((bench) => ({
    label: bench.name,
    sublabel: bench.detail,
    bars: SYSTEMS.map((s) => {
      const pct = (100 * bench[s.key]) / bench.n;
      return { value: pct, color: PARITY_COLORS[s.key], strong: s.ours, text: bench.count ? `${bench[s.key]}/${bench.n}` : `${pct.toFixed(1)}%`,
        tip: `${s.name}: ${bench[s.key]} of ${bench.n}` };
    }),
  })), { yMax: 100, ticks: [0, 25, 50, 75, 100], tickLabel: (v) => `${v}%` });

  const historyColors = ["#fcd9bd", "#fcd9bd", "#fbb47c", "#f87d19"];
  columnChart($("history-chart"), HISTORY.map((h, i) => ({
    label: h.step,
    sublabel: h.detail,
    bars: [{ value: h.value, color: historyColors[i], strong: h.ours, text: `${h.value} tok/s`, tip: `${h.step}: ${h.value} tok/s` }],
  })), { yMax: 100, ticks: [0, 25, 50, 75, 100], height: 340, barWidth: 96 });

  lineChart($("tps-chart"), DECODE, { yMax: 100, ticks: [0, 25, 50, 75, 100], pointLabel: (v) => `${v.toFixed(0)} tok/s`,
    tip: (a, b) => `${formatTokens(a)} tokens of context: ${b.toFixed(1)} tok/s` });
  lineChart($("ttft-chart"), FIRST_TOKEN, { yMax: 16, ticks: [0, 4, 8, 12, 16], pointLabel: formatSeconds, firstLabelLift: 12,
    tip: (a, b) => `${formatTokens(a)}-token prompt: ${formatSeconds(b)}` });
}

// ------------------------------------------------------------------ screenshot mode and layout

const params = new URLSearchParams(location.search);
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
