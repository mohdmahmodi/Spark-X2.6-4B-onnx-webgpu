// Small SVG chart helpers for the benchmarks page: vertical (grouped) bars and single-series lines.
// Colors come from CSS variables so the same charts render on the dark page and in light model-card images.

const SVG_NS = "http://www.w3.org/2000/svg";

function svg(tag, attrs = {}, text) {
  const node = document.createElementNS(SVG_NS, tag);
  for (const [k, v] of Object.entries(attrs)) node.setAttribute(k, v);
  if (text != null) node.textContent = text;
  return node;
}

let tooltip;
function withTooltip(node, text) {
  tooltip ||= document.getElementById("tooltip");
  if (!tooltip) return;
  node.addEventListener("pointermove", (event) => {
    tooltip.textContent = text;
    tooltip.style.opacity = 1;
    tooltip.style.left = `${Math.min(event.clientX + 14, innerWidth - tooltip.offsetWidth - 8)}px`;
    tooltip.style.top = `${event.clientY - 38}px`;
  });
  node.addEventListener("pointerleave", () => (tooltip.style.opacity = 0));
}

function yAxis(root, { left, right, width, y, ticks, label }) {
  for (const tick of ticks) {
    root.append(svg("line", { x1: left, x2: width - right, y1: y(tick), y2: y(tick), stroke: tick ? "var(--chart-grid)" : "var(--chart-axis)" }));
    root.append(svg("text", { x: left - 10, y: y(tick) + 4, "text-anchor": "end", "font-size": 12, fill: "var(--chart-ink)" }, label(tick)));
  }
}

// Vertical bars. groups: [{ label, sublabel, bars: [{ value, color, text, tip, strong }] }]
export function columnChart(target, groups, { yMax, ticks, tickLabel = String, height = 360, barWidth = 0, gap = 3, labelSize = 0 }) {
  const width = target.clientWidth || 900;
  const labelled = groups.some((g) => g.label);
  const L = 36, R = 4, T = 26, B = labelled ? 56 : 10;
  const y = (v) => T + (1 - v / yMax) * (height - T - B);
  const root = svg("svg", { viewBox: `0 0 ${width} ${height}`, role: "img" });
  yAxis(root, { left: L, right: R, width, y, ticks, label: tickLabel });

  const slot = (width - L - R) / groups.length;
  const perGroup = groups[0].bars.length;
  const bw = barWidth || Math.min(64, (slot * 0.86 - gap * (perGroup - 1)) / perGroup);
  groups.forEach((group, gi) => {
    const groupWidth = perGroup * bw + (perGroup - 1) * gap;
    const x0 = L + gi * slot + (slot - groupWidth) / 2;
    group.bars.forEach((bar, bi) => {
      const x = x0 + bi * (bw + gap);
      const g = svg("g");
      g.append(svg("rect", { x, y: y(bar.value), width: bw, height: y(0) - y(bar.value), rx: 2, fill: bar.color }));
      g.append(svg("rect", { x, y: T, width: bw, height: y(0) - T, fill: "transparent" }));
      g.append(svg("text", { x: x + bw / 2, y: y(bar.value) - 8, "text-anchor": "middle", "font-size": labelSize || (perGroup > 2 ? 12.5 : 14),
        "font-weight": bar.strong ? 600 : 400, fill: bar.strong ? "var(--text)" : "var(--text-2)", "font-variant-numeric": "tabular-nums" }, bar.text));
      if (bar.tip) withTooltip(g, bar.tip);
      root.append(g);
    });
    const cx = L + gi * slot + slot / 2;
    if (group.label) root.append(svg("text", { x: cx, y: height - 30, "text-anchor": "middle", "font-size": 13.5, "font-weight": 500, fill: "var(--text)" }, group.label));
    if (group.sublabel) root.append(svg("text", { x: cx, y: height - 11, "text-anchor": "middle", "font-size": 12.5, fill: "var(--text-3)" }, group.sublabel));
  });
  target.replaceChildren(root);
}

// Single-series line over tokens of context, 0 to 16k.
export function lineChart(target, data, { yMax, ticks, tickLabel = String, pointLabel, tip, firstLabelLift = 0 }) {
  const width = target.clientWidth || 520;
  const height = Math.round(Math.min(300, Math.max(220, width * 0.5)));
  const L = 40, R = 12, T = 26, B = 42, xMax = 16384;
  const x = (v) => L + (v / xMax) * (width - L - R);
  const y = (v) => T + (1 - v / yMax) * (height - T - B);
  const root = svg("svg", { viewBox: `0 0 ${width} ${height}`, role: "img" });
  yAxis(root, { left: L, right: R, width, y, ticks, label: tickLabel });
  for (const tick of [0, 4096, 8192, 12288, 16384]) {
    root.append(svg("text", { x: x(tick), y: height - 20, "text-anchor": tick === 0 ? "start" : tick === xMax ? "end" : "middle",
      "font-size": 12, fill: "var(--chart-ink)" }, tick ? `${tick / 1024}k` : "0"));
  }
  root.append(svg("text", { x: width - R, y: height - 2, "text-anchor": "end", "font-size": 12, fill: "var(--chart-ink)" }, "tokens in context"));
  const points = data.map(([a, b]) => `${x(a).toFixed(1)},${y(b).toFixed(1)}`).join(" ");
  root.append(svg("polyline", { points, fill: "none", stroke: "var(--series-1)", "stroke-width": 2.5, "stroke-linejoin": "round", "stroke-linecap": "round" }));
  data.forEach(([a, b], i) => {
    const g = svg("g");
    g.append(svg("circle", { cx: x(a), cy: y(b), r: 12, fill: "transparent" }));
    g.append(svg("circle", { cx: x(a), cy: y(b), r: 4.5, fill: "var(--series-1)", stroke: "var(--bg)", "stroke-width": 2 }));
    const last = i === data.length - 1;
    if (i === 0 || last) {
      g.append(svg("text", { x: x(a) + (last ? -4 : 8), y: y(b) - 13 - (last ? 0 : firstLabelLift), "text-anchor": last ? "end" : "start", "font-size": 13, "font-weight": 600,
        fill: "var(--text)" }, pointLabel(b)));
    }
    withTooltip(g, tip(a, b));
    root.append(g);
  });
  target.replaceChildren(root);
}
