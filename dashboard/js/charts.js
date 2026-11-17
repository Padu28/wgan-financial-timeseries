/* Tempora: small SVG chart toolkit (no dependencies). */
(function () {
  "use strict";
  const NS = "http://www.w3.org/2000/svg";
  function el(name, attrs, parent) {
    const n = document.createElementNS(NS, name);
    for (const k in attrs || {}) n.setAttribute(k, attrs[k]);
    if (parent) parent.appendChild(n);
    return n;
  }
  function svg(container, w, h) { container.innerHTML = ""; return el("svg", { viewBox: `0 0 ${w} ${h}`, role: "img" }, container); }
  const pct = (x, d) => (x * 100).toFixed(d == null ? 1 : d) + "%";
  const signed = (x, d) => (x >= 0 ? "+" : "−") + pct(Math.abs(x), d);

  function niceTicks(lo, hi, n) {
    const span = hi - lo || 1, raw = span / (n || 5), p = Math.pow(10, Math.floor(Math.log10(raw)));
    const step = [1, 2, 2.5, 5, 10].map(m => m * p).find(c => span / c <= (n || 5)) || 10 * p;
    const out = [];
    for (let v = Math.ceil(lo / step) * step; v <= hi + 1e-9; v += step) out.push(+v.toFixed(10));
    return out;
  }
  function extent(arrs) {
    let lo = Infinity, hi = -Infinity;
    arrs.forEach(a => a.forEach(v => { if (v != null && isFinite(v)) { lo = Math.min(lo, v); hi = Math.max(hi, v); } }));
    return [lo, hi];
  }
  const line = pts => pts.map((p, i) => (i ? "L" : "M") + p[0].toFixed(1) + "," + p[1].toFixed(1)).join("");
  function setLen(p, delay, dur) {
    if (!p.getTotalLength) return;
    p.style.setProperty("--len", Math.ceil(p.getTotalLength() + 4));
    if (delay != null) p.style.setProperty("--delay", delay + "s");
    if (dur != null) p.style.setProperty("--dur", dur + "s");
    p.classList.add("draw");
  }

  /* frame: axes + grid, returns scales */
  function frame(s, o) {
    const { W, H, L, R, T, B } = o;
    const [x0, x1] = o.x, [y0, y1] = o.y;
    const X = v => L + (v - x0) / (x1 - x0 || 1) * (W - L - R);
    const Y = o.ylog ? (v => T + (1 - (Math.log10(Math.max(v, o.yfloor)) - Math.log10(y0)) / (Math.log10(y1) - Math.log10(y0))) * (H - T - B))
      : (v => T + (1 - (v - y0) / (y1 - y0 || 1)) * (H - T - B));
    const g = el("g", { class: "grid" }, s);
    const yt = o.yticks || (o.ylog ? [] : niceTicks(y0, y1, o.ny || 4));
    yt.forEach(v => { el("line", { x1: L, x2: W - R, y1: Y(v), y2: Y(v) }, g); el("text", { x: L - 6, y: Y(v) + 4, "text-anchor": "end" }, s).textContent = o.yfmt ? o.yfmt(v) : v; });
    (o.xticks || niceTicks(x0, x1, o.nx || 6)).forEach(v => { el("text", { x: X(v), y: H - B + 15, "text-anchor": "middle" }, s).textContent = o.xfmt ? o.xfmt(v) : v; });
    if (o.xlabel) el("text", { x: (W + L - R) / 2, y: H - 2, "text-anchor": "middle" }, s).textContent = o.xlabel;
    return { X, Y };
  }

  /* tooltip */
  let tipEl;
  function tip(html, e) {
    tipEl = tipEl || document.getElementById("tip");
    if (!html) { tipEl.classList.remove("show"); return; }
    tipEl.innerHTML = html;
    tipEl.style.left = Math.min(e.clientX + 14, innerWidth - tipEl.offsetWidth - 8) + "px";
    tipEl.style.top = Math.min(e.clientY + 14, innerHeight - tipEl.offsetHeight - 8) + "px";
    tipEl.classList.add("show");
  }
  function bindTip(node, html) { node.addEventListener("mousemove", e => tip(typeof html === "function" ? html(e) : html, e)); node.addEventListener("mouseleave", () => tip(null)); }

  /* generic multi-line chart */
  function lines(container, o) {
    const W = o.W || 640, H = o.H || 260, L = o.L || 46, R = o.R || 14, T = o.T || 12, B = o.B || 30;
    const s = svg(container, W, H);
    const xs = o.x || o.series[0].values.map((_, i) => i);
    let [y0, y1] = o.y || extent(o.series.map(x => x.values).concat(o.refs ? [o.refs.map(r => r.v)] : []));
    if (!o.y) { const pad = (y1 - y0) * 0.08 || 0.1; y0 -= pad; y1 += pad; }
    const sc = frame(s, { W, H, L, R, T, B, x: [xs[0], xs[xs.length - 1]], y: [y0, y1], yfmt: o.yfmt, xfmt: o.xfmt, xticks: o.xticks, ny: o.ny, xlabel: o.xlabel });
    (o.refs || []).forEach(r => el("line", { x1: L, x2: W - R, y1: sc.Y(r.v), y2: sc.Y(r.v), stroke: r.color || "var(--ink-3)", "stroke-dasharray": "5 4", "stroke-width": 1.4 }, s));
    o.series.forEach((sr, si) => {
      const p = el("path", { d: line(sr.values.map((v, i) => [sc.X(xs[i]), sc.Y(v)])), fill: "none", stroke: sr.color, "stroke-width": sr.width || 2, "stroke-dasharray": sr.dash || "", "stroke-linejoin": "round" }, s);
      if (o.animate !== false && !sr.dash) setLen(p, si * 0.15, 1.2);
      if (sr.dots) sr.values.forEach((v, i) => el("circle", { cx: sc.X(xs[i]), cy: sc.Y(v), r: 3, fill: sr.color }, s));
    });
    if (o.legend) {
      const lg = el("g", { transform: `translate(${L + 6},${T + 4})` }, s);
      let x = 0;
      o.series.forEach(sr => { if (!sr.label) return; el("rect", { x, y: 0, width: 14, height: 3, rx: 1.5, fill: sr.color }, lg); const t = el("text", { x: x + 18, y: 5 }, lg); t.textContent = sr.label; x += 26 + sr.label.length * 5.6; });
    }
    return { s, sc };
  }

  function bars(container, values, o) {
    const W = o.W || 640, H = o.H || 240, L = o.L || 40, R = 10, T = 12, B = 30;
    const s = svg(container, W, H);
    const max = Math.max(...values, o.ref || 0) * 1.12;
    const sc = frame(s, { W, H, L, R, T, B, x: [0, values.length], y: [0, max], xticks: [], yfmt: o.yfmt, ny: 4 });
    const bw = (W - L - R) / values.length;
    values.forEach((v, i) => {
      const r = el("rect", { x: L + i * bw + bw * .12, y: sc.Y(v), width: bw * .76, height: sc.Y(0) - sc.Y(v), rx: 4, fill: o.color || "var(--synth)", class: "fadein", style: `--delay:${i * 0.05}s;--o:.9` }, s);
      el("text", { x: L + (i + .5) * bw, y: H - B + 15, "text-anchor": "middle" }, s).textContent = o.labels ? o.labels[i] : i + 1;
      bindTip(r, o.tip ? o.tip(v, i) : String(v));
    });
    if (o.ref != null) el("line", { x1: L, x2: W - R, y1: sc.Y(o.ref), y2: sc.Y(o.ref), stroke: "var(--ink)", "stroke-dasharray": "5 4", "stroke-width": 1.4 }, s);
    return s;
  }

  function scatter(container, groups, o) {
    const W = o.W || 640, H = o.H || 320, L = 40, R = 12, T = 12, B = 30;
    const s = svg(container, W, H);
    const all = groups.flatMap(g => g.pts);
    const [x0, x1] = extent([all.map(p => p[0])]), [y0, y1] = extent([all.map(p => p[1])]);
    const px = (x1 - x0) * .05, py = (y1 - y0) * .05;
    const sc = frame(s, { W, H, L, R, T, B, x: [x0 - px, x1 + px], y: [y0 - py, y1 + py], xfmt: v => v.toFixed(0), yfmt: v => v.toFixed(0), xlabel: o.xlabel });
    groups.forEach((g, gi) => g.pts.forEach((p, i) => el("circle", { cx: sc.X(p[0]), cy: sc.Y(p[1]), r: 3, fill: g.color, class: "fadein", style: `--o:.55;--delay:${(gi * 0.3 + i * 0.002).toFixed(3)}s` }, s)));
    const lg = el("g", { transform: `translate(${L + 8},${T + 6})` }, s);
    groups.forEach((g, i) => { el("circle", { cx: i * 110 + 4, cy: 0, r: 4, fill: g.color }, lg); el("text", { x: i * 110 + 12, y: 4 }, lg).textContent = g.label; });
    return s;
  }

  window.Charts = { el, svg, pct, signed, niceTicks, extent, line, setLen, frame, tip, bindTip, lines, bars, scatter };
})();
