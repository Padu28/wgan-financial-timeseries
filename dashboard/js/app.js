/* Tempora dashboard. Plain script; open index.html directly or serve the folder. */
(function () {
  "use strict";
  const C = window.Charts, D = window.TEMPORA_DATA;
  const $ = (s, r) => (r || document).querySelector(s);
  const $$ = (s, r) => Array.from((r || document).querySelectorAll(s));
  const esc = s => String(s).replace(/[&<>"']/g, c => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
  const reduce = matchMedia("(prefers-reduced-motion: reduce)").matches;
  if (!D) { document.querySelector("main").innerHTML = `<section class="sec"><h2>No results yet</h2><p>Run <code>tempora train</code> to create <code>dashboard/data/tempora_data.js</code>, then reload.</p></section>`; return; }

  const store = {
    get(k, d) { try { const v = localStorage.getItem("tempora:" + k); return v == null ? d : JSON.parse(v); } catch (e) { return d; } },
    set(k, v) { try { localStorage.setItem("tempora:" + k, JSON.stringify(v)); } catch (e) { /* unavailable */ } }
  };
  const fmtDate = s => new Date(s + "T00:00:00Z").toLocaleDateString("en-GB", { day: "numeric", month: "short", year: "numeric", timeZone: "UTC" });
  const meta = D.meta, gcfg = meta.config.gan;

  /* ---------------- header & scrollspy ---------------- */
  $("#dataset").textContent = `${meta.name === "benchmark" ? "Benchmark market (GJR-GARCH-t)" : meta.name.toUpperCase()}, ${meta.start.slice(0, 4)}–${meta.end.slice(0, 4)}`;
  $("#footMeta").textContent = `Tempora v${meta.version}. Trained on ${meta.train_windows.toLocaleString()} windows up to ${fmtDate(meta.train_end)}; tested on ${meta.test_windows} later windows. Generated ${meta.generated.slice(0, 10)}.`;
  const links = $$(".sections a");
  const spy = new IntersectionObserver(es => es.forEach(e => {
    if (e.isIntersecting) links.forEach(a => a.classList.toggle("on", a.getAttribute("href") === "#" + e.target.id));
  }), { rootMargin: "-45% 0px -50% 0px" });
  $$("main .sec").forEach(s => spy.observe(s));
  const theme = store.get("theme", null);
  if (theme) document.documentElement.dataset.theme = theme;
  else if (matchMedia("(prefers-color-scheme: dark)").matches) document.documentElement.dataset.theme = "dark";
  $("#themeBtn").addEventListener("click", toggleTheme);
  function toggleTheme() { const h = document.documentElement; h.dataset.theme = h.dataset.theme === "dark" ? "light" : "dark"; store.set("theme", h.dataset.theme); }

  /* ---------------- 1. generate ---------------- */
  const CTX = D.contexts;
  let ci = Math.min(CTX.length - 1, Math.max(0, +(new URLSearchParams(location.hash.slice(1)).get("m") || Math.floor(CTX.length * 0.55))));
  let revealed = false;
  $("#ctxSelect").innerHTML = CTX.map((c, i) => `<option value="${i}">${fmtDate(c.start)}</option>`).join("");
  $("#ctxSelect").addEventListener("change", e => { ci = +e.target.value; revealed = false; renderFan(true); });
  $("#prevCtx").addEventListener("click", () => moveCtx(-1));
  $("#nextCtx").addEventListener("click", () => moveCtx(1));
  $("#genBtn").addEventListener("click", () => renderFan(true));
  $("#revealBtn").addEventListener("click", () => { revealed = !revealed; renderFan(false); });
  function moveCtx(d) { ci = (ci + d + CTX.length) % CTX.length; revealed = false; renderFan(true); }

  function renderFan(animate) {
    const c = CTX[ci], el = C.el;
    $("#ctxSelect").value = ci;
    $("#revealBtn").setAttribute("aria-pressed", String(revealed));
    $("#revealBtn").textContent = revealed ? "Hide what happened" : "Reveal what happened";
    history.replaceState(null, "", "#m=" + ci);
    const H = c.history.length, k = c.cond.length, n = c.real.length;
    const base = c.cond[k - 1];
    const xs = { hist: c.history.map((_, i) => i - (H - 1) - k), cond: c.cond.map((_, i) => i - k + 1), fut: c.real.map((_, i) => i + 1) };
    const allY = [c.history, c.cond, c.q["0.05"], c.q["0.95"], c.real];
    let [y0, y1] = C.extent(allY); const pad = (y1 - y0) * 0.08; y0 -= pad; y1 += pad;
    const narrow = $("#fan").clientWidth && $("#fan").clientWidth < 640;
    const W = narrow ? 640 : 980, Hh = narrow ? 620 : 440, L = 58, R = 16, T = 16, B = 36;
    const s = C.svg($("#fan"), W, Hh);
    const x0 = xs.hist[0], x1 = n;
    const sc = C.frame(s, { W, H: Hh, L, R, T, B, x: [x0, x1], y: [y0, y1], yfmt: v => v.toFixed(v < 10 ? 1 : 0), xfmt: v => v === 0 ? "today" : (v > 0 ? "+" : "") + v, ny: 5, nx: 8, xlabel: "Trading days relative to the last known day" });
    const P = (x, y) => [sc.X(x), sc.Y(y)];
    const clip = el("clipPath", { id: "fanClip" }, el("defs", {}, s));
    el("rect", { x: L, y: T, width: W - L - R, height: Hh - T - B }, clip);
    el("line", { x1: sc.X(0), x2: sc.X(0), y1: T, y2: Hh - B, class: "now" }, s);
    // bands
    const fx = [0].concat(xs.fut);
    const band = (lo, hi, cls, delay, op) => {
      const top = [base].concat(c.q[hi]).map((v, i) => P(fx[i], v)), bot = [base].concat(c.q[lo]).map((v, i) => P(fx[i], v)).reverse();
      el("path", { d: C.line(top) + "L" + C.line(bot).slice(1) + "Z", class: cls + (animate ? " fadein" : ""), style: animate ? `--delay:${delay}s;--o:${op}` : "" }, s);
    };
    band("0.05", "0.95", "band90", 1.0, 0.12);
    band("0.25", "0.75", "band50", 1.2, 0.22);
    // paths
    const pathG = el("g", { "clip-path": "url(#fanClip)" }, s);
    c.paths.forEach((p, i) => {
      const e = el("path", { d: C.line([P(0, base)].concat(p.map((v, j) => P(xs.fut[j], v)))), class: "path" }, pathG);
      if (animate && !reduce) C.setLen(e, i * 0.018, 0.7);
    });
    const med = el("path", { d: C.line([P(0, base)].concat(c.q["0.5"].map((v, j) => P(xs.fut[j], v)))), class: "median" + (animate ? " fadein" : ""), style: animate ? "--delay:1.3s" : "" }, s);
    void med;
    const hist = el("path", { d: C.line(c.history.map((v, i) => P(xs.hist[i], v))), class: "hist" }, s);
    if (animate && !reduce) C.setLen(hist, 0, 0.5);
    el("path", { d: C.line([P(xs.hist[H - 1], c.history[H - 1])].concat(c.cond.map((v, i) => P(xs.cond[i], v)))), class: "cond" }, s);
    if (revealed) {
      const rf = el("path", { d: C.line([P(0, base)].concat(c.real.map((v, j) => P(xs.fut[j], v)))), class: "realf" }, s);
      C.setLen(rf, 0, 0.9);
      el("circle", { cx: sc.X(n), cy: sc.Y(c.real[n - 1]), r: 5, fill: "var(--real)" }, s);
    }
    el("text", { x: sc.X(-k + 1) - 4, y: T + 12, "text-anchor": "end", class: "strong" }, s).textContent = "What the generator sees";
    el("text", { x: sc.X(2), y: T + 12, class: "strong" }, s).textContent = `${c.paths.length} of ${gcfg && meta.config.n_scenarios} simulated futures`;
    // hover readout
    const hover = el("line", { y1: T, y2: Hh - B, stroke: "var(--ink-3)", "stroke-dasharray": "2 3", x1: -9, x2: -9 }, s);
    s.addEventListener("mousemove", e => {
      const r = s.getBoundingClientRect(), vx = (e.clientX - r.left) / r.width * W;
      const d = Math.round(x0 + (vx - L) / (W - L - R) * (x1 - x0));
      if (d < 1 || d > n) { hover.setAttribute("x1", -9); hover.setAttribute("x2", -9); C.tip(null); return; }
      hover.setAttribute("x1", sc.X(d)); hover.setAttribute("x2", sc.X(d));
      const q = key => c.q[key][d - 1];
      C.tip(`<b>Day +${d}</b><br>Median ${q("0.5").toFixed(2)} (${C.signed(q("0.5") / base - 1)})<br>90% band ${q("0.05").toFixed(2)} to ${q("0.95").toFixed(2)}${revealed ? `<br>Actual ${c.real[d - 1].toFixed(2)}` : ""}`, e);
    });
    s.addEventListener("mouseleave", () => { C.tip(null); hover.setAttribute("x1", -9); hover.setAttribute("x2", -9); });
    $("#fanLegend").innerHTML = `<span><i style="--c:var(--real)"></i>Market price</span><span><i style="--c:var(--synth)"></i>Simulated paths</span><span><i class="band" style="--c:var(--synth)"></i>50% and 90% bands</span><span><i style="--c:var(--synth);height:2px"></i>Median (dashed)</span>`;
    renderReadout(c, base);
  }

  function renderReadout(c, base) {
    const last = c.paths.map(p => p[p.length - 1]);
    const pLoss = last.filter(v => v < base).length / last.length;
    const med = c.q["0.5"][c.q["0.5"].length - 1] / base - 1;
    const var5 = 1 - c.q["0.05"][c.q["0.05"].length - 1] / base;
    const realRet = c.real[c.real.length - 1] / base - 1;
    const rank = last.filter(v => v < c.real[c.real.length - 1]).length / last.length;
    $("#readout").innerHTML = `
      <div class="stat synth"><span class="l">Median 20-day return</span><span class="v">${C.signed(med)}</span><span class="n">Starting price ${base.toFixed(2)} on ${fmtDate(c.start)}</span></div>
      <div class="stat synth"><span class="l">Chance of a loss</span><span class="v">${C.pct(pLoss, 0)}</span><span class="n">Share of simulated paths ending lower</span></div>
      <div class="stat synth"><span class="l">5% worst case (VaR)</span><span class="v">−${C.pct(var5)}</span><span class="n">One in twenty futures loses more than this</span></div>
      <div class="stat ${revealed ? "" : "hide"}"><span class="l">What actually happened</span><span class="v">${C.signed(realRet)}</span>
        <div class="pctbar" aria-hidden="true"><i style="left:${revealed ? (rank * 100).toFixed(1) : 50}%"></i></div>
        <span class="n">${revealed ? `Beat ${C.pct(rank, 0)} of simulated futures` : "Press Reveal to compare"}</span></div>`;
  }

  /* ---------------- 2. realism ---------------- */
  function renderRealism() {
    const sc = D.scorecard;
    $("#scorecard").innerHTML = sc.map((c, i) => `<div class="check" style="--c:var(--${c.pass ? "pass" : "fail"});--cs:var(--${c.pass ? "pass" : "fail"}-soft);transition-delay:${i * 80}ms">
      <div class="top"><h3>${esc(c.name)}</h3><span class="badge">${c.pass ? "Reproduced" : "Missed"}</span></div><p>${esc(c.test)}</p>
      <div class="vals"><span>Real<b>${esc(c.real)}</b></span><span class="s">Synthetic<b>${esc(c.synthetic)}</b></span></div></div>`).join("");
    const obs = new IntersectionObserver(es => es.forEach(e => { if (e.isIntersecting) { $("#scorecard").classList.add("in"); obs.disconnect(); } }), { threshold: 0.2 });
    obs.observe($("#scorecard"));

    // histogram (log density)
    const h = D.histogram, mids = h.edges.slice(1).map((e, i) => (e + h.edges[i]) / 2 * 100);
    const floor = 0.01;
    const W = 640, H = 270, L = 46, R = 12, T = 12, B = 34;
    const s = C.svg($("#hist"), W, H);
    const ymax = Math.max(...h.real, ...h.fake) * 1.5;
    const fr = C.frame(s, { W, H, L, R, T, B, x: [mids[0], mids[mids.length - 1]], y: [floor, ymax], ylog: true, yfloor: floor, xfmt: v => v.toFixed(0) + "%", xlabel: "Daily return" });
    [1, 10, 100].filter(v => v < ymax && v > floor).forEach(v => { C.el("line", { x1: L, x2: W - R, y1: fr.Y(v), y2: fr.Y(v), stroke: "var(--line)" }, s); C.el("text", { x: L - 6, y: fr.Y(v) + 4, "text-anchor": "end" }, s).textContent = v; });
    const step = vals => {   // break the line where a bin is empty instead of plunging to the floor
      let d = "", pen = false;
      vals.forEach((v, i) => { if (v <= 0) { pen = false; return; } d += (pen ? "L" : "M") + fr.X(mids[i]).toFixed(1) + "," + fr.Y(v).toFixed(1); pen = true; });
      return d;
    };
    const pf = C.el("path", { d: step(h.fake), fill: "none", stroke: "var(--synth)", "stroke-width": 2.4 }, s);
    const pr = C.el("path", { d: step(h.real), fill: "none", stroke: "var(--real)", "stroke-width": 2 }, s);
    C.setLen(pr, 0, 1.2); C.setLen(pf, 0.2, 1.2);

    // QQ
    const qq = D.qq.map(p => [p[0] * 100, p[1] * 100]);
    const [lo, hi] = C.extent([qq.map(p => p[0]), qq.map(p => p[1])]);
    const s2 = C.svg($("#qq"), 640, 270);
    const f2 = C.frame(s2, { W: 640, H: 270, L: 46, R: 12, T: 12, B: 34, x: [lo, hi], y: [lo, hi], xfmt: v => v.toFixed(1) + "%", yfmt: v => v.toFixed(1) + "%", xlabel: "Real quantiles (synthetic on the vertical axis)" });
    C.el("line", { x1: f2.X(lo), y1: f2.Y(lo), x2: f2.X(hi), y2: f2.Y(hi), stroke: "var(--ink-3)", "stroke-dasharray": "5 4" }, s2);
    qq.forEach((p, i) => { const c = C.el("circle", { cx: f2.X(p[0]), cy: f2.Y(p[1]), r: 3.4, fill: "var(--synth)", class: "fadein", style: `--o:.85;--delay:${i * 0.008}s` }, s2); C.bindTip(c, `<b>Quantile ${(0.5 + i).toFixed(0)}%</b><br>Real ${p[0].toFixed(2)}%<br>Synthetic ${p[1].toFixed(2)}%`); });

    const f = D.facts, lags = f.lags;
    const fact = (id, key, refs) => C.lines($("#" + id), { x: lags, W: 420, H: 220, series: [{ values: f.real[key], color: "var(--real)", label: "Real", dots: true }, { values: f.synthetic[key], color: "var(--synth)", label: "Synthetic", dots: true, width: 2.4 }], refs: [{ v: 0 }].concat(refs || []), yfmt: v => v.toFixed(2), xticks: lags.filter(l => l % 2 === 1), legend: true, xlabel: "Lag (days)" });
    fact("acfR", "acf_returns"); fact("acfA", "acf_abs"); fact("lev", "leverage");

    const lp = D.long_paths, n = lp.real.length;
    const series = [{ values: lp.real, color: "var(--real)", label: "Real", width: 2.4 }].concat(lp.synthetic.map((p, i) => ({ values: p.slice(0, n), color: ["var(--synth)", "var(--synth-2)", "#C9432A"][i], label: `Synthetic ${i + 1}`, width: 1.8 })));
    C.lines($("#long"), { W: 1240, H: 300, series, legend: true, yfmt: v => v.toFixed(0), xlabel: "Trading days", xfmt: v => v.toFixed(0) });
  }

  /* ---------------- 3. turing test ---------------- */
  const TT = D.turing;
  let ttRound = 0, ttScore = [], ttOrder = [], ttLock = false, ttRealSide = "A";
  function ttNewGame() { ttRound = 0; ttScore = []; ttOrder = TT.map((_, i) => i).sort(() => Math.random() - 0.5).slice(0, 10); ttShow(); }
  function ttDraw(container, vals) {
    const s = C.svg(container, 460, 220);
    const [lo, hi] = C.extent([vals]); const pad = (hi - lo) * .1 || 1;
    const X = i => 10 + i / (vals.length - 1) * 440, Y = v => 10 + (1 - (v - lo + pad) / (hi - lo + 2 * pad)) * 200;
    const p = C.el("path", { d: C.line(vals.map((v, i) => [X(i), Y(v)])), class: "ln" }, s);
    C.setLen(p, 0, 0.8);
  }
  function ttShow() {
    ttLock = false;
    const pair = TT[ttOrder[ttRound]];
    ttRealSide = Math.random() < 0.5 ? "A" : "B";
    ttDraw($("#ttChartA"), ttRealSide === "A" ? pair.real : pair.synthetic);
    ttDraw($("#ttChartB"), ttRealSide === "B" ? pair.real : pair.synthetic);
    $$(".tt-opt").forEach(b => b.className = "tt-opt");
    $("#ttMsg").innerHTML = `Round ${ttRound + 1} of 10. Which path is the real market? (keys <kbd>A</kbd> / <kbd>B</kbd>)`;
    $("#ttNext").textContent = "Skip pair";
    ttPaintScore();
  }
  function ttPaintScore() { $("#ttScore").innerHTML = [...Array(10)].map((_, i) => `<i class="${ttScore[i] === true ? "r" : ttScore[i] === false ? "w" : ""}"></i>`).join(""); }
  function ttPick(side) {
    if (ttLock) return; ttLock = true;
    const ok = side === ttRealSide;
    ttScore.push(ok); ttPaintScore();
    const chosen = side === "A" ? $("#ttA") : $("#ttB"), other = side === "A" ? $("#ttB") : $("#ttA");
    chosen.classList.add(ok ? "right" : "wrong");
    (ttRealSide === "A" ? $("#ttA") : $("#ttB")).classList.add("reveal-real");
    (ttRealSide === "A" ? $("#ttB") : $("#ttA")).classList.add("reveal-synth");
    void other;
    const right = ttScore.filter(Boolean).length;
    if (ttScore.length >= 10) {
      const verdict = right <= 6 ? "That's close to coin-flipping, which is what a good generator should force." : "You spotted more than chance would. Short paths hide differences better than long ones.";
      $("#ttMsg").textContent = `You got ${right} of 10 right. ${verdict}`;
      $("#ttNext").textContent = "Play again";
      store.set("best", Math.max(store.get("best", 0), right));
    } else {
      $("#ttMsg").textContent = ok ? `Correct. ${right} of ${ttScore.length} so far.` : `Not this time: that one was synthetic. ${right} of ${ttScore.length} so far.`;
      $("#ttNext").textContent = "Next pair";
    }
  }
  $("#ttA").addEventListener("click", () => ttPick("A"));
  $("#ttB").addEventListener("click", () => ttPick("B"));
  $("#ttNext").addEventListener("click", () => { if (ttScore.length >= 10) return ttNewGame(); if (!ttLock) ttScore.push(null); ttRound = Math.min(9, ttScore.length); if (ttScore.length >= 10) { ttPaintScore(); $("#ttMsg").textContent = "Game over. Press Play again."; $("#ttNext").textContent = "Play again"; return; } ttShow(); });

  /* ---------------- 4. risk ---------------- */
  function renderRisk() {
    const cov = D.coverage, vb = D.var_backtest, S = D.summary;
    const kp = [
      ["90% band hit rate", C.pct(cov.coverage["0.9"], 0), "Target 90%", Math.abs(cov.coverage["0.9"] - 0.9) < 0.05],
      ["50% band hit rate", C.pct(cov.coverage["0.5"], 0), "Target 50%", Math.abs(cov.coverage["0.5"] - 0.5) < 0.07],
      ["95% VaR breaches", C.pct(vb.rate, 1), `${vb.breaches} of ${vb.n} days; target 5%`, Math.abs(vb.rate - 0.05) < 0.03],
      ["CRPS, 20-day return", (cov.crps * 100).toFixed(2) + "%", "Forecast error in return units; lower is better", null],
    ];
    $("#riskKpis").innerHTML = kp.map(k => `<div class="stat"><span class="l">${k[0]}</span><span class="v" style="color:${k[3] == null ? "var(--ink)" : k[3] ? "var(--pass)" : "var(--fail)"}">${k[1]}</span><span class="n">${k[2]}</span></div>`).join("");
    const hz = [...Array(cov.horizon)].map((_, i) => i + 1), cols = ["var(--synth-2)", "var(--synth)", "#C9432A"];
    C.lines($("#coverage"), { x: hz, series: cov.levels.map((lv, i) => ({ values: cov.coverage_by_horizon[String(lv)], color: cols[i], label: `${lv * 100}% band`, width: 2.4 })), refs: cov.levels.map((lv, i) => ({ v: lv, color: cols[i] })), y: [0.3, 1], yfmt: v => C.pct(v, 0), legend: true, xlabel: "Days ahead" });
    const expected = cov.pit_hist.reduce((a, b) => a + b, 0) / cov.pit_hist.length;
    C.bars($("#pit"), cov.pit_hist, { labels: cov.pit_hist.map((_, i) => `${i * 10}–${i * 10 + 10}`), ref: expected, tip: (v, i) => `<b>${i * 10}–${i * 10 + 10}% rank</b><br>${v} test windows`, color: "var(--synth)" });
    // VaR chart
    const n = vb.realised.length, W = 1240, H = 280, L = 56, R = 12, T = 12, B = 32;
    const s = C.svg($("#varChart"), W, H);
    const ys = vb.realised.concat(vb.var.map(v => -v));
    let [lo, hi] = C.extent([ys]); const pad = (hi - lo) * .08; lo -= pad; hi += pad;
    const ticks = []; for (let i = 0; i < n; i += Math.ceil(n / 8)) ticks.push(i);
    const f = C.frame(s, { W, H, L, R, T, B, x: [0, n - 1], y: [lo, hi], yfmt: v => C.pct(v, 0), xticks: ticks, xfmt: i => vb.dates[i] ? vb.dates[i].slice(0, 7) : "" });
    const area = vb.var.map((v, i) => [f.X(i), f.Y(-v)]);
    C.el("path", { d: C.line(area) + `L${f.X(n - 1)},${f.Y(lo)}L${f.X(0)},${f.Y(lo)}Z`, fill: "var(--synth)", opacity: .14 }, s);
    const vl = C.el("path", { d: C.line(area), fill: "none", stroke: "var(--synth)", "stroke-width": 1.8 }, s); C.setLen(vl, 0, 1.4);
    C.el("line", { x1: L, x2: W - R, y1: f.Y(0), y2: f.Y(0), stroke: "var(--line)" }, s);
    vb.realised.forEach((v, i) => {
      const br = vb.series[i] === 1;
      const c = C.el("circle", { cx: f.X(i), cy: f.Y(v), r: br ? 5 : 2.6, fill: br ? "var(--fail)" : "var(--real)", class: "fadein", style: `--o:${br ? 1 : .55};--delay:${(i / n).toFixed(3)}s` }, s);
      C.bindTip(c, `<b>${fmtDate(vb.dates[i])}</b><br>Realised ${C.signed(v, 2)}<br>95% VaR −${C.pct(vb.var[i], 2)}${br ? "<br><b>Breach</b>" : ""}`);
    });
  }

  /* ---------------- 5. training ---------------- */
  const SN = D.snapshots;
  let snapIdx = SN.length - 1, playing = null;
  function renderTraining() {
    const t = D.training;
    C.lines($("#wChart"), { x: t.step, W: 420, H: 220, series: [{ values: t.wasserstein, color: "var(--synth)" }], yfmt: v => v.toFixed(2), xfmt: v => (v / 1000).toFixed(1) + "k", xlabel: "Iteration" });
    C.lines($("#gnChart"), { x: t.step, W: 420, H: 220, series: [{ values: t.grad_norm, color: "var(--cobalt)" }], refs: [{ v: 1, color: "var(--ink)" }], yfmt: v => v.toFixed(2), xfmt: v => (v / 1000).toFixed(1) + "k", xlabel: "Iteration" });
    C.lines($("#lrChart"), { x: t.step, W: 420, H: 220, series: [{ values: t.lr_d.map(v => v * 1e4), color: "var(--real)", label: "Critic" }, { values: t.lr_g.map(v => v * 1e4), color: "var(--synth)", label: "Generator" }], yfmt: v => v.toFixed(0) + "e-4", xfmt: v => (v / 1000).toFixed(1) + "k", legend: true, animate: false, xlabel: "Iteration" });
    const sl = $("#snapSlider"); sl.max = SN.length - 1; sl.value = snapIdx;
    sl.addEventListener("input", () => { snapIdx = +sl.value; drawSnap(); });
    $("#playBtn").addEventListener("click", togglePlay);
    drawSnap();
  }
  function drawSnap() {
    const sn = SN[snapIdx], last = SN[SN.length - 1], real = D.facts.real;
    $("#snapSlider").value = snapIdx;
    $("#snapOut").textContent = `Iteration ${sn.step.toLocaleString()}`;
    const W = 760, H = 300, L = 50, R = 12, T = 12, B = 30;
    const allP = SN.flatMap(x => x.paths.flat()).map(v => v * 100);
    let lo = Math.max(-30, Math.min(...allP)), hi = Math.min(30, Math.max(...allP));
    const s = C.svg($("#replayChart"), W, H);
    const n = sn.paths[0].length;
    const f = C.frame(s, { W, H, L, R, T, B, x: [0, n - 1], y: [lo, hi], yfmt: v => v.toFixed(0) + "%", xlabel: "Day in window (cumulative return)" });
    const shades = ["var(--synth)", "var(--synth-2)", "#C9432A", "var(--real)"];
    sn.paths.forEach((p, i) => C.el("path", { d: C.line(p.map((v, j) => [f.X(j), f.Y(Math.max(lo, Math.min(hi, v * 100)))])), fill: "none", stroke: shades[i], "stroke-width": 2.2, "stroke-linejoin": "round" }, s));
    C.el("line", { x1: f.X(gcfg.conditional - 1), x2: f.X(gcfg.conditional - 1), y1: T, y2: H - B, stroke: "var(--ink-3)", "stroke-dasharray": "3 4" }, s);
    C.el("text", { x: f.X(gcfg.conditional - 1) + 6, y: T + 12 }, s).textContent = "generated from here";
    const row = (l, v, target, fmt) => `<div class="stat"><span class="l">${l}</span><span class="v" style="font-size:1.4rem">${fmt(v)}</span><span class="n">Real data: ${fmt(target)}</span></div>`;
    $("#replayStats").innerHTML = row("Excess kurtosis", sn.kurtosis, real.kurtosis, v => v.toFixed(1)) + row("Volatility clustering (lag 1)", sn.acf_abs1, real.acf_abs[0], v => v.toFixed(2)) + row("Daily volatility", sn.std, real.std, v => (v * 100).toFixed(2) + "%");
    void last;
  }
  function togglePlay() {
    const b = $("#playBtn");
    if (playing) { clearInterval(playing); playing = null; b.textContent = "Play"; b.setAttribute("aria-pressed", "false"); return; }
    if (snapIdx >= SN.length - 1) snapIdx = 0;
    b.textContent = "Pause"; b.setAttribute("aria-pressed", "true");
    playing = setInterval(() => { snapIdx++; drawSnap(); if (snapIdx >= SN.length - 1) togglePlay(); }, reduce ? 50 : 380);
  }

  /* ---------------- 6. model ---------------- */
  function renderModel() {
    const g = gcfg, k = g.conditional;
    $("#arch").innerHTML = [
      ["Prices", `${meta.n_prices.toLocaleString()} daily closes`, ""],
      ["Transform", `Log returns, standardised, inverse Lambert W (δ = ${D.transform.delta})`, ""],
      ["Generator", `${g.latent_dim}-d noise with the first ${k} values set to real returns; 2 × ${g.hidden} LeakyReLU, residual skip, shape and volatility heads`, "g"],
      ["Window", `${k} real + ${g.ts_dim - k} generated days`, "g"],
      ["Critic", `2 × ${g.hidden} LeakyReLU MLP, Wasserstein loss, gradient penalty λ = ${g.gp_weight}`, "d"],
    ].map(n => `<div class="node ${n[2]}"><b>${n[0]}</b><span>${esc(n[1])}</span></div>`).join("");
    const d = D.discriminative;
    $("#discNote").textContent = `A gradient-boosting classifier separates real from synthetic windows with ${C.pct(d.accuracy, 0)} accuracy (50% would mean indistinguishable). Below: PCA of real and synthetic windows, explaining ${C.pct(D.pca.explained[0] + D.pca.explained[1], 0)} of the variance.`;
    C.scatter($("#pca"), [{ label: "Real", color: "var(--real)", pts: D.pca.real }, { label: "Synthetic", color: "var(--synth)", pts: D.pca.fake }], { xlabel: "First principal component" });
    const spec = [
      ["Data", `${meta.name === "benchmark" ? "Simulated GJR-GARCH(1,1)-t market" : meta.name}, ${fmtDate(meta.start)} to ${fmtDate(meta.end)}`],
      ["Train / test split", `${C.pct(meta.config.train_frac, 0)} / ${C.pct(1 - meta.config.train_frac, 0)} by date (test from ${fmtDate(meta.train_end)})`],
      ["Windows", `${g.ts_dim} days, stride ${meta.config.train_stride}; ${meta.train_windows.toLocaleString()} for training`],
      ["Training", `${g.iterations.toLocaleString()} iterations × ${g.critic_iters} critic steps, batch ${g.batch_size}`],
      ["Optimiser", `Adam (β = 0, 0.9), triangular cyclical LR ${g.lr_d[0]}–${g.lr_d[1]} (critic), ${g.lr_g[0]}–${g.lr_g[1]} (generator)`],
      ["Distance", `Wasserstein-1 between return distributions: ${D.distances.wasserstein_rel.toFixed(3)} × daily volatility; KS ${D.distances.ks.toFixed(3)}`],
      ["Scenarios", `${meta.config.n_scenarios} per test window`],
      ["Runtime", `${D.summary.train_seconds}s on one CPU core (NumPy engine)`],
    ];
    $("#spec").innerHTML = spec.map(s => `<dt>${s[0]}</dt><dd>${esc(s[1])}</dd>`).join("");
  }

  /* ---------------- keyboard ---------------- */
  document.addEventListener("keydown", e => {
    if (/INPUT|SELECT|TEXTAREA/.test(document.activeElement.tagName) || e.ctrlKey || e.metaKey) return;
    const k = e.key.toLowerCase();
    const inTT = $("#turing").getBoundingClientRect().top < innerHeight * 0.6 && $("#turing").getBoundingClientRect().bottom > innerHeight * 0.3;
    if (inTT && (k === "a" || k === "b")) ttPick(k.toUpperCase());
    else if (k === "g") renderFan(true);
    else if (k === "r") { revealed = !revealed; renderFan(false); }
    else if (e.key === "ArrowRight" && !inTT) moveCtx(1);
    else if (e.key === "ArrowLeft" && !inTT) moveCtx(-1);
    else if (k === "t") toggleTheme();
  });

  renderFan(true);
  renderRealism();
  ttNewGame();
  renderRisk();
  renderTraining();
  renderModel();
})();
