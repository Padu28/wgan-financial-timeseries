"""End-to-end experiment: prices -> transforms -> WGAN-GP -> evaluation -> dashboard + reports."""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from . import __version__
from . import metrics as M
from .gan import WGANGP, GANConfig
from .transforms import ReturnTransform, log_returns, windows


@dataclass
class ExperimentConfig:
    name: str = "benchmark"
    train_frac: float = 0.8
    train_stride: int = 1
    test_stride: int = 5
    n_scenarios: int = 200
    n_dashboard_paths: int = 60
    n_contexts: int = 36
    long_path_windows: int = 30
    gan: GANConfig = field(default_factory=GANConfig)


@dataclass
class Experiment:
    prices: pd.Series
    config: ExperimentConfig
    transform: ReturnTransform
    model: WGANGP
    results: dict
    seconds: float


def _chain(model: WGANGP, tr: ReturnTransform, start_cond: np.ndarray, n_windows: int) -> np.ndarray:
    """Generate one long synthetic return path by feeding each window's last steps back as the condition."""
    k = model.cfg.conditional
    cond = start_cond.copy()
    out = []
    for _ in range(n_windows):
        w = model.generate(cond[None, :], 1)[0]
        out.append(w[k:])
        cond = w[-k:]
    return tr.inverse(np.concatenate(out))


def run(prices: pd.Series, config: ExperimentConfig | None = None, verbose: bool = True) -> Experiment:
    cfg = config or ExperimentConfig()
    g = cfg.gan
    t0 = time.perf_counter()
    dates = prices.index[1:]
    r = log_returns(prices.to_numpy())
    n_train = int(len(r) * cfg.train_frac)
    tr = ReturnTransform().fit(r[:n_train])
    z = tr.transform(r)
    train_w = windows(z[:n_train], g.ts_dim, cfg.train_stride)
    test_idx = np.arange(n_train, len(r) - g.ts_dim + 1, cfg.test_stride)
    g.output_bound = float(1.05 * np.abs(train_w).max())

    model = WGANGP(g)
    rng = np.random.default_rng(g.seed)
    replay_ctx = train_w[rng.choice(len(train_w), 4, replace=False)]
    replay_noise_rng = np.random.default_rng(g.seed + 1)
    replay_noise = replay_noise_rng.standard_normal((4, g.latent_dim))
    snapshots = []

    def snapshot(m: WGANGP, step: int) -> None:
        if step % g.snapshot_every and step != 1:
            return
        zz = replay_noise.copy()
        zz[:, :g.conditional] = replay_ctx[:, :g.conditional]
        y, _ = m.G.forward(zz)
        fake = np.concatenate([replay_ctx[:, :g.conditional], y], 1)
        sample = m.generate(train_w[: min(400, len(train_w)), :g.conditional], 1)
        fr = tr.inverse(sample[:, g.conditional:])
        snapshots.append({
            "step": step,
            "paths": np.round(np.cumsum(tr.inverse(fake), 1), 5).tolist(),
            "kurtosis": round(float(pd.Series(fr.ravel()).kurt()), 2),
            "acf_abs1": round(M.acf_curve(np.abs(fr), [1])[0], 3),
            "std": round(float(fr.std()), 5),
        })
        if verbose and step % (g.snapshot_every * 4) == 0:
            h = m.history
            print(f"  step {step:>5}  W≈{np.mean(h.wasserstein[-50:]):.3f}  grad-norm {np.mean(h.grad_norm[-50:]):.2f}")

    if verbose:
        print(f"Training WGAN-GP on {len(train_w):,} windows of {g.ts_dim} days ({g.iterations:,} iterations)")
    model.train(train_w, callback=snapshot)

    # ---------- evaluation on the training period (distribution) and test period (forecasts)
    k = g.conditional
    real_tr = tr.inverse(train_w[:, k:])
    fake_tr = tr.inverse(model.generate(train_w[:, :k], 3)[:, k:])
    facts_real, facts_fake = M.stylized_facts(real_tr), M.stylized_facts(fake_tr)

    test_windows = np.stack([z[i:i + g.ts_dim] for i in test_idx])
    test_real = tr.inverse(test_windows[:, k:])
    scen = [tr.inverse(model.generate(w[:k][None, :], cfg.n_scenarios)[:, k:]) for w in test_windows]
    cov = M.coverage_backtest(scen, test_real)
    # value-at-risk backtest on 1-day-ahead returns (first generated step)
    var_pred = np.array([-np.quantile(s[:, 0], 0.05) for s in scen])
    breaches = (test_real[:, 0] < -var_pred).astype(int)

    # ---------- dashboard payload
    px = prices.to_numpy()
    ctx_ids = np.linspace(0, len(test_idx) - 1, min(cfg.n_contexts, len(test_idx))).astype(int)
    contexts = []
    for ci in ctx_ids:
        i = test_idx[ci]
        start_price = float(px[i])                    # price before the first return of the window
        cond_prices = start_price * np.exp(np.cumsum(r[i:i + k]))
        base = float(cond_prices[-1])
        paths = scen[ci][: cfg.n_dashboard_paths]
        sim = base * np.exp(np.cumsum(paths, 1))
        real_future = base * np.exp(np.cumsum(r[i + k:i + g.ts_dim]))
        hist_start = max(0, i - 40)
        contexts.append({
            "start": str(dates[i].date()),
            "dates": [str(d.date()) for d in dates[i - 1 if i > 0 else 0:i + g.ts_dim]][-(g.ts_dim + 1):],
            "history": np.round(px[hist_start:i + 1], 3).tolist(),
            "history_start": str(prices.index[hist_start].date()),
            "cond": np.round(cond_prices, 3).tolist(),
            "real": np.round(real_future, 3).tolist(),
            "paths": np.round(sim, 3).tolist(),
            "q": {str(q): np.round(base * np.exp(np.quantile(np.cumsum(scen[ci], 1), q, axis=0)), 3).tolist()
                  for q in (0.05, 0.25, 0.5, 0.75, 0.95)},
        })

    long_n = cfg.long_path_windows
    long_paths = []
    for _ in range(3):
        lr_ = _chain(model, tr, train_w[rng.integers(len(train_w))][-k:], long_n)
        long_paths.append(np.round(100 * np.exp(np.cumsum(lr_)), 3).tolist())
    real_long_start = n_train - long_n * (g.ts_dim - k)
    real_long = 100 * np.exp(np.cumsum(r[max(real_long_start, 0):n_train]))

    # Turing test pairs: a real and a synthetic 20-day price path, shuffled
    tt = []
    for _ in range(40):
        a = real_tr[rng.integers(len(real_tr))]
        b = fake_tr[rng.integers(len(fake_tr))]
        tt.append({"real": np.round(100 * np.exp(np.cumsum(a)), 3).tolist(),
                   "synthetic": np.round(100 * np.exp(np.cumsum(b)), 3).tolist()})

    h = model.history
    every = max(1, len(h.g_loss) // 400)
    smooth = lambda x: pd.Series(x).rolling(every * 5, min_periods=1).mean().iloc[::every].round(4).tolist()  # noqa: E731
    disc = M.discriminative_score(real_tr, fake_tr, g.seed)
    results = {
        "meta": {
            "version": __version__, "name": cfg.name,
            "generated": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "start": str(prices.index[0].date()), "end": str(prices.index[-1].date()),
            "train_end": str(dates[n_train - 1].date()), "n_prices": len(prices),
            "train_windows": int(len(train_w)), "test_windows": int(len(test_idx)),
            "config": asdict(cfg) | {"gan": asdict(g)},
        },
        "prices": {"dates": [str(d.date()) for d in prices.index[::2]], "values": np.round(px[::2], 3).tolist(),
                   "train_end_index": int(n_train // 2)},
        "facts": {"real": facts_real, "synthetic": facts_fake, "lags": M.FACT_LAGS},
        "scorecard": M.realism_scorecard(facts_real, facts_fake),
        "histogram": M.histogram(real_tr, fake_tr),
        "qq": M.qq(real_tr.ravel(), fake_tr.ravel()),
        "distances": M.distances(real_tr, fake_tr),
        "discriminative": disc,
        "pca": M.pca_embedding(real_tr, fake_tr),
        "coverage": cov,
        "var_backtest": {"level": 0.95, "breaches": int(breaches.sum()), "n": int(len(breaches)),
                         "rate": float(breaches.mean()), "series": breaches.tolist(),
                         "var": np.round(var_pred, 5).tolist(), "realised": np.round(test_real[:, 0], 5).tolist(),
                         "dates": [str(dates[i + k].date()) for i in test_idx]},
        "contexts": contexts,
        "long_paths": {"synthetic": long_paths, "real": np.round(real_long, 3).tolist()},
        "turing": tt,
        "training": {"step": list(range(1, len(h.g_loss) + 1))[::every], "g_loss": smooth(h.g_loss),
                     "d_loss": smooth(h.d_loss), "wasserstein": smooth(h.wasserstein), "gp": smooth(h.gp),
                     "grad_norm": smooth(h.grad_norm), "lr_g": h.lr_g[::every], "lr_d": h.lr_d[::every]},
        "snapshots": snapshots,
        "transform": tr.to_dict(),
    }
    return Experiment(prices, cfg, tr, model, results, time.perf_counter() - t0)


def export(exp: Experiment, out_dir: str | Path = "data/processed", model_dir: str | Path = "models",
           dashboard_path: str | Path | None = "dashboard/data/tempora_data.js") -> list[Path]:
    out, mdir = Path(out_dir), Path(model_dir)
    out.mkdir(parents=True, exist_ok=True)
    mdir.mkdir(parents=True, exist_ok=True)
    res = exp.results
    written = []
    name = exp.config.name

    p = mdir / f"wgan_gp_{name}.npz"
    exp.model.save(p)
    written.append(p)
    p = mdir / f"transform_{name}.json"
    p.write_text(json.dumps(exp.transform.to_dict(), indent=2))
    written.append(p)

    f = res["facts"]
    rows = []
    for key in ("mean", "std", "skew", "kurtosis", "var95", "es95", "var99", "es99", "window_vol_dispersion"):
        rows.append({"statistic": key, "real": f["real"][key], "synthetic": f["synthetic"][key]})
    for curve in ("acf_returns", "acf_abs", "acf_sq", "leverage"):
        for lag, a, b in zip(f["lags"], f["real"][curve], f["synthetic"][curve]):
            rows.append({"statistic": f"{curve}_lag{lag}", "real": a, "synthetic": b})
    pd.DataFrame(rows).to_csv(out / "stylized_facts.csv", index=False)
    pd.DataFrame(res["scorecard"]).to_csv(out / "realism_scorecard.csv", index=False)
    tr_ = res["training"]
    pd.DataFrame({k: tr_[k] for k in ("step", "g_loss", "d_loss", "wasserstein", "gp", "grad_norm")}).to_csv(
        out / "training_history.csv", index=False)
    cov = res["coverage"]
    pd.DataFrame({"horizon": range(1, cov["horizon"] + 1)} | {f"coverage_{k}": v for k, v in
                                                              cov["coverage_by_horizon"].items()}).to_csv(
        out / "forecast_coverage.csv", index=False)
    vb = res["var_backtest"]
    pd.DataFrame({"date": vb["dates"], "var95": vb["var"], "realised": vb["realised"], "breach": vb["series"]}).to_csv(
        out / "var_backtest.csv", index=False)
    long = res["long_paths"]["synthetic"]
    pd.DataFrame({f"synthetic_path_{i + 1}": p for i, p in enumerate(long)}).to_csv(out / "synthetic_paths.csv",
                                                                                   index_label="day")
    summary = {"distances": res["distances"], "discriminative": res["discriminative"],
               "coverage": cov["coverage"], "crps": cov["crps"], "var_breach_rate": vb["rate"],
               "scorecard_passed": sum(c["pass"] for c in res["scorecard"]), "scorecard_total": len(res["scorecard"]),
               "train_seconds": round(exp.seconds, 1)}
    (out / "evaluation_summary.json").write_text(json.dumps(summary, indent=2))
    written += [out / n for n in ("stylized_facts.csv", "realism_scorecard.csv", "training_history.csv",
                                  "forecast_coverage.csv", "var_backtest.csv", "synthetic_paths.csv",
                                  "evaluation_summary.json")]
    if dashboard_path:
        dp = Path(dashboard_path)
        dp.parent.mkdir(parents=True, exist_ok=True)
        payload = json.dumps(res | {"summary": summary}, separators=(",", ":"), allow_nan=False)
        dp.write_text("window.TEMPORA_DATA=" + payload + ";\n", encoding="utf-8")
        written.append(dp)
    return written
