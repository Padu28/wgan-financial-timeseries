"""Command-line interface.

    tempora data simulate                         # benchmark market -> data/raw/benchmark_prices.csv
    tempora data download --ticker AAPL           # real prices (needs yfinance)
    tempora train                                 # train + evaluate + export dashboard data
    tempora train --data data/raw/AAPL.csv --name aapl --iterations 6000
    tempora sample --name benchmark --paths 100 --days 250
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

DEFAULT_DATA = "data/raw/benchmark_prices.csv"


def cmd_data(a: argparse.Namespace) -> int:
    from .data import download, save_prices, simulate_market

    if a.source == "simulate":
        s = simulate_market(a.days, a.seed)
        out = a.out or DEFAULT_DATA
    else:
        s = download(a.ticker, a.start)
        out = a.out or f"data/raw/{a.ticker.upper()}.csv"
    save_prices(s, out)
    print(f"Saved {len(s):,} daily prices ({s.index[0].date()} to {s.index[-1].date()}) -> {out}")
    return 0


def cmd_train(a: argparse.Namespace) -> int:
    from .data import load_prices, save_prices, simulate_market
    from .gan import GANConfig
    from .pipeline import ExperimentConfig, export, run

    path = Path(a.data)
    if not path.exists():
        if a.data == DEFAULT_DATA:
            save_prices(simulate_market(), path)
            print(f"Created benchmark prices at {path}")
        else:
            print(f"Data file not found: {path}", file=sys.stderr)
            return 1
    prices = load_prices(path)
    gan = GANConfig(iterations=a.iterations, seed=a.seed, latent_dim=a.latent_dim, hidden=a.hidden,
                    batch_size=a.batch_size)
    cfg = ExperimentConfig(name=a.name, gan=gan, train_frac=a.train_frac)
    exp = run(prices, cfg)
    files = export(exp, a.output, a.models, a.dashboard or None)
    s = json.loads((Path(a.output) / "evaluation_summary.json").read_text())
    print(f"\nTempora  |  {a.name}  |  {exp.results['meta']['start']} to {exp.results['meta']['end']}  |  "
          f"{exp.seconds:.0f}s\n")
    print("  Realism checks")
    for c in exp.results["scorecard"]:
        print(f"    {'PASS' if c['pass'] else 'FAIL'}  {c['name']:<26} real {c['real']:>8}   synthetic {c['synthetic']:>8}")
    d = s["discriminative"]
    print(f"\n  Discriminative accuracy   {d['accuracy']:.1%}  (50% = indistinguishable)")
    print(f"  Wasserstein distance      {s['distances']['wasserstein_rel']:.3f} × daily volatility")
    cov = s["coverage"]
    print("  Forecast interval hits    " + ", ".join(f"{float(k):.0%} → {v:.0%}" for k, v in cov.items()))
    print(f"  95% VaR breach rate       {s['var_breach_rate']:.1%}  (target 5%)")
    print(f"\n  {len(files)} files written; dashboard data -> {a.dashboard}")
    return 0


def cmd_sample(a: argparse.Namespace) -> int:
    import numpy as np
    import pandas as pd

    from .gan import WGANGP
    from .pipeline import _chain
    from .transforms import ReturnTransform

    model = WGANGP.load(Path(a.models) / f"wgan_gp_{a.name}.npz")
    tr = ReturnTransform.from_dict(json.loads((Path(a.models) / f"transform_{a.name}.json").read_text()))
    k, step = model.cfg.conditional, model.cfg.ts_dim - model.cfg.conditional
    rng = np.random.default_rng(a.seed)
    model.rng = rng
    n_windows = int(np.ceil(a.days / step))
    paths = {}
    for i in range(a.paths):
        lr = _chain(model, tr, rng.standard_normal(k) * 0.5, n_windows)[: a.days]
        paths[f"path_{i + 1}"] = a.start_price * np.exp(np.cumsum(lr))
    df = pd.DataFrame(paths)
    df.index.name = "day"
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.round(4).to_csv(out)
    print(f"Wrote {a.paths} synthetic price paths of {a.days} days -> {out}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="tempora", description="Synthetic financial time series with WGAN-GP")
    sub = p.add_subparsers(dest="command", required=True)

    d = sub.add_parser("data", help="Create or download price data")
    d.add_argument("source", choices=["simulate", "download"])
    d.add_argument("--ticker", default="AAPL")
    d.add_argument("--start", default="2010-01-01")
    d.add_argument("--days", type=int, default=3000)
    d.add_argument("--seed", type=int, default=7)
    d.add_argument("--out")
    d.set_defaults(func=cmd_data)

    t = sub.add_parser("train", help="Train, evaluate and export results")
    t.add_argument("--data", default=DEFAULT_DATA)
    t.add_argument("--name", default="benchmark")
    t.add_argument("--iterations", type=int, default=4000)
    t.add_argument("--latent-dim", type=int, default=16)
    t.add_argument("--hidden", type=int, default=128)
    t.add_argument("--batch-size", type=int, default=128)
    t.add_argument("--train-frac", type=float, default=0.8)
    t.add_argument("--seed", type=int, default=42)
    t.add_argument("--output", default="data/processed")
    t.add_argument("--models", default="models")
    t.add_argument("--dashboard", default="dashboard/data/tempora_data.js")
    t.set_defaults(func=cmd_train)

    s = sub.add_parser("sample", help="Generate synthetic price paths from a trained model")
    s.add_argument("--name", default="benchmark")
    s.add_argument("--models", default="models")
    s.add_argument("--paths", type=int, default=100)
    s.add_argument("--days", type=int, default=250)
    s.add_argument("--start-price", type=float, default=100.0)
    s.add_argument("--seed", type=int, default=0)
    s.add_argument("--out", default="data/processed/sampled_paths.csv")
    s.set_defaults(func=cmd_sample)
    return p


def main(argv: list[str] | None = None) -> int:
    a = build_parser().parse_args(argv)
    return a.func(a)


if __name__ == "__main__":
    raise SystemExit(main())
