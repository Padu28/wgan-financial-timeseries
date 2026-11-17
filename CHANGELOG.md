# Changelog

Format: [Keep a Changelog](https://keepachangelog.com/en/1.1.0/). Versioning: [SemVer](https://semver.org/).

## [2.0.0] - 2026-11-17

### Added
* `tempora` package and CLI (`tempora data | train | sample`).
* Dependency-light NumPy WGAN-GP engine with hand-derived gradients, including the gradient-penalty term, all verified against finite differences.
* PyTorch backend with the same architecture and weight export to NumPy.
* Volatility head and tanh output bound in the generator.
* Per-sample, two-sided gradient penalty as in Gulrajani et al. (2017).
* Evaluation suite: stylized-fact scorecard, Wasserstein and KS distances, QQ, discriminative score, PCA embedding, interval coverage, PIT, CRPS and VaR backtest on a held-out period.
* Chained long synthetic paths and `tempora sample` for one-year scenario sets.
* Simulated GJR-GARCH-t benchmark market; loader for the original `dataAPPL.csv` and yfinance exports; optional download of any ticker.
* Interactive dashboard: scenario fan with reveal, realism scorecard and stylized-fact charts, "real or synthetic?" game, risk lab, training replay, model card; dark mode, keyboard shortcuts, responsive layout.
* Tests, linting, GitHub Actions CI and GitHub Pages deployment.

### Changed
* Hard-coded absolute paths replaced with CLI options.
* Training uses stride-1 windows (stride 4 in the original) and a held-out test period.
* Latent dimension 16 (10 in the original).

## [1.0.0] - 2020

* Original thesis code by Casper Hogenboom: conditional WGAN-GP in PyTorch on AAPL prices with Lambert W preprocessing and cyclical learning rates.
