"""How realistic are the synthetic returns? Stylized facts, distances, discriminability and forecasts.

All functions take log returns arranged as windows ``(n_windows, length)``
unless stated otherwise; statistics that span time (autocorrelations,
leverage) are pooled within windows so real and synthetic windows are measured
the same way.
"""

from __future__ import annotations

import numpy as np
from scipy import stats

FACT_LAGS = list(range(1, 11))


def _acf(x: np.ndarray, lag: int) -> float:
    x = x - x.mean()
    v = (x ** 2).mean()
    return float((x[:, :-lag] * x[:, lag:]).mean() / v) if v > 0 else 0.0


def acf_curve(w: np.ndarray, lags=FACT_LAGS) -> list[float]:
    return [round(_acf(w, k), 4) for k in lags]


def leverage_curve(w: np.ndarray, lags=FACT_LAGS) -> list[float]:
    """Correlation between today's return and future squared returns: negative
    values mean falls are followed by higher volatility (the leverage effect)."""
    out = []
    for k in lags:
        a, b = w[:, :-k].ravel(), (w[:, k:] ** 2).ravel()
        out.append(round(float(np.corrcoef(a, b)[0, 1]), 4))
    return out


def var_es(x: np.ndarray, level: float) -> tuple[float, float]:
    """Historical value at risk and expected shortfall of a return sample (as positive losses)."""
    q = np.quantile(x, 1 - level)
    return float(-q), float(-x[x <= q].mean())


def stylized_facts(w: np.ndarray) -> dict:
    """Summary statistics that real asset returns are known to share."""
    flat = w.ravel()
    var95, es95 = var_es(flat, 0.95)
    var99, es99 = var_es(flat, 0.99)
    return {
        "mean": float(flat.mean()), "std": float(flat.std()),
        "skew": float(stats.skew(flat)), "kurtosis": float(stats.kurtosis(flat)),
        "acf_returns": acf_curve(w), "acf_abs": acf_curve(np.abs(w)), "acf_sq": acf_curve(w ** 2),
        "leverage": leverage_curve(w),
        "var95": var95, "es95": es95, "var99": var99, "es99": es99,
        "window_vol_dispersion": float(np.log(w.std(1) + 1e-12).std()),
    }


def histogram(real: np.ndarray, fake: np.ndarray, bins: int = 44, clip: float = 6.0) -> dict:
    s = real.std()
    edges = np.linspace(-clip * s, clip * s, bins + 1)
    hr, _ = np.histogram(np.clip(real.ravel(), edges[0], edges[-1]), edges, density=True)
    hf, _ = np.histogram(np.clip(fake.ravel(), edges[0], edges[-1]), edges, density=True)
    return {"edges": np.round(edges, 6).tolist(), "real": np.round(hr, 4).tolist(), "fake": np.round(hf, 4).tolist()}


def qq(real: np.ndarray, fake: np.ndarray, n: int = 99) -> list[list[float]]:
    q = np.linspace(0.005, 0.995, n)
    return np.round(np.stack([np.quantile(real, q), np.quantile(fake, q)], 1), 6).tolist()


def distances(real: np.ndarray, fake: np.ndarray) -> dict:
    r, f = real.ravel(), fake.ravel()
    return {"wasserstein": float(stats.wasserstein_distance(r, f)),
            "ks": float(stats.ks_2samp(r, f).statistic),
            "wasserstein_rel": float(stats.wasserstein_distance(r, f) / r.std())}


def _features(w: np.ndarray) -> np.ndarray:
    s = np.sort(w, axis=1)
    return np.hstack([w, np.abs(w), s, w.std(1, keepdims=True), np.abs(np.diff(w, axis=1)).mean(1, keepdims=True)])


def discriminative_score(real: np.ndarray, fake: np.ndarray, seed: int = 0) -> dict:
    """Train a classifier to tell real from synthetic windows.

    Test accuracy near 50% means the classifier can't separate them; the
    *score* reported is ``|accuracy - 0.5|`` (0 is perfect, 0.5 is trivially
    separable), following Yoon et al. (TimeGAN, 2019).
    """
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.model_selection import train_test_split

    n = min(len(real), len(fake))
    rng = np.random.default_rng(seed)
    r = real[rng.choice(len(real), n, replace=False)]
    f = fake[rng.choice(len(fake), n, replace=False)]
    x = _features(np.vstack([r, f]) / real.std())
    y = np.r_[np.ones(n), np.zeros(n)]
    xtr, xte, ytr, yte = train_test_split(x, y, test_size=0.3, random_state=seed, stratify=y)
    clf = HistGradientBoostingClassifier(max_iter=150, random_state=seed).fit(xtr, ytr)
    acc = float((clf.predict(xte) == yte).mean())
    return {"accuracy": acc, "score": abs(acc - 0.5)}


def pca_embedding(real: np.ndarray, fake: np.ndarray, n: int = 400, seed: int = 0) -> dict:
    """2-D PCA of standardised windows, fitted on real data."""
    rng = np.random.default_rng(seed)
    r = real[rng.choice(len(real), min(n, len(real)), replace=False)]
    f = fake[rng.choice(len(fake), min(n, len(fake)), replace=False)]
    s = real.std()
    feats = lambda w: np.hstack([np.sort(w, 1), np.abs(w)]) / s  # noqa: E731
    fr, ff = feats(r), feats(f)
    mu = fr.mean(0)
    _, sv, vt = np.linalg.svd(fr - mu, full_matrices=False)
    comp = vt[:2]
    explained = (sv[:2] ** 2 / (sv ** 2).sum()).tolist()
    return {"real": np.round((fr - mu) @ comp.T, 3).tolist(), "fake": np.round((ff - mu) @ comp.T, 3).tolist(),
            "explained": [round(e, 3) for e in explained]}


def crps_sample(samples: np.ndarray, y: float) -> float:
    """Continuous ranked probability score of a sample forecast (lower is better)."""
    s = np.sort(samples)
    n = len(s)
    return float(np.abs(s - y).mean() - (np.abs(s[:, None] - s[None, :]).sum()) / (2 * n * n))


def coverage_backtest(paths: list[np.ndarray], realised: np.ndarray, levels=(0.5, 0.8, 0.9)) -> dict:
    """Check probabilistic forecasts of cumulative returns.

    ``paths[i]`` is an ``(n_paths, horizon)`` array of simulated log returns for
    test window ``i``; ``realised[i]`` the actual ``horizon`` returns. A
    well-calibrated generator puts the realised cumulative return inside its
    central x% interval about x% of the time, and its PIT values (the rank of
    the realised value among simulations) are uniform.
    """
    cov = {str(lv): [] for lv in levels}
    pit, crps = [], []
    horizon = realised.shape[1]
    for p, y in zip(paths, realised):
        cp, cy = np.cumsum(p, 1), np.cumsum(y)
        for lv in levels:
            lo, hi = np.quantile(cp, [(1 - lv) / 2, (1 + lv) / 2], axis=0)
            cov[str(lv)].append(((cy >= lo) & (cy <= hi)).astype(float))
        pit.append(float((cp[:, -1] < cy[-1]).mean()))
        crps.append(crps_sample(cp[:, -1], cy[-1]))
    by_h = {lv: np.mean(v, axis=0).round(4).tolist() for lv, v in cov.items()}
    return {
        "levels": list(levels),
        "coverage_by_horizon": by_h,
        "coverage": {lv: float(np.mean(v)) for lv, v in cov.items()},
        "pit_hist": np.histogram(pit, bins=10, range=(0, 1))[0].tolist(),
        "crps": float(np.mean(crps)),
        "horizon": horizon,
    }


def realism_scorecard(real: dict, fake: dict) -> list[dict]:
    """Pass/fail checks for the main stylized facts, for the dashboard."""
    def close(a, b, tol):
        return abs(a - b) <= tol

    checks = [
        ("Fat tails", "Excess kurtosis well above 0, like real returns",
         fake["kurtosis"] > 1.0, f"{real['kurtosis']:.1f}", f"{fake['kurtosis']:.1f}"),
        ("No linear predictability", "Return autocorrelation close to zero at lag 1",
         abs(fake["acf_returns"][0]) < 0.1, f"{real['acf_returns'][0]:.2f}", f"{fake['acf_returns'][0]:.2f}"),
        ("Volatility clustering", "Absolute returns are positively autocorrelated",
         np.mean(fake["acf_abs"][:5]) > 0.03, f"{np.mean(real['acf_abs'][:5]):.2f}", f"{np.mean(fake['acf_abs'][:5]):.2f}"),
        ("Leverage effect", "Falls are followed by higher volatility",
         np.mean(fake["leverage"][:3]) < 0, f"{np.mean(real['leverage'][:3]):.2f}", f"{np.mean(fake['leverage'][:3]):.2f}"),
        ("Volatility level", "Daily volatility within 20% of the real series",
         close(fake["std"], real["std"], 0.2 * real["std"]), f"{real['std'] * 100:.2f}%", f"{fake['std'] * 100:.2f}%"),
        ("Tail risk", "99% value at risk within 25% of the real series",
         close(fake["var99"], real["var99"], 0.25 * real["var99"]), f"{real['var99'] * 100:.2f}%", f"{fake['var99'] * 100:.2f}%"),
        ("Negative skew", "Large falls more common than large rises",
         (fake["skew"] < 0) == (real["skew"] < 0), f"{real['skew']:.2f}", f"{fake['skew']:.2f}"),
    ]
    return [{"name": n, "test": t, "pass": bool(ok), "real": r, "synthetic": f} for n, t, ok, r, f in checks]
