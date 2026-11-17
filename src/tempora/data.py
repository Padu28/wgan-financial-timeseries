"""Price data: loading CSVs, downloading tickers and simulating a benchmark market."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


def load_prices(path: str | Path, price_col: str | None = None) -> pd.Series:
    """Load a daily price series from CSV and return it indexed by date.

    Handles the common layouts:

    * ``date,Close`` (the original project's ``dataAPPL.csv``)
    * Yahoo Finance / ``yfinance`` exports, including the extra ticker row that
      the original code skipped with ``skiprows=[1]``
    * any CSV with a date-like first column and a ``Close`` / ``Adj Close`` /
      ``price`` column
    """
    df = pd.read_csv(path)
    # yfinance multi-index export: 2nd/3rd rows hold ticker names / blanks
    first = df.columns[0]
    while len(df) and not _looks_like_date(df.iloc[0][first]):
        df = df.iloc[1:]
    date_col = next((c for c in df.columns if c.lower() in ("date", "datetime", "timestamp", "price")
                     and _looks_like_date(df[c].iloc[0])), first)
    if price_col is None:
        for cand in ("Adj Close", "adj_close", "Close", "close", "price", "Price"):
            if cand in df.columns and cand != date_col:
                price_col = cand
                break
    if price_col is None:
        raise ValueError(f"Could not find a price column in {path}; pass price_col=...")
    s = pd.Series(pd.to_numeric(df[price_col], errors="coerce").to_numpy(),
                  index=pd.to_datetime(df[date_col], errors="coerce"), name="price")
    s = s[s.index.notna()].dropna()
    s = s[~s.index.duplicated()].sort_index()
    if (s <= 0).any():
        raise ValueError("Prices must be strictly positive to compute log returns")
    if len(s) < 100:
        raise ValueError(f"Need at least 100 prices, got {len(s)}")
    return s


def _looks_like_date(v) -> bool:
    try:
        pd.Timestamp(str(v))
        return True
    except (ValueError, TypeError):
        return False


def download(ticker: str = "AAPL", start: str = "2010-01-01", end: str | None = None) -> pd.Series:
    """Download adjusted daily closes with ``yfinance`` (optional dependency)."""
    try:
        import yfinance as yf
    except ImportError as exc:  # pragma: no cover - optional
        raise SystemExit("Install the downloader first: pip install yfinance") from exc
    df = yf.download(ticker, start=start, end=end, auto_adjust=True, progress=False)
    if df.empty:
        raise SystemExit(f"No data returned for {ticker}")
    close = df["Close"]
    if isinstance(close, pd.DataFrame):
        close = close.iloc[:, 0]
    close.name = "price"
    return close.dropna()


def simulate_market(n_days: int = 3000, seed: int = 7, start: str = "2012-01-02",
                    mu: float = 0.0004, omega: float = 3e-6, alpha: float = 0.05,
                    gamma: float = 0.08, beta: float = 0.88, nu: float = 5.0) -> pd.Series:
    """Simulate a GJR-GARCH(1,1) price path with Student-t shocks.

    This is the standard benchmark for generative models of returns: it has the
    stylized facts of real equity returns (fat tails, volatility clustering,
    near-zero return autocorrelation and the leverage effect, where falls raise
    volatility more than rises), and the true process is known.
    """
    rng = np.random.default_rng(seed)
    eps = rng.standard_t(nu, n_days) * np.sqrt((nu - 2) / nu)
    r = np.empty(n_days)
    var = omega / (1 - alpha - beta - gamma / 2)
    for t in range(n_days):
        r[t] = mu + np.sqrt(var) * eps[t]
        shock = r[t] - mu
        var = omega + (alpha + gamma * (shock < 0)) * shock ** 2 + beta * var
    prices = 100 * np.exp(np.cumsum(r))
    idx = pd.bdate_range(start, periods=n_days)
    return pd.Series(prices, index=idx, name="price")


def save_prices(s: pd.Series, path: str | Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    s.rename("Close").rename_axis("date").to_frame().to_csv(path, float_format="%.6f")
