"""Return transforms used before training and their exact inverses.

The pipeline follows the original project:

1. log returns ``r_t = log(P_t / P_{t-1})``
2. standardise to zero mean, unit variance
3. **inverse Lambert W** transform: heavy-tailed returns are mapped to a much
   more Gaussian variable (Goerg, 2015), which is far easier for a GAN to learn
4. standardise again

``inverse`` undoes every step, so generated values become prices again.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.special import lambertw


@dataclass
class ReturnTransform:
    delta: float = 0.6
    mean1: float = 0.0
    std1: float = 1.0
    mean2: float = 0.0
    std2: float = 1.0
    clip: float = 6.0
    fitted: bool = False

    def fit(self, log_returns: np.ndarray) -> ReturnTransform:
        r = np.asarray(log_returns, dtype=float)
        self.mean1, self.std1 = float(r.mean()), float(r.std())
        u = (r - self.mean1) / self.std1
        w = self._gaussianize(u)
        self.mean2, self.std2 = float(w.mean()), float(w.std())
        # generated values beyond the observed range would explode through exp(δw²/2)
        self.clip = float(1.05 * np.abs(w).max())
        self.fitted = True
        return self

    def _gaussianize(self, u: np.ndarray) -> np.ndarray:
        return (np.sign(u) * np.sqrt(lambertw(self.delta * u ** 2).real / self.delta)).real

    def transform(self, log_returns: np.ndarray) -> np.ndarray:
        self._check()
        u = (np.asarray(log_returns, dtype=float) - self.mean1) / self.std1
        return (self._gaussianize(u) - self.mean2) / self.std2

    def inverse(self, z: np.ndarray) -> np.ndarray:
        """Map model space back to log returns."""
        self._check()
        w = np.clip(np.asarray(z, dtype=float) * self.std2 + self.mean2, -self.clip, self.clip)
        u = w * np.exp(0.5 * self.delta * w ** 2)
        return u * self.std1 + self.mean1

    def to_dict(self) -> dict:
        return {k: getattr(self, k) for k in ("delta", "mean1", "std1", "mean2", "std2", "clip")}

    @classmethod
    def from_dict(cls, d: dict) -> ReturnTransform:
        return cls(**d, fitted=True)

    def _check(self) -> None:
        if not self.fitted:
            raise RuntimeError("ReturnTransform must be fitted first")


def log_returns(prices: np.ndarray) -> np.ndarray:
    p = np.asarray(prices, dtype=float)
    return np.diff(np.log(p))


def to_prices(log_rets: np.ndarray, start_price: float) -> np.ndarray:
    """Rebuild a price path (including the start price) from log returns."""
    lr = np.asarray(log_rets, dtype=float)
    return start_price * np.exp(np.concatenate([[0.0], np.cumsum(lr, axis=-1)])) if lr.ndim == 1 else \
        start_price[:, None] * np.exp(np.concatenate([np.zeros((lr.shape[0], 1)), np.cumsum(lr, axis=1)], axis=1))


def windows(x: np.ndarray, length: int, stride: int = 4) -> np.ndarray:
    """Overlapping windows (the original used length 23, stride 4)."""
    x = np.asarray(x)
    if len(x) < length:
        raise ValueError("Series shorter than the window length")
    starts = range(0, len(x) - length + 1, stride)
    return np.stack([x[s:s + length] for s in starts])
