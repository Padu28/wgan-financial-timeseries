"""Conditional WGAN-GP in pure NumPy, with analytic gradients.

Why NumPy? It trains a 23-step return window model in a couple of minutes on a
laptop CPU, needs no GPU or deep-learning framework, is fully deterministic, and
every gradient (including the gradient-penalty term, which needs derivatives of
derivatives) is written out explicitly and verified against finite differences
in ``tests/test_gan.py``. A PyTorch implementation of the same model lives in
:mod:`tempora.torch_backend` for larger experiments.

Model
-----
* **Generator** ``G``: noise ``z`` (``latent_dim``) whose first ``conditional``
  entries are replaced by the real first returns of the window (the original
  project's conditioning trick). Two LeakyReLU layers with a residual skip,
  then two heads: a linear *shape* head for the remaining ``ts_dim - conditional``
  steps and a scalar *volatility* head ``v``; the output is
  ``b·tanh(shape · exp(v) / b)``. The multiplicative volatility head lets one
  latent factor scale a whole window up or down, which is how volatility
  clustering appears in real markets, and the tanh keeps values inside the
  range seen in training (the inverse Lambert W transform is very sensitive to
  extreme values).
  The generated window is ``[real conditioning steps, G(z)]``.
* **Critic** ``D``: a 3-layer LeakyReLU MLP on the whole window.
* **Loss** (Gulrajani et al., 2017)::

      L_D = E[D(fake)] - E[D(real)] + λ E[(||∇x D(x̂)|| - 1)²]
      L_G = -E[D(fake)]

  with ``x̂`` sampled uniformly on lines between real and fake windows. The
  penalty is per sample and two-sided, as in the paper (the original code
  penalised only the batch-mean norm above 1).

Because LeakyReLU is piecewise linear, ``∇x D`` is linear in each weight matrix
given the activation pattern, which makes the penalty gradient compact (see
``Critic.gp_backward``).
"""

from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np

ALPHA = 0.2  # LeakyReLU slope


def _lrelu(a: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    s = np.where(a > 0, 1.0, ALPHA)
    return a * s, s


def _he(rng: np.random.Generator, fan_out: int, fan_in: int) -> np.ndarray:
    return rng.normal(0, np.sqrt(2.0 / fan_in), (fan_out, fan_in))


class Params(dict):
    """A dict of named parameter arrays."""

    def copy(self) -> Params:  # type: ignore[override]
        return Params({k: v.copy() for k, v in self.items()})


# --------------------------------------------------------------------- critic
class Critic:
    def __init__(self, ts_dim: int, hidden: int = 128, rng: np.random.Generator | None = None):
        rng = rng or np.random.default_rng(0)
        self.p = Params(W1=_he(rng, hidden, ts_dim), b1=np.zeros(hidden),
                        W2=_he(rng, hidden, hidden), b2=np.zeros(hidden),
                        w3=rng.normal(0, np.sqrt(1.0 / hidden), hidden), b3=np.zeros(1))

    def forward(self, x: np.ndarray) -> tuple[np.ndarray, dict]:
        p = self.p
        a1 = x @ p["W1"].T + p["b1"]
        h1, s1 = _lrelu(a1)
        a2 = h1 @ p["W2"].T + p["b2"]
        h2, s2 = _lrelu(a2)
        out = h2 @ p["w3"] + p["b3"][0]
        return out, {"x": x, "h1": h1, "s1": s1, "h2": h2, "s2": s2}

    def backward(self, cache: dict, dout: np.ndarray) -> tuple[Params, np.ndarray]:
        """Gradients of ``sum(dout * D(x))`` w.r.t. parameters and input."""
        p = self.p
        g = Params()
        g["w3"] = cache["h2"].T @ dout
        g["b3"] = np.array([dout.sum()])
        da2 = (dout[:, None] * p["w3"]) * cache["s2"]
        g["W2"] = da2.T @ cache["h1"]
        g["b2"] = da2.sum(0)
        da1 = (da2 @ p["W2"]) * cache["s1"]
        g["W1"] = da1.T @ cache["x"]
        g["b1"] = da1.sum(0)
        dx = da1 @ p["W1"]
        return g, dx

    def input_grad(self, cache: dict) -> tuple[np.ndarray, dict]:
        """Per-sample ∇x D(x), i.e. ((s2 ⊙ w3) W2 ⊙ s1) W1."""
        u2 = cache["s2"] * self.p["w3"]
        v1 = u2 @ self.p["W2"]
        u1 = cache["s1"] * v1
        return u1 @ self.p["W1"], {"u2": u2, "u1": u1, "s1": cache["s1"], "s2": cache["s2"]}

    def gradient_penalty(self, x_hat: np.ndarray, weight: float) -> tuple[float, Params, float]:
        """λ·mean((||∇x D(x̂)|| - 1)²), its parameter gradients and the mean norm."""
        _, cache = self.forward(x_hat)
        gx, gc = self.input_grad(cache)
        norm = np.sqrt((gx ** 2).sum(1) + 1e-12)
        b = len(x_hat)
        penalty = weight * np.mean((norm - 1) ** 2)
        gamma = (2 * weight / b) * ((norm - 1) / norm)[:, None] * gx      # dP/d(gx)
        g = Params({k: np.zeros_like(v) for k, v in self.p.items()})
        g["W1"] = gc["u1"].T @ gamma
        du1 = gamma @ self.p["W1"].T
        dv1 = du1 * gc["s1"]
        g["W2"] = gc["u2"].T @ dv1
        du2 = dv1 @ self.p["W2"].T
        g["w3"] = (du2 * gc["s2"]).sum(0)
        return float(penalty), g, float(norm.mean())


# ------------------------------------------------------------------ generator
class Generator:
    def __init__(self, latent_dim: int, out_dim: int, hidden: int = 128,
                 rng: np.random.Generator | None = None, bound: float = 4.0):
        rng = rng or np.random.default_rng(1)
        self.latent_dim, self.out_dim, self.bound = latent_dim, out_dim, bound
        self.p = Params(G1=_he(rng, hidden, latent_dim), c1=np.zeros(hidden),
                        G2=_he(rng, hidden, hidden) * 0.5, c2=np.zeros(hidden),
                        G3=rng.normal(0, np.sqrt(1.0 / hidden), (out_dim, hidden)), c3=np.zeros(out_dim),
                        gv=rng.normal(0, 0.1 / np.sqrt(hidden), hidden), cv=np.zeros(1))

    def forward(self, z: np.ndarray) -> tuple[np.ndarray, dict]:
        p = self.p
        a1 = z @ p["G1"].T + p["c1"]
        k1, s1 = _lrelu(a1)
        a2 = k1 @ p["G2"].T + p["c2"]
        r2, s2 = _lrelu(a2)
        k2 = r2 + k1                                   # residual skip
        a = k2 @ p["G3"].T + p["c3"]                   # shape of the path
        scale = np.exp(np.clip(k2 @ p["gv"] + p["cv"][0], -4, 4))[:, None]  # window volatility
        raw = a * scale
        th = np.tanh(raw / self.bound)
        y = self.bound * th                            # smooth bound to the data range
        return y, {"z": z, "k1": k1, "s1": s1, "s2": s2, "k2": k2, "th": th, "a": a, "scale": scale, "raw": raw}

    def backward(self, cache: dict, dy: np.ndarray) -> Params:
        p = self.p
        g = Params()
        draw = dy * (1 - cache["th"] ** 2)
        da = draw * cache["scale"]
        dv = (draw * cache["raw"]).sum(1)              # d raw / d v = raw
        g["G3"] = da.T @ cache["k2"]
        g["c3"] = da.sum(0)
        g["gv"] = cache["k2"].T @ dv
        g["cv"] = np.array([dv.sum()])
        dk2 = da @ p["G3"] + dv[:, None] * p["gv"]
        da2 = dk2 * cache["s2"]
        g["G2"] = da2.T @ cache["k1"]
        g["c2"] = da2.sum(0)
        dk1 = da2 @ p["G2"] + dk2
        da1 = dk1 * cache["s1"]
        g["G1"] = da1.T @ cache["z"]
        g["c1"] = da1.sum(0)
        return g


# ------------------------------------------------------------------ optimiser
class Adam:
    def __init__(self, params: Params, lr: float = 1e-4, betas=(0.0, 0.9), eps: float = 1e-8):
        self.p, self.lr, self.b1, self.b2, self.eps = params, lr, betas[0], betas[1], eps
        self.m = {k: np.zeros_like(v) for k, v in params.items()}
        self.v = {k: np.zeros_like(v) for k, v in params.items()}
        self.t = 0

    def step(self, grads: Params, lr: float | None = None) -> None:
        self.t += 1
        lr = self.lr if lr is None else lr
        for k, g in grads.items():
            self.m[k] = self.b1 * self.m[k] + (1 - self.b1) * g
            self.v[k] = self.b2 * self.v[k] + (1 - self.b2) * g * g
            mhat = self.m[k] / (1 - self.b1 ** self.t) if self.b1 > 0 else self.m[k]
            vhat = self.v[k] / (1 - self.b2 ** self.t)
            self.p[k] -= lr * mhat / (np.sqrt(vhat) + self.eps)


def cyclic_lr(step: int, base: float, peak: float, up: int = 100, down: int = 900) -> float:
    """Triangular cyclical learning rate, as in the original project."""
    pos = step % (up + down)
    return base + (peak - base) * (pos / up if pos < up else 1 - (pos - up) / down)


# ---------------------------------------------------------------- the WGAN-GP
@dataclass
class GANConfig:
    ts_dim: int = 23
    latent_dim: int = 16
    conditional: int = 3
    hidden: int = 128
    batch_size: int = 128
    critic_iters: int = 5
    gp_weight: float = 10.0
    lr_g: tuple = (1e-4, 6e-4)
    lr_d: tuple = (1e-4, 8e-4)
    iterations: int = 4000
    seed: int = 42
    snapshot_every: int = 250
    output_bound: float = 4.0


@dataclass
class History:
    g_loss: list = field(default_factory=list)
    d_loss: list = field(default_factory=list)
    wasserstein: list = field(default_factory=list)
    gp: list = field(default_factory=list)
    grad_norm: list = field(default_factory=list)
    lr_g: list = field(default_factory=list)
    lr_d: list = field(default_factory=list)


class WGANGP:
    """Conditional WGAN-GP for fixed-length windows of transformed returns."""

    def __init__(self, config: GANConfig | None = None):
        self.cfg = config or GANConfig()
        c = self.cfg
        rng = np.random.default_rng(c.seed)
        self.rng = rng
        self.G = Generator(c.latent_dim, c.ts_dim - c.conditional, c.hidden, rng, c.output_bound)
        self.D = Critic(c.ts_dim, c.hidden, rng)
        self.opt_g = Adam(self.G.p, c.lr_g[0])
        self.opt_d = Adam(self.D.p, c.lr_d[0])
        self.history = History()
        self.step = 0

    # -- sampling
    def _noise(self, cond: np.ndarray) -> np.ndarray:
        z = self.rng.standard_normal((len(cond), self.cfg.latent_dim))
        z[:, :self.cfg.conditional] = cond
        return z

    def generate(self, cond: np.ndarray, n_per: int = 1) -> np.ndarray:
        """Generate full windows for each conditioning prefix (shape ``(len(cond)*n_per, ts_dim)``)."""
        cond = np.repeat(np.atleast_2d(cond), n_per, axis=0)
        y, _ = self.G.forward(self._noise(cond))
        return np.concatenate([cond, y], axis=1)

    # -- training
    def train(self, data: np.ndarray, iterations: int | None = None,
              callback: Callable[[WGANGP, int], None] | None = None) -> History:
        c = self.cfg
        k = c.conditional
        n_iter = iterations or c.iterations
        b = c.batch_size
        h = self.history
        for _ in range(n_iter):
            lr_d = cyclic_lr(self.step, *c.lr_d)
            lr_g = cyclic_lr(self.step, *c.lr_g)
            for _ in range(c.critic_iters):
                real = data[self.rng.integers(0, len(data), b)]
                y, _ = self.G.forward(self._noise(real[:, :k]))
                fake = np.concatenate([real[:, :k], y], axis=1)
                out_r, cache_r = self.D.forward(real)
                out_f, cache_f = self.D.forward(fake)
                g_r, _ = self.D.backward(cache_r, -np.ones(b) / b)
                g_f, _ = self.D.backward(cache_f, np.ones(b) / b)
                t = self.rng.random((b, 1))
                gp, g_gp, norm = self.D.gradient_penalty(t * real + (1 - t) * fake, c.gp_weight)
                grads = Params({n: g_r[n] + g_f[n] + g_gp[n] for n in self.D.p})
                self.opt_d.step(grads, lr_d)
            w_est = float(out_r.mean() - out_f.mean())
            h.d_loss.append(-w_est + gp)
            h.wasserstein.append(w_est)
            h.gp.append(gp)
            h.grad_norm.append(norm)

            real = data[self.rng.integers(0, len(data), b)]
            z = self._noise(real[:, :k])
            y, cache_g = self.G.forward(z)
            fake = np.concatenate([real[:, :k], y], axis=1)
            out_f, cache_f = self.D.forward(fake)
            _, dx = self.D.backward(cache_f, -np.ones(b) / b)
            self.opt_g.step(self.G.backward(cache_g, dx[:, k:]), lr_g)
            h.g_loss.append(float(-out_f.mean()))
            h.lr_g.append(lr_g)
            h.lr_d.append(lr_d)
            self.step += 1
            if callback:
                callback(self, self.step)
        return h

    # -- persistence
    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        arrays = {f"G_{k}": v for k, v in self.G.p.items()} | {f"D_{k}": v for k, v in self.D.p.items()}
        np.savez_compressed(path, **arrays, config=json.dumps(asdict(self.cfg)), step=self.step)

    @classmethod
    def load(cls, path: str | Path) -> WGANGP:
        z = np.load(path, allow_pickle=False)
        cfg = GANConfig(**{k: tuple(v) if isinstance(v, list) else v
                           for k, v in json.loads(str(z["config"])).items()})
        m = cls(cfg)
        for k in m.G.p:
            m.G.p[k][...] = z[f"G_{k}"]
        for k in m.D.p:
            m.D.p[k][...] = z[f"D_{k}"]
        m.step = int(z["step"])
        return m
