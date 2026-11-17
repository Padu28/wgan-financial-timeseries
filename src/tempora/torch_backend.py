"""PyTorch implementation of the same conditional WGAN-GP (optional, for GPUs and larger models).

    pip install torch
    python -m tempora.torch_backend --data data/raw/benchmark_prices.csv --iterations 6000

The architecture mirrors :mod:`tempora.gan` layer for layer, so trained weights
can be copied into the NumPy model with :func:`to_numpy` and evaluated with the
regular pipeline and dashboard. The gradient penalty uses autograd with
``create_graph=True`` (double backpropagation).
"""

from __future__ import annotations

import argparse

import numpy as np

try:
    import torch
    from torch import nn
except ImportError as exc:  # pragma: no cover - optional dependency
    raise ImportError("The PyTorch backend needs torch: pip install torch") from exc

from .gan import WGANGP, GANConfig, cyclic_lr


class Generator(nn.Module):
    def __init__(self, latent_dim: int, out_dim: int, hidden: int = 128, bound: float = 4.0):
        super().__init__()
        self.l1 = nn.Linear(latent_dim, hidden)
        self.l2 = nn.Linear(hidden, hidden)
        self.shape = nn.Linear(hidden, out_dim)
        self.vol = nn.Linear(hidden, 1)
        self.act = nn.LeakyReLU(0.2)
        self.bound = bound

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        k1 = self.act(self.l1(z))
        k2 = self.act(self.l2(k1)) + k1
        scale = torch.exp(torch.clamp(self.vol(k2), -4, 4))
        return self.bound * torch.tanh(self.shape(k2) * scale / self.bound)


class Critic(nn.Module):
    def __init__(self, ts_dim: int, hidden: int = 128):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(ts_dim, hidden), nn.LeakyReLU(0.2),
                                 nn.Linear(hidden, hidden), nn.LeakyReLU(0.2), nn.Linear(hidden, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


def gradient_penalty(critic: Critic, real: torch.Tensor, fake: torch.Tensor, weight: float) -> torch.Tensor:
    t = torch.rand(real.size(0), 1, device=real.device)
    x_hat = (t * real + (1 - t) * fake).requires_grad_(True)
    out = critic(x_hat)
    grads = torch.autograd.grad(out.sum(), x_hat, create_graph=True)[0]
    return weight * ((grads.norm(2, dim=1) - 1) ** 2).mean()


def train(data: np.ndarray, cfg: GANConfig, device: str | None = None) -> tuple[Generator, Critic]:
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(cfg.seed)
    k = cfg.conditional
    x = torch.tensor(data, dtype=torch.float32, device=device)
    G = Generator(cfg.latent_dim, cfg.ts_dim - k, cfg.hidden, cfg.output_bound).to(device)
    D = Critic(cfg.ts_dim, cfg.hidden).to(device)
    opt_g = torch.optim.Adam(G.parameters(), lr=cfg.lr_g[0], betas=(0.0, 0.9))
    opt_d = torch.optim.Adam(D.parameters(), lr=cfg.lr_d[0], betas=(0.0, 0.9))

    def batch() -> tuple[torch.Tensor, torch.Tensor]:
        real = x[torch.randint(0, len(x), (cfg.batch_size,), device=device)]
        z = torch.randn(cfg.batch_size, cfg.latent_dim, device=device)
        z[:, :k] = real[:, :k]
        fake = torch.cat([real[:, :k], G(z)], dim=1)
        return real, fake

    for step in range(cfg.iterations):
        for grp in opt_d.param_groups:
            grp["lr"] = cyclic_lr(step, *cfg.lr_d)
        for grp in opt_g.param_groups:
            grp["lr"] = cyclic_lr(step, *cfg.lr_g)
        for _ in range(cfg.critic_iters):
            real, fake = batch()
            fake = fake.detach()
            loss_d = D(fake).mean() - D(real).mean() + gradient_penalty(D, real, fake, cfg.gp_weight)
            opt_d.zero_grad()
            loss_d.backward()
            opt_d.step()
        _, fake = batch()
        loss_g = -D(fake).mean()
        opt_g.zero_grad()
        loss_g.backward()
        opt_g.step()
        if (step + 1) % 500 == 0:
            print(f"step {step + 1:>6}  W≈{-loss_d.item():.3f}  G {loss_g.item():.3f}")
    return G, D


def to_numpy(G: Generator, D: Critic, cfg: GANConfig) -> WGANGP:
    """Copy PyTorch weights into the NumPy model (for the pipeline, CLI sampling and dashboard)."""
    m = WGANGP(cfg)
    t = lambda p: p.detach().cpu().numpy().astype(float)  # noqa: E731
    m.G.p.update(G1=t(G.l1.weight), c1=t(G.l1.bias), G2=t(G.l2.weight), c2=t(G.l2.bias),
                 G3=t(G.shape.weight), c3=t(G.shape.bias), gv=t(G.vol.weight)[0], cv=t(G.vol.bias))
    lin = [mod for mod in D.net if isinstance(mod, nn.Linear)]
    m.D.p.update(W1=t(lin[0].weight), b1=t(lin[0].bias), W2=t(lin[1].weight), b2=t(lin[1].bias),
                 w3=t(lin[2].weight)[0], b3=t(lin[2].bias))
    m.step = cfg.iterations
    return m


def main() -> None:  # pragma: no cover - CLI wrapper
    from .data import load_prices
    from .transforms import ReturnTransform, log_returns, windows

    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--data", default="data/raw/benchmark_prices.csv")
    ap.add_argument("--iterations", type=int, default=6000)
    ap.add_argument("--out", default="models/wgan_gp_torch.npz")
    a = ap.parse_args()
    r = log_returns(load_prices(a.data).to_numpy())
    n = int(len(r) * 0.8)
    tr = ReturnTransform().fit(r[:n])
    w = windows(tr.transform(r[:n]), 23, 1)
    cfg = GANConfig(iterations=a.iterations, output_bound=float(1.05 * np.abs(w).max()))
    G, D = train(w, cfg)
    to_numpy(G, D, cfg).save(a.out)
    print(f"Saved NumPy-compatible weights to {a.out}")


if __name__ == "__main__":  # pragma: no cover
    main()
