import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from tempora import metrics as M  # noqa: E402
from tempora.cli import main as cli  # noqa: E402
from tempora.data import load_prices, save_prices, simulate_market  # noqa: E402
from tempora.gan import WGANGP, Critic, GANConfig, Generator, cyclic_lr  # noqa: E402
from tempora.pipeline import ExperimentConfig, export, run  # noqa: E402
from tempora.transforms import ReturnTransform, log_returns, to_prices, windows  # noqa: E402


def numgrad(f, arr, eps=1e-6):
    g = np.zeros_like(arr)
    for i in np.ndindex(arr.shape):
        o = arr[i]
        arr[i] = o + eps
        a = f()
        arr[i] = o - eps
        b = f()
        arr[i] = o
        g[i] = (a - b) / (2 * eps)
    return g


# ------------------------------------------------------------ gradients
@pytest.fixture
def rng():
    return np.random.default_rng(3)


def test_critic_gradients(rng):
    D = Critic(7, 9, rng)
    x, c = rng.normal(size=(5, 7)), rng.normal(size=5)
    _, cache = D.forward(x)
    g, dx = D.backward(cache, c)
    f = lambda: float((D.forward(x)[0] * c).sum())  # noqa: E731
    for k, v in D.p.items():
        assert np.allclose(numgrad(f, v), g[k], atol=1e-6), k
    assert np.allclose(numgrad(f, x), dx, atol=1e-6)


def test_gradient_penalty_gradients(rng):
    D = Critic(6, 8, rng)
    x = rng.normal(size=(4, 6))
    _, g, _ = D.gradient_penalty(x, 10.0)
    f = lambda: D.gradient_penalty(x, 10.0)[0]  # noqa: E731
    for k, v in D.p.items():
        assert np.allclose(numgrad(f, v), g[k], atol=1e-5), k


def test_generator_gradients(rng):
    G = Generator(6, 4, 8, rng, bound=3.0)
    z, dy = rng.normal(size=(5, 6)), rng.normal(size=(5, 4))
    _, cache = G.forward(z)
    g = G.backward(cache, dy)
    f = lambda: float((G.forward(z)[0] * dy).sum())  # noqa: E731
    for k, v in G.p.items():
        assert np.allclose(numgrad(f, v), g[k], atol=1e-6), k


def test_generator_output_is_bounded(rng):
    G = Generator(4, 5, 8, rng, bound=2.5)
    y, _ = G.forward(rng.normal(size=(200, 4)) * 50)
    assert np.abs(y).max() <= 2.5


def test_cyclic_lr_shape():
    assert cyclic_lr(0, 1e-4, 8e-4) == pytest.approx(1e-4)
    assert cyclic_lr(100, 1e-4, 8e-4) == pytest.approx(8e-4)
    assert cyclic_lr(1000, 1e-4, 8e-4) == pytest.approx(1e-4)


# ----------------------------------------------------------- transforms
def test_transform_roundtrip():
    r = log_returns(simulate_market(800).to_numpy())
    tr = ReturnTransform().fit(r)
    z = tr.transform(r)
    assert abs(z.mean()) < 1e-6 and abs(z.std() - 1) < 1e-6
    assert np.allclose(tr.inverse(z), r, atol=1e-10)
    assert abs(pd.Series(z).kurt()) < abs(pd.Series(r).kurt())   # Gaussianised
    tr2 = ReturnTransform.from_dict(tr.to_dict())
    assert np.allclose(tr2.inverse(z), r, atol=1e-10)


def test_prices_and_windows():
    p = np.array([100, 101, 99, 102.0])
    assert np.allclose(to_prices(log_returns(p), 100.0), p)
    w = windows(np.arange(10), 4, 2)
    assert w.shape == (4, 4) and w[1, 0] == 2


def test_load_prices_formats(tmp_path):
    s = simulate_market(300)
    save_prices(s, tmp_path / "a.csv")
    assert len(load_prices(tmp_path / "a.csv")) == 300
    # yfinance-style export with extra header rows, as in the original dataAPPL.csv
    lines = ["Price,Close,High", "Ticker,AAPL,AAPL", "Date,,"] + \
            [f"{d.date()},{v:.4f},{v:.4f}" for d, v in s.items()]
    (tmp_path / "b.csv").write_text("\n".join(lines))
    assert np.allclose(load_prices(tmp_path / "b.csv").to_numpy(), s.to_numpy(), atol=1e-3)


def test_benchmark_has_stylized_facts():
    r = log_returns(simulate_market(3000).to_numpy())
    w = windows(r, 23, 1)
    f = M.stylized_facts(w)
    assert f["kurtosis"] > 2
    assert np.mean(f["acf_abs"][:5]) > 0.05
    assert np.mean(f["leverage"][:3]) < 0


# -------------------------------------------------------------- metrics
def test_metrics_basics():
    rng = np.random.default_rng(0)
    a = rng.normal(0, 0.01, (300, 20))
    b = rng.normal(0, 0.01, (300, 20))
    assert M.distances(a, b)["ks"] < 0.05
    assert M.discriminative_score(a, b)["score"] < 0.12
    assert M.discriminative_score(a, b * 3)["accuracy"] > 0.8
    v, e = M.var_es(a.ravel(), 0.95)
    assert 0 < v < e
    assert M.crps_sample(np.zeros(10), 0.0) == pytest.approx(0.0)


def test_coverage_is_calibrated_for_true_model():
    rng = np.random.default_rng(1)
    paths = [rng.normal(0, 0.01, (400, 10)) for _ in range(300)]
    realised = rng.normal(0, 0.01, (300, 10))
    cov = M.coverage_backtest(paths, realised)["coverage"]
    assert abs(cov["0.9"] - 0.9) < 0.04 and abs(cov["0.5"] - 0.5) < 0.05


# ------------------------------------------------------------- pipeline
@pytest.fixture(scope="module")
def experiment():
    cfg = ExperimentConfig(gan=GANConfig(iterations=60, snapshot_every=20, batch_size=64, hidden=32),
                           n_scenarios=40, n_contexts=5, n_dashboard_paths=10, long_path_windows=4)
    return run(simulate_market(900), cfg, verbose=False)


def test_training_learns(experiment):
    h = experiment.model.history
    assert len(h.g_loss) == 60
    assert np.isfinite(h.d_loss).all()
    assert 0.5 < np.mean(h.grad_norm[-10:]) < 1.5          # gradient penalty keeps the critic ~1-Lipschitz


def test_pipeline_outputs(experiment, tmp_path):
    r = experiment.results
    assert {"facts", "scorecard", "coverage", "contexts", "turing", "training", "snapshots", "pca"} <= set(r)
    ctx = r["contexts"][0]
    assert len(ctx["paths"][0]) == experiment.config.gan.ts_dim - experiment.config.gan.conditional
    files = export(experiment, tmp_path / "out", tmp_path / "models", tmp_path / "d.js")
    assert (tmp_path / "d.js").read_text().startswith("window.TEMPORA_DATA=")
    m = WGANGP.load(tmp_path / "models" / "wgan_gp_benchmark.npz")
    assert np.allclose(m.G.p["G1"], experiment.model.G.p["G1"])
    assert len(files) >= 10


def test_cli_end_to_end(tmp_path):
    data = tmp_path / "p.csv"
    assert cli(["data", "simulate", "--days", "700", "--out", str(data)]) == 0
    assert cli(["train", "--data", str(data), "--iterations", "30", "--hidden", "32",
                "--output", str(tmp_path / "o"), "--models", str(tmp_path / "m"),
                "--dashboard", str(tmp_path / "d.js")]) == 0
    out = tmp_path / "s.csv"
    assert cli(["sample", "--models", str(tmp_path / "m"), "--paths", "3", "--days", "50", "--out", str(out)]) == 0
    df = pd.read_csv(out)
    assert df.shape == (50, 4) and (df.iloc[:, 1:] > 0).all().all()


def test_torch_backend_matches_numpy():
    torch = pytest.importorskip("torch")
    from tempora.torch_backend import Critic as TC
    from tempora.torch_backend import Generator as TG
    from tempora.torch_backend import to_numpy

    cfg = GANConfig(ts_dim=8, latent_dim=5, conditional=2, hidden=16, output_bound=3.0)
    G, D = TG(5, 6, 16, 3.0), TC(8, 16)
    m = to_numpy(G, D, cfg)
    z = np.random.default_rng(0).normal(size=(4, 5))
    with torch.no_grad():
        yt = G(torch.tensor(z, dtype=torch.float32)).numpy()
        x = np.random.default_rng(1).normal(size=(4, 8))
        dt = D(torch.tensor(x, dtype=torch.float32)).numpy()
    assert np.allclose(m.G.forward(z)[0], yt, atol=1e-4)
    assert np.allclose(m.D.forward(x)[0], dt, atol=1e-4)
