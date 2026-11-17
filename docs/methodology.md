# Methodology

## 1. Problem

Generate realistic synthetic sequences of daily asset returns, and use the same model as a **probabilistic forecaster**: given the last few days, produce a distribution of possible next 20 days. Synthetic data is used for stress testing, model validation, data augmentation and sharing data without exposing real positions.

## 2. Data and transform

* **Returns.** `r_t = log(P_t / P_{t-1})`.
* **Split.** First 80% of days for training, last 20% held out for forecast and risk backtests. The transform is fitted on training data only.
* **Transform** (following the original project):
  1. standardise: `u = (r − μ) / σ`
  2. inverse Lambert W ("Gaussianise", Goerg 2015): `w = sign(u) · sqrt(W(δu²) / δ)`, with `δ = 0.6` and `W` the Lambert W function. This shrinks heavy tails so the target distribution is close to Gaussian, which GANs learn far more easily.
  3. standardise `w` again.

  The inverse `u = w · exp(δw²/2)` is exact. Because it grows very fast, generated values are clipped to 1.05 × the largest `|w|` seen in training before inversion.
* **Windows.** 23 trading days (3 conditioning + 20 generated); stride 1 for training (2,377 windows on the benchmark), stride 5 for testing.

## 3. Model

**Generator.** Input `z ∈ R^16`, standard normal, with the first 3 entries replaced by the window's real first 3 transformed returns (the original project's conditioning trick).

```
k1 = LeakyReLU(G1 z + c1)                       128 units
k2 = LeakyReLU(G2 k1 + c2) + k1                 residual skip
shape = G3 k2 + c3                              20 values
v     = clip(gv · k2 + cv, −4, 4)               scalar log-volatility
y     = b · tanh(shape · exp(v) / b)            b = 1.05 × max |training value|
window = [real first 3 days, y]
```

The **volatility head** is the main architectural change from the original. One latent factor scales the whole window, which is how calm and turbulent spells appear in real markets. In ablations on the benchmark it doubled the volatility-clustering autocorrelation of generated returns and brought the excess kurtosis from about 6 to about 12 (real: 14.7). The **tanh bound** prevents rare out-of-range outputs that the exponential inverse transform would turn into absurd returns.

**Critic.** `x ∈ R^23 → 128 → 128 → 1`, LeakyReLU (slope 0.2), linear output.

**Loss (WGAN-GP, Gulrajani et al. 2017).**

```
L_D = E[D(fake)] − E[D(real)] + λ · E[(‖∇x D(x̂)‖₂ − 1)²],   x̂ = t·real + (1−t)·fake, t ~ U(0,1)
L_G = −E[D(fake)]
```

with `λ = 10`, 5 critic steps per generator step, batch 128. The penalty is applied per sample and is two-sided, as in the paper; the original code penalised only the batch-mean norm above 1.

**Optimisation.** Adam with `β = (0, 0.9)` and the original project's triangular cyclical learning rates: critic 1e-4 → 8e-4, generator 1e-4 → 6e-4, 100 steps up, 900 down. 4,000 generator iterations.

## 4. The NumPy engine

All forward and backward passes are written out in `src/tempora/gan.py`. The gradient penalty needs derivatives of `‖∇x D‖` with respect to the critic weights (double backpropagation). With LeakyReLU, `∇x D` is linear in each weight matrix given the activation pattern:

```
∇x D = ((s2 ⊙ w3) W2 ⊙ s1) W1,   s1, s2 = LeakyReLU slopes (1 or 0.2)
```

so its gradient has a closed form (`Critic.gradient_penalty`). Every gradient (critic, input, penalty and generator) is checked against central finite differences in `tests/test_tempora.py`, agreeing to about 1e-9. A PyTorch version of the identical architecture (`tempora.torch_backend`) uses autograd for GPU training and can export its weights back to the NumPy model.

## 5. Evaluation

**Stylized facts** (pooled within 20-day windows, real training windows against synthetic windows conditioned on the same prefixes):

| Fact | Statistic |
| --- | --- |
| Fat tails | Excess kurtosis |
| No linear predictability | Autocorrelation of returns |
| Volatility clustering | Autocorrelation of absolute and squared returns |
| Leverage effect | corr(r_t, r²_{t+k}) |
| Asymmetry | Skewness |
| Tail risk | 95% / 99% value at risk and expected shortfall |

The scorecard passes a fact when the synthetic statistic has the right sign and is within a stated tolerance (see `metrics.realism_scorecard`).

**Distribution distance.** Wasserstein-1 distance (reported relative to daily volatility) and Kolmogorov–Smirnov statistic between pooled real and synthetic returns, plus a QQ plot.

**Discriminative score** (Yoon et al., 2019). A gradient-boosting classifier is trained to separate real from synthetic windows using raw, absolute and sorted returns; 50% test accuracy means indistinguishable. This is a deliberately strict test and the main open gap of the current model (86% on the benchmark).

**Forecast calibration** on the held-out test period. For each test window, 200 scenarios are generated from its first 3 days:
* *interval coverage*: the share of realised cumulative returns inside the central 50/80/90% bands, by horizon;
* *PIT histogram*: the rank of the realised 20-day return among the scenarios (uniform if calibrated);
* *CRPS* of the 20-day return;
* *VaR backtest*: next-day 95% VaR from the scenarios; the breach rate should be close to 5%.

## 6. Benchmark data

`tempora data simulate` produces 3,000 business days from a GJR-GARCH(1,1) with Student-t (ν = 5) shocks: `σ²_t = ω + (α + γ·1[ε<0]) ε²_{t−1} + β σ²_{t−1}`, with ω = 3e-6, α = 0.05, γ = 0.08, β = 0.88. Because the true process is known and has the stylized facts above, it is a standard testbed for generative models of returns. Train on real prices with `tempora data download --ticker AAPL` followed by `tempora train --data data/raw/AAPL.csv --name aapl`.

## 7. Limitations

* The conditioning window is short (3 days), so forecasts mainly capture the volatility level, not longer regimes.
* MLP generators treat a window as a vector; temporal architectures (TCN, LSTM, transformers) may capture clustering and leverage more fully.
* The critic can still separate real from synthetic windows; see the roadmap in the README.
* Results on the simulated benchmark don't guarantee the same quality on any particular asset.
