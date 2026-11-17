# Model card: Tempora conditional WGAN-GP (benchmark)

| | |
| --- | --- |
| **Files** | `wgan_gp_benchmark.npz` (weights + config), `transform_benchmark.json` (return transform) |
| **Task** | Generate 20 days of daily log returns conditioned on the previous 3 |
| **Architecture** | Generator: 16-d input → 2 × 128 LeakyReLU with residual skip → shape head (20) × exp(volatility head), tanh-bounded. Critic: 23 → 128 → 128 → 1 LeakyReLU |
| **Training data** | Simulated GJR-GARCH(1,1)-t market, first 80% of 3,000 days, 2,377 windows |
| **Training** | WGAN-GP, λ = 10, 5 critic steps, batch 128, Adam β = (0, 0.9), cyclical LR, 4,000 iterations, ~65 s on one CPU core |
| **Evaluation** | 7 of 7 stylized-fact checks passed; Wasserstein-1 = 0.041 × daily volatility; 90% interval coverage 92%; 95% VaR breach rate 6.0%; discriminative accuracy 86% |
| **Intended use** | Research, teaching, stress-scenario generation and testing of risk models on synthetic data |
| **Not intended for** | Trading decisions or regulatory capital without validation on the target asset |
| **Limitations** | Short conditioning window; trained on simulated data; a classifier can still distinguish synthetic windows |

Load and sample:

```python
import json
from tempora import WGANGP, ReturnTransform
m = WGANGP.load("models/wgan_gp_benchmark.npz")
tr = ReturnTransform.from_dict(json.load(open("models/transform_benchmark.json")))
```

or `tempora sample --paths 100 --days 250`.
