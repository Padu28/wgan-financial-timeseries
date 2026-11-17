# Data

| Path | Contents |
| --- | --- |
| `raw/benchmark_prices.csv` | 3,000 business days simulated with GJR-GARCH(1,1)-t (`tempora data simulate`). The dataset behind the published dashboard. |
| `raw/<TICKER>.csv` | Real prices downloaded with `tempora data download --ticker <TICKER>` (not committed) |
| `processed/` | Outputs of `tempora train` and `tempora sample` |

## Using real prices

```bash
pip install yfinance
tempora data download --ticker AAPL --start 2010-01-01      # -> data/raw/AAPL.csv
tempora train --data data/raw/AAPL.csv --name aapl
```

Any CSV with a date column and a `Close`/`Adj Close` column works, including the original project's `dataAPPL.csv` and yfinance exports with extra header rows.

## Processed outputs

| File | Contents |
| --- | --- |
| `stylized_facts.csv` | Real vs synthetic statistics: moments, VaR/ES, autocorrelations and leverage by lag |
| `realism_scorecard.csv` | Pass/fail for each stylized fact |
| `training_history.csv` | Losses, Wasserstein estimate, gradient penalty and gradient norm |
| `forecast_coverage.csv` | Interval coverage by horizon on the test period |
| `var_backtest.csv` | Daily 95% VaR forecasts, realised returns and breaches |
| `synthetic_paths.csv` | Long synthetic price paths (chained windows) |
| `sampled_paths.csv` | 20 one-year synthetic paths from `tempora sample` |
| `evaluation_summary.json` | Headline metrics |
