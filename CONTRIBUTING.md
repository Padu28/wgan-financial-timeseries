# Contributing

```bash
git clone https://github.com/Padu28/wgan-financial-timeseries.git
cd wgan-financial-timeseries
python -m venv .venv && source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -e ".[dev,notebook]"
```

Before a pull request:

```bash
ruff check src tests
pytest                       # includes finite-difference checks of every analytic gradient
tempora train                # if model or evaluation changed, refresh dashboard data
make dashboard               # http://localhost:8000
```

* Any change to the NumPy layers must keep `tests/test_tempora.py` gradient checks passing, and should be mirrored in `torch_backend.py`.
* Keep the core dependencies to NumPy, pandas, SciPy and scikit-learn.
* Report evaluation results honestly, including the checks that fail.
* Update `CHANGELOG.md` and `docs/methodology.md`.
