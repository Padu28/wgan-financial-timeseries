"""Tempora – synthetic financial time series with a conditional WGAN-GP.

Quick start::

    from tempora import simulate_market, run, export
    exp = run(simulate_market())
    export(exp)
"""

__version__ = "2.0.0"

from .data import download, load_prices, simulate_market  # noqa: E402
from .gan import WGANGP, GANConfig  # noqa: E402
from .pipeline import ExperimentConfig, export, run  # noqa: E402
from .transforms import ReturnTransform  # noqa: E402

__all__ = ["load_prices", "download", "simulate_market", "GANConfig", "WGANGP", "ReturnTransform",
           "ExperimentConfig", "run", "export", "__version__"]
