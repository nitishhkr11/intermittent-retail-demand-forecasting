"""Syntetos-Boylan Approximation (SBA) for intermittent demand.

SBA applies the first-order bias correction to Croston's forecast:
``(1 - alpha / 2) * z_hat / q_hat``.  The smoothed demand size and interval
are otherwise estimated exactly as in classic Croston.  All-zero training
series remain undefined and are reported as failed fits rather than being
silently assigned an arbitrary forecast.
"""

from __future__ import annotations

from typing import Any, Dict

import pandas as pd

from src.models.croston import CrostonForecaster


class SBAForecaster(CrostonForecaster):
    """Croston's intermittent-demand rate with the SBA bias correction."""

    def __init__(self, alpha: float = 0.1):
        super().__init__(alpha=alpha)
        self.name = f"SBA(alpha={alpha})"

    def fit(self, history: pd.Series) -> "SBAForecaster":
        super().fit(history)
        assert self._forecast_value is not None
        self._forecast_value *= 1.0 - self.alpha / 2.0
        return self

    def fit_diagnostics(self) -> Dict[str, Any]:
        result = super().fit_diagnostics()
        result["bias_correction_factor"] = 1.0 - self.alpha / 2.0
        result["forecast_value"] = self._forecast_value
        return result
