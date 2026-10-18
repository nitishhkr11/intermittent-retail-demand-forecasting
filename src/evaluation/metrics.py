"""RMSE / MAE / MAPE (MAPE only where appropriate).

MAE and RMSE are this project's primary error metrics (see the locked
validation rules) because they stay well-defined for intermittent,
zero-inflated demand. MAPE is still reported alongside them, but is
explicitly protected against division by zero: the CA_1/FOODS_3
aggregate series includes a handful of genuine zero-demand days
(Walmart stores are closed on Christmas Day each year), and a naive
MAPE divides by exactly those days. `mape()` therefore excludes
zero-actual points from its average rather than returning inf/NaN for
the whole window, and the point count it excluded is reported alongside
it so a reader can judge how much of the window MAPE actually covers.
If every point in a window has actual == 0, no percentage error is
computable at all and `mape()` returns None.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np


def _as_float_arrays(actual: np.ndarray, predicted: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    actual = np.asarray(actual, dtype=float)
    predicted = np.asarray(predicted, dtype=float)
    if actual.shape != predicted.shape:
        raise ValueError(
            f"actual and predicted must have the same shape, got {actual.shape} vs {predicted.shape}."
        )
    return actual, predicted


def mae(actual: np.ndarray, predicted: np.ndarray) -> float:
    """Mean Absolute Error. Always defined -- the primary error metric."""
    actual, predicted = _as_float_arrays(actual, predicted)
    return float(np.mean(np.abs(actual - predicted)))


def rmse(actual: np.ndarray, predicted: np.ndarray) -> float:
    """Root Mean Squared Error. Always defined -- the primary error metric."""
    actual, predicted = _as_float_arrays(actual, predicted)
    return float(np.sqrt(np.mean((actual - predicted) ** 2)))


def count_zero_actuals(actual: np.ndarray) -> int:
    """How many points in `actual` are exactly zero (what MAPE has to exclude)."""
    return int(np.sum(np.asarray(actual, dtype=float) == 0))


def mape(actual: np.ndarray, predicted: np.ndarray) -> Optional[float]:
    """Mean Absolute Percentage Error, as a percentage, protected against zero actuals.

    Points where `actual == 0` are excluded from the average -- dividing
    by zero there is undefined, not just numerically unstable. Returns
    None if every point in the window has actual == 0.
    """
    actual, predicted = _as_float_arrays(actual, predicted)
    mask = actual != 0
    if not mask.any():
        return None
    return float(np.mean(np.abs((actual[mask] - predicted[mask]) / actual[mask])) * 100)


@dataclass
class MetricResult:
    """MAE/RMSE/MAPE for one forecast window, bundled with MAPE's own coverage."""

    mae: float
    rmse: float
    mape: Optional[float]
    mape_n_excluded: int
    n: int


def compute_metrics(actual: np.ndarray, predicted: np.ndarray) -> MetricResult:
    """Compute MAE, RMSE, and zero-protected MAPE for one forecast window in a single call."""
    actual_arr, predicted_arr = _as_float_arrays(actual, predicted)
    return MetricResult(
        mae=mae(actual_arr, predicted_arr),
        rmse=rmse(actual_arr, predicted_arr),
        mape=mape(actual_arr, predicted_arr),
        mape_n_excluded=count_zero_actuals(actual_arr),
        n=len(actual_arr),
    )
