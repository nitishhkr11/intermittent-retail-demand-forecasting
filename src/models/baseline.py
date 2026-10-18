"""Seasonal Naive baseline forecaster.

The mandatory Phase 2 baseline every later model is compared against.
Weekly seasonality (season_length=7) for daily retail data:
forecast(t) = actual(t - 7). It has no learned parameters -- fitting
only records the last `season_length` observed values, and forecasting
repeats that single observed week forward to cover the requested
horizon (tiling it, e.g. 4x to cover a 28-day horizon).

This deliberately never feeds a prior forecast back in as if it were an
actual -- doing that would silently compound forecast error fold over
fold. Repeating the last truly-observed week is both the textbook
definition of seasonal naive and what avoids that trap.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd

from src.models.base import BaseForecaster


class SeasonalNaive(BaseForecaster):
    """forecast(t) = actual(t - season_length), tiled forward to cover any horizon."""

    name = "SeasonalNaive"

    def __init__(self, season_length: int = 7):
        self.season_length = season_length
        self._last_season: Optional[np.ndarray] = None

    def fit(self, history: pd.Series) -> "SeasonalNaive":
        if len(history) < self.season_length:
            raise ValueError(
                f"Need at least {self.season_length} observations to fit "
                f"SeasonalNaive(season_length={self.season_length}), got {len(history)}."
            )
        self._last_season = history.to_numpy(dtype=float)[-self.season_length:]
        return self

    def predict(self, horizon: int) -> np.ndarray:
        if self._last_season is None:
            raise RuntimeError("SeasonalNaive.predict() called before fit().")
        n_reps = int(np.ceil(horizon / self.season_length))
        tiled = np.tile(self._last_season, n_reps)
        return tiled[:horizon]
