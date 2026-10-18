"""Common forecaster interface (fit/predict) every model wrapper below implements.

Every Phase 2/3 model (Seasonal Naive now; ARIMA, SARIMA, Holt-Winters,
Croston, TSB, XGBoost, LSTM/GRU, Prophet later) wraps itself behind this
same two-method interface, so src/evaluation/backtesting.py can run any
of them through an identical rolling-origin loop with no model-specific
branching.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np
import pandas as pd


class BaseForecaster(ABC):
    """A forecaster that fits on a chronological history and predicts forward.

    `fit` is called once per backtest fold, on the data available up to
    that fold's origin only (an expanding window), and must not look
    beyond it. `predict` then returns exactly `horizon` forecasts for
    the days immediately following the end of the series `fit` was
    given -- it never has access to the actual values for those future
    days, since by construction the backtest loop hasn't revealed them
    yet.
    """

    #: Short, stable identifier used in result tables and saved artifacts.
    name: str = "BaseForecaster"

    @abstractmethod
    def fit(self, history: pd.Series) -> "BaseForecaster":
        """Fit on `history`: a date-indexed Series of the target variable, ascending.

        Returns self, so `forecaster.fit(history).predict(horizon)` chains.
        """
        raise NotImplementedError

    @abstractmethod
    def predict(self, horizon: int) -> np.ndarray:
        """Return `horizon` forecasts for the days immediately after the fitted history."""
        raise NotImplementedError

    def fit_diagnostics(self) -> dict:
        """Optional extra diagnostics captured during the most recent `fit()` call.

        Default is an empty dict, so models with nothing fit-time to
        report (SeasonalNaive has no estimated parameters at all) need
        no change. Models that fit via an optimizer (ARIMA, SARIMA,
        Holt-Winters) override this to report things like AIC, the
        number of observations fit on, and any warnings raised during
        fitting, so `src.evaluation.backtesting.run_backtest` can record
        them per fold without needing to know which concrete model
        produced them.
        """
        return {}
