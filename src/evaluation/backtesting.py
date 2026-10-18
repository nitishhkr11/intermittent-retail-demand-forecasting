"""Rolling-origin / walk-forward backtesting engine.

Lays out an expanding-window sequence of (train, test) folds over a
chronological series and runs any src.models.base.BaseForecaster
through every fold identically -- no model-specific logic lives here,
so this is the one place ARIMA, SARIMA, Holt-Winters, Croston, TSB,
XGBoost, LSTM/GRU, and Prophet will all eventually be evaluated through
too. (src.data.splitter.split_final_holdout handles the separate,
one-off final-holdout split; this module is only about development-
period model comparison.)

No random splitting anywhere: every fold's training window is a
contiguous prefix of the series, and every fold's test window is the
`horizon` days immediately following it. A forecaster fit on a given
fold only ever sees observations strictly before that fold's origin --
never its own test window, and never a later fold's.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

import numpy as np
import pandas as pd

from src.models.base import BaseForecaster
from src.utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class RollingOrigin:
    """One backtest fold's boundaries, as integer positions into the series."""

    fold: int
    train_end: int  # exclusive -- series.iloc[:train_end] is this fold's training window
    test_end: int    # exclusive -- series.iloc[train_end:test_end] is this fold's test window

    @property
    def test_start(self) -> int:
        return self.train_end


def generate_rolling_origins(
    n_obs: int, horizon: int = 28, n_folds: int = 5
) -> List[RollingOrigin]:
    """Lay out `n_folds` expanding-window folds, walking backward from the end of the series.

    Folds are spaced `horizon` days apart so their test windows are
    contiguous and non-overlapping, with the *last* fold's test window
    ending exactly at the end of the series -- so with n_obs=1913
    (development length after a 28-day final holdout), horizon=28,
    n_folds=5: fold 5 tests on the last 28 days of development, fold 4
    the 28 days before that, and so on, covering the final 5*28=140
    days of development with five non-overlapping test windows, each
    fold's training window strictly larger than the last.

    Raises ValueError if there isn't enough data for `n_folds` non-
    overlapping `horizon`-length test windows plus at least one
    training observation before the first one.
    """
    min_required = n_folds * horizon + 1
    if n_obs < min_required:
        raise ValueError(
            f"Need at least {min_required} observations for {n_folds} fold(s) "
            f"of horizon {horizon} (got {n_obs})."
        )

    origins = []
    for fold in range(1, n_folds + 1):
        test_end = n_obs - (n_folds - fold) * horizon
        train_end = test_end - horizon
        origins.append(RollingOrigin(fold=fold, train_end=train_end, test_end=test_end))
    return origins


@dataclass
class BacktestFoldResult:
    """One fold's outcome: which model, which fold, and its actual vs. predicted arrays.

    `actuals` is always populated -- it comes straight from the fold's
    test window and does not depend on the model at all. `predictions`
    is only populated when the model's fit/predict succeeded on this
    fold (`success=True`); on failure it is `None` and `error` carries
    `f"{type(exc).__name__}: {exc}"` from whatever exception was raised,
    per the Step 5 requirement to log and report a failed fold rather
    than silently dropping it or fabricating a prediction. `diagnostics`
    carries whatever the model's own `fit_diagnostics()` returned (AIC,
    convergence warnings, ...) -- `{}` for models that don't override it
    (SeasonalNaive) or when a fold failed before diagnostics could be
    gathered.
    """

    model_name: str
    fold: int
    origin_date: pd.Timestamp
    forecast_start: pd.Timestamp
    forecast_end: pd.Timestamp
    dates: pd.DatetimeIndex
    actuals: np.ndarray
    predictions: Optional[np.ndarray] = None
    success: bool = True
    error: Optional[str] = None
    diagnostics: Dict[str, Any] = field(default_factory=dict)


def run_backtest(
    series: pd.Series,
    forecaster_factory: Callable[[], BaseForecaster],
    horizon: int = 28,
    n_folds: int = 5,
) -> List[BacktestFoldResult]:
    """Run one forecaster through every rolling-origin fold of `series`.

    `series` must be a date-indexed (pd.DatetimeIndex), ascending,
    continuous daily Series -- the development period only, never the
    final holdout. A fresh forecaster instance is built via
    `forecaster_factory()` for each fold and fit only on that fold's
    training prefix, so no fold's model ever sees its own test window
    or any later fold's data. Pass a factory (e.g. `lambda:
    SeasonalNaive()`), not a pre-built instance, so each fold starts
    from an unfit model.

    If construction, `fit`, or `predict` raises for a given fold, that
    exception is caught, logged, and recorded on that fold's
    `BacktestFoldResult` (`success=False`, `error=str(exc)`,
    `predictions=None`) rather than propagating -- one model failing on
    one fold must not crash the whole run or silently drop that fold.
    `actuals` for a failed fold is still populated, since it comes from
    the series itself, not the model.
    """
    if not isinstance(series.index, pd.DatetimeIndex):
        raise ValueError("series must be indexed by date (a pd.DatetimeIndex).")
    if not series.index.is_monotonic_increasing:
        raise ValueError("series must be sorted ascending by date.")

    origins = generate_rolling_origins(len(series), horizon=horizon, n_folds=n_folds)
    results = []
    for origin in origins:
        train = series.iloc[: origin.train_end]
        test = series.iloc[origin.test_start : origin.test_end]

        forecaster = None
        predictions: Optional[np.ndarray] = None
        diagnostics: Dict[str, Any] = {}
        success = True
        error: Optional[str] = None
        try:
            forecaster = forecaster_factory()
            forecaster.fit(train)
            predictions = np.asarray(forecaster.predict(horizon), dtype=float)
        except Exception as exc:  # noqa: BLE001 -- deliberately broad: any model failure is a reportable fold, not a crash
            success = False
            error = f"{type(exc).__name__}: {exc}"
        else:
            try:
                diagnostics = forecaster.fit_diagnostics() or {}
            except Exception as diag_exc:  # noqa: BLE001
                logger.warning(
                    "[%s] fold %d: fit_diagnostics() raised %s -- continuing without diagnostics",
                    getattr(forecaster, "name", "UnknownModel"),
                    origin.fold,
                    diag_exc,
                )
                diagnostics = {}

        model_name = getattr(forecaster, "name", None) or getattr(
            forecaster_factory, "__name__", "UnknownModel"
        )

        result = BacktestFoldResult(
            model_name=model_name,
            fold=origin.fold,
            origin_date=train.index[-1],
            forecast_start=test.index[0],
            forecast_end=test.index[-1],
            dates=test.index,
            actuals=test.to_numpy(dtype=float),
            predictions=predictions,
            success=success,
            error=error,
            diagnostics=diagnostics,
        )
        if success:
            logger.info(
                "[%s] fold %d: origin=%s forecast=%s..%s",
                result.model_name,
                result.fold,
                result.origin_date.date(),
                result.forecast_start.date(),
                result.forecast_end.date(),
            )
        else:
            logger.error(
                "[%s] fold %d FAILED during fit/predict: %s",
                result.model_name,
                result.fold,
                error,
            )
        results.append(result)
    return results


def summarize_backtest(results: List[BacktestFoldResult]) -> pd.DataFrame:
    """Build the model/fold/forecast_start/forecast_end/MAE/RMSE/MAPE summary table.

    A failed fold (`result.success is False`) contributes a row with
    `MAE`/`RMSE`/`MAPE`/`mape_n_excluded` all `None` (NaN once in the
    DataFrame) rather than raising or being silently skipped -- `n` is
    still the fold's test-window length, since `actuals` is always
    available. `success` and `error` columns are appended so a reader
    (or a test) can tell a genuinely zero-error fold apart from a failed
    one without re-deriving it from NaN-ness.
    """
    from src.evaluation.metrics import compute_metrics

    rows = []
    for r in results:
        if r.success:
            m = compute_metrics(r.actuals, r.predictions)
            mae_, rmse_, mape_, mape_n_excluded, n = m.mae, m.rmse, m.mape, m.mape_n_excluded, m.n
        else:
            mae_ = rmse_ = mape_ = None
            mape_n_excluded = None
            n = len(r.actuals)
        rows.append(
            {
                "model": r.model_name,
                "fold": r.fold,
                "forecast_start": r.forecast_start.date(),
                "forecast_end": r.forecast_end.date(),
                "MAE": mae_,
                "RMSE": rmse_,
                "MAPE": mape_,
                "mape_n_excluded": mape_n_excluded,
                "n": n,
                "success": r.success,
                "error": r.error,
            }
        )
    return pd.DataFrame(rows)


def plot_fold_forecast(result: BacktestFoldResult, ax=None):
    """Plot one fold's actual vs. predicted values as a forecast-vs-actual line chart.

    A failed fold (`result.success is False`, `predictions is None`)
    has nothing to plot as a forecast line -- rather than crashing on
    `None`, this plots the actual series alone and titles the axes with
    the recorded error, so a failed fold is still visible (not silently
    missing) in a grid of fold plots.
    """
    import matplotlib.pyplot as plt

    if ax is None:
        _, ax = plt.subplots(figsize=(9, 4))

    ax.plot(result.dates, result.actuals, label="Actual", color="black", linewidth=1.5)

    if result.success and result.predictions is not None:
        ax.plot(
            result.dates,
            result.predictions,
            label=f"{result.model_name} forecast",
            color="#8C8C8C",
            linewidth=1.5,
            linestyle="--",
        )
        ax.set_title(
            f"{result.model_name} -- fold {result.fold} "
            f"({result.forecast_start.date()} to {result.forecast_end.date()})"
        )
    else:
        ax.set_title(
            f"{result.model_name} -- fold {result.fold} -- FAILED: {result.error}",
            color="#B00000",
        )
    ax.set_xlabel("Date")
    ax.set_ylabel("Sales (units)")
    ax.legend()
    ax.tick_params(axis="x", rotation=45)
    return ax
