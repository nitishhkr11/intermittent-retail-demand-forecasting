"""Prophet forecasters for the AGGREGATE CA_1/FOODS_3 daily sales series (Phase 3, notebook 08).

What this module is
-------------------
A thin, leakage-safe wrapper around Facebook/Meta Prophet that plugs into the project's
`BaseForecaster` interface, so the existing `run_backtest` loop evaluates it on exactly the
same five expanding folds and 28-day horizon as the Phase 2 classical models.

Prophet fits  y(t) = trend(t) + seasonality(t) + regressors(t) + noise  where the trend is
piecewise linear with automatically placed changepoints, the seasonality is a Fourier series
(weekly and yearly here) and the regressors are 0/1 calendar facts. In `multiplicative` mode
the seasonal and regressor effects scale the trend instead of being added to it.

Pre-declared variants (config.yaml, section `prophet`; fixed, never tuned)
  Prophet(additive)                 additive seasonality + calendar regressors   (primary)
  Prophet(multiplicative)           multiplicative seasonality + calendar regressors
  Prophet(additive,no-regressors)   additive seasonality, history only (same information
                                    set as the Phase 2 classical models)

Leakage rules enforced here (each has a unit test in tests/models/test_prophet_model.py)
* `fit` receives only the history available at the forecast origin; nothing else is read.
* Future dates are generated from the LAST HISTORY DATE (+1 .. +horizon) -- never from sales.
* Regressors are calendar facts published in advance (SNAP days, M5 event types, the Christmas
  closure). A date missing from the calendar raises; it is never zero-filled. No future
  sales value can reach the model.
* Forecasts and interval bounds are clipped at zero (units sold cannot be negative); the
  number of clipped values is reported in `fit_diagnostics()`.
* MAP estimation (`mcmc_samples: 0`) gives a deterministic point forecast. Uncertainty
  intervals are simulated; the simulation is seeded (`random_seed`) and the caller's global
  NumPy random state is restored afterwards, so intervals are reproducible too.

Prophet is imported lazily. Importing this module never needs prophet; using a forecaster
without it raises an ImportError naming the exact install command. If Prophet's Stan backend
cannot run, a `ProphetBackendError` explains the likely cause -- no other method is ever
substituted and labelled Prophet.
"""

from __future__ import annotations

import logging
import time
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from src.models.base import BaseForecaster
from src.models.xgboost_model import EVENT_FLAG_COLUMNS
from src.utils.config import load_config
from src.utils.logging import get_logger

logger = get_logger(__name__)

SUPPORTED_MODES: Tuple[str, ...] = ("additive", "multiplicative")
#: Prophet needs at least two full weeks (and in practice many more) to estimate a weekly pattern.
MIN_HISTORY_DAYS = 28

PROPHET_INSTALL_COMMAND = "python -m pip install prophet"
PHASE3_REQUIREMENTS_COMMAND = "python -m pip install -r requirements-phase3.txt"


class ProphetBackendError(RuntimeError):
    """Prophet is installed but its Stan backend failed; the message says what to check."""


# =============================================================================================
# Configuration
# =============================================================================================

def _config_error(key: str, message: str) -> ValueError:
    return ValueError(f"Invalid configuration: prophet.{key} {message}")


def _is_int(value: Any) -> bool:
    return isinstance(value, (int, np.integer)) and not isinstance(value, bool)


def _is_number(value: Any) -> bool:
    return isinstance(value, (int, float, np.integer, np.floating)) and not isinstance(value, bool) and bool(np.isfinite(value))


@dataclass(frozen=True)
class ProphetSettings:
    """Validated Prophet settings shared by every variant (config.yaml, section `prophet`)."""

    yearly_seasonality: bool = True
    weekly_seasonality: bool = True
    daily_seasonality: bool = False
    changepoint_prior_scale: float = 0.05
    seasonality_prior_scale: float = 10.0
    n_changepoints: int = 25
    changepoint_range: float = 0.8
    interval_width: float = 0.8
    mcmc_samples: int = 0
    uncertainty_samples: int = 1000
    regressors: Tuple[str, ...] = tuple(EVENT_FLAG_COLUMNS)
    random_seed: int = 42
    horizon: int = 28

    def __post_init__(self) -> None:
        for key in ("yearly_seasonality", "weekly_seasonality", "daily_seasonality"):
            if not isinstance(getattr(self, key), bool):
                raise _config_error(key, f"must be true or false (got {getattr(self, key)!r}).")
        for key in ("changepoint_prior_scale", "seasonality_prior_scale"):
            value = getattr(self, key)
            if not _is_number(value) or value <= 0:
                raise _config_error(key, f"must be a positive number (got {value!r}).")
        if not _is_int(self.n_changepoints) or self.n_changepoints < 0:
            raise _config_error("n_changepoints", f"must be an integer >= 0 (got {self.n_changepoints!r}).")
        if not _is_number(self.changepoint_range) or not 0 < self.changepoint_range <= 1:
            raise _config_error("changepoint_range", f"must be in (0, 1] (got {self.changepoint_range!r}).")
        if not _is_number(self.interval_width) or not 0 < self.interval_width < 1:
            raise _config_error("interval_width", f"must be strictly between 0 and 1 (got {self.interval_width!r}).")
        if not _is_int(self.mcmc_samples) or self.mcmc_samples != 0:
            raise _config_error(
                "mcmc_samples",
                f"must be 0: this wrapper uses MAP estimation (deterministic point forecast); full MCMC is not supported (got {self.mcmc_samples!r}).",
            )
        if not _is_int(self.uncertainty_samples) or self.uncertainty_samples < 1:
            raise _config_error("uncertainty_samples", f"must be an integer >= 1 (got {self.uncertainty_samples!r}).")
        if not _is_int(self.random_seed) or self.random_seed < 0:
            raise _config_error("random_seed", f"must be an integer >= 0 (got {self.random_seed!r}).")
        if not _is_int(self.horizon) or self.horizon < 1:
            raise _config_error("horizon", f"must be an integer >= 1 (got {self.horizon!r}).")
        regressors = tuple(self.regressors)
        if any(not isinstance(r, str) for r in regressors) or len(set(regressors)) != len(regressors):
            raise _config_error("regressors", f"must be a list of unique column names (got {list(self.regressors)!r}).")
        unknown = [r for r in regressors if r not in EVENT_FLAG_COLUMNS]
        if unknown:
            raise _config_error("regressors", f"has unknown name(s) {unknown}; allowed: {list(EVENT_FLAG_COLUMNS)}.")
        object.__setattr__(self, "regressors", regressors)

    @classmethod
    def from_config(cls, config: Optional[Mapping[str, Any]] = None, horizon: Optional[int] = None) -> "ProphetSettings":
        """Read and validate `config["prophet"]`; the horizon defaults to `config["backtest"]["horizon"]`."""
        config = load_config() if config is None else config
        section = config.get("prophet") if isinstance(config, Mapping) else None
        if not isinstance(section, Mapping):
            raise ValueError("Invalid configuration: config.yaml has no 'prophet' section.")
        if horizon is None:
            backtest = config.get("backtest") or {}
            if "horizon" not in backtest:
                raise ValueError("Invalid configuration: backtest.horizon is missing.")
            horizon = backtest["horizon"]
        defaults = cls()
        return cls(
            yearly_seasonality=section.get("yearly_seasonality", defaults.yearly_seasonality),
            weekly_seasonality=section.get("weekly_seasonality", defaults.weekly_seasonality),
            daily_seasonality=section.get("daily_seasonality", defaults.daily_seasonality),
            changepoint_prior_scale=section.get("changepoint_prior_scale", defaults.changepoint_prior_scale),
            seasonality_prior_scale=section.get("seasonality_prior_scale", defaults.seasonality_prior_scale),
            n_changepoints=section.get("n_changepoints", defaults.n_changepoints),
            changepoint_range=section.get("changepoint_range", defaults.changepoint_range),
            interval_width=section.get("interval_width", defaults.interval_width),
            mcmc_samples=section.get("mcmc_samples", defaults.mcmc_samples),
            uncertainty_samples=section.get("uncertainty_samples", defaults.uncertainty_samples),
            regressors=tuple(section.get("regressors", defaults.regressors) or ()),
            random_seed=section.get("random_seed", defaults.random_seed),
            horizon=horizon,
        )


@dataclass(frozen=True)
class ProphetVariant:
    """One pre-declared experiment cell."""

    name: str
    seasonality_mode: str
    use_regressors: bool

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name:
            raise _config_error("variants", f"entries need a non-empty name (got {self.name!r}).")
        if self.seasonality_mode not in SUPPORTED_MODES:
            raise _config_error("variants", f"seasonality_mode must be one of {list(SUPPORTED_MODES)} (got {self.seasonality_mode!r}).")
        if not isinstance(self.use_regressors, bool):
            raise _config_error("variants", f"use_regressors must be true or false (got {self.use_regressors!r}).")


def variants_from_config(config: Optional[Mapping[str, Any]] = None) -> List[ProphetVariant]:
    """The experiment cells declared in `config["prophet"]["variants"]` (validated, unique names)."""
    config = load_config() if config is None else config
    section = config.get("prophet") if isinstance(config, Mapping) else None
    if not isinstance(section, Mapping):
        raise ValueError("Invalid configuration: config.yaml has no 'prophet' section.")
    declared = section.get("variants")
    if not isinstance(declared, Sequence) or isinstance(declared, (str, bytes)) or len(declared) == 0:
        raise _config_error("variants", f"must be a non-empty list of variant entries (got {declared!r}).")
    variants: List[ProphetVariant] = []
    for entry in declared:
        if not isinstance(entry, Mapping) or set(entry) != {"name", "seasonality_mode", "use_regressors"}:
            raise _config_error("variants", f"entries must have exactly the keys name, seasonality_mode, use_regressors (got {entry!r}).")
        variants.append(ProphetVariant(entry["name"], entry["seasonality_mode"], entry["use_regressors"]))
    if len({v.name for v in variants}) != len(variants):
        raise _config_error("variants", "must not list the same variant name twice.")
    return variants


def primary_variant_name(config: Optional[Mapping[str, Any]] = None) -> str:
    """The primary variant's name; it must be one of the declared variants."""
    config = load_config() if config is None else config
    names = [v.name for v in variants_from_config(config)]
    primary = config["prophet"].get("primary_variant")
    if primary not in names:
        raise _config_error("primary_variant", f"must be one of {names} (got {primary!r}).")
    return str(primary)


# =============================================================================================
# Pure helpers (numpy / pandas only -- importable and testable without prophet)
# =============================================================================================

def check_history(history: pd.Series) -> pd.Series:
    """Validate a training history: date-indexed, ascending, gap-free daily, finite, non-negative."""
    if not isinstance(history, pd.Series):
        raise TypeError("history must be a pandas Series of daily sales indexed by date.")
    if not isinstance(history.index, pd.DatetimeIndex):
        raise ValueError("history must be indexed by date (a pd.DatetimeIndex).")
    if len(history) == 0:
        raise ValueError("history is empty.")
    if history.index.has_duplicates or not history.index.is_monotonic_increasing:
        raise ValueError("history must be sorted ascending by date with no duplicated dates.")
    if not history.index.equals(pd.date_range(history.index[0], periods=len(history), freq="D")):
        raise ValueError("history must be a continuous daily series (no missing calendar days).")
    values = history.to_numpy(dtype=float)
    if not np.all(np.isfinite(values)):
        raise ValueError("history contains missing or non-finite values.")
    if np.any(values < 0):
        raise ValueError("history contains negative sales; unit sales cannot be negative.")
    return history


def future_dates(history_end: pd.Timestamp, horizon: int) -> pd.DatetimeIndex:
    """The `horizon` calendar days AFTER `history_end`: generated from the date alone, never from sales."""
    if not _is_int(horizon) or horizon < 1:
        raise ValueError(f"horizon must be a positive integer (got {horizon!r}).")
    return pd.date_range(pd.Timestamp(history_end) + pd.Timedelta(days=1), periods=int(horizon), freq="D")


def regressor_frame(index: pd.DatetimeIndex, calendar: pd.DataFrame, regressors: Sequence[str]) -> pd.DataFrame:
    """0/1 regressor columns for `index`, read from the calendar (calendar FACTS, never sales).

    Raises if the calendar does not cover every requested date: a missing date is never filled
    with zeros, because "no event" and "unknown" are different things.
    """
    index = pd.DatetimeIndex(index)
    if not isinstance(calendar, pd.DataFrame) or not isinstance(calendar.index, pd.DatetimeIndex):
        raise ValueError("calendar must be a DataFrame indexed by date (see load_event_calendar).")
    absent = [r for r in regressors if r not in calendar.columns]
    if absent:
        raise ValueError(f"calendar is missing regressor column(s): {absent}.")
    uncovered = index.difference(calendar.index)
    if len(uncovered):
        raise ValueError(
            f"The event calendar does not cover {len(uncovered)} requested date(s) "
            f"(first: {uncovered[0].date()}). Missing dates are never filled with zeros."
        )
    frame = calendar.loc[index, list(regressors)].astype(float)
    if not np.all(np.isfinite(frame.to_numpy())) or not np.all((frame.to_numpy() == 0) | (frame.to_numpy() == 1)):
        raise ValueError("calendar regressors must be 0/1 flags with no missing values.")
    return frame.reset_index(drop=True)


def build_training_frame(history: pd.Series, calendar: Optional[pd.DataFrame], regressors: Sequence[str]) -> pd.DataFrame:
    """Prophet's training table: ds, y and the regressor columns for the HISTORY days only."""
    check_history(history)
    frame = pd.DataFrame({"ds": history.index.to_numpy(), "y": history.to_numpy(dtype=float)})
    if regressors:
        if calendar is None:
            raise ValueError("An event calendar is required when regressors are used.")
        frame = pd.concat([frame, regressor_frame(history.index, calendar, regressors)], axis=1)
    return frame


def build_future_frame(history_end: pd.Timestamp, horizon: int, calendar: Optional[pd.DataFrame], regressors: Sequence[str]) -> pd.DataFrame:
    """Prophet's prediction table: ds (the `horizon` days after the history) and the regressor columns."""
    days = future_dates(history_end, horizon)
    frame = pd.DataFrame({"ds": days.to_numpy()})
    if regressors:
        if calendar is None:
            raise ValueError("An event calendar is required when regressors are used.")
        frame = pd.concat([frame, regressor_frame(days, calendar, regressors)], axis=1)
    return frame


# =============================================================================================
# Prophet plumbing (lazy)
# =============================================================================================

def _require_prophet():
    """Import and return the `Prophet` class, or raise an ImportError that says how to install it."""
    try:
        from prophet import Prophet
    except ImportError as exc:
        raise ImportError(
            "Prophet forecasting needs the 'prophet' package, which is not installed in this Python environment. "
            f"Install it with:  {PROPHET_INSTALL_COMMAND}  (or everything Phase 3 needs with:  {PHASE3_REQUIREMENTS_COMMAND}). "
            "No other method is substituted for Prophet."
        ) from exc
    return Prophet


@contextmanager
def _quiet_stan():
    """Silence cmdstanpy/prophet INFO logging (it prints per-fit progress lines) and restore it afterwards."""
    loggers = [logging.getLogger(name) for name in ("cmdstanpy", "prophet")]
    previous = [lg.disabled for lg in loggers]
    for lg in loggers:
        lg.disabled = True
    try:
        yield
    finally:
        for lg, state in zip(loggers, previous):
            lg.disabled = state


@contextmanager
def _seeded_numpy(seed: int):
    """Seed NumPy's global generator (Prophet's interval simulation uses it) and restore the caller's state."""
    state = np.random.get_state()
    np.random.seed(int(seed))
    try:
        yield
    finally:
        np.random.set_state(state)


_BACKEND_HINTS = ("cmdstan", "stan", "optimiz", "executable", "winerror", "permission", "no such file", "access is denied")


def _explain_backend_failure(exc: BaseException) -> ProphetBackendError:
    text = f"{type(exc).__name__}: {exc}"
    hint = ""
    if any(token in text.lower() for token in _BACKEND_HINTS):
        hint = (
            " If this happens on Windows, check (1) that the project and its .venv are in a folder WITHOUT spaces or non-ASCII characters in "
            "the path (Prophet's bundled Stan executable can fail to start from such a path: move or clone the project to e.g. C:\\riverdemand "
            "and recreate the .venv there), and (2) that antivirus software is not blocking the executable inside .venv\\Lib\\site-packages\\prophet\\stan_model. "
            f"Reinstall with:  python -m pip install --force-reinstall prophet."
        )
    return ProphetBackendError(f"Prophet's Stan backend failed ({text}).{hint} No other method is substituted for Prophet.")


# =============================================================================================
# The forecaster
# =============================================================================================

class ProphetForecaster(BaseForecaster):
    """Prophet on the aggregate series with optional published-in-advance calendar regressors.

    Parameters
    ----------
    variant    a `ProphetVariant` (name, seasonality mode, whether regressors are used)
    settings   validated `ProphetSettings` (use `from_config` to read config.yaml)
    calendar   date-indexed 0/1 event-flag table (`load_event_calendar`); required when the variant
               uses regressors, and it must cover the training dates AND the forecast dates.

    `fit(history)` estimates the model by MAP and immediately builds the `settings.horizon`-day forecast
    (point forecast, interval bounds, all clipped at zero); `predict(h)` returns its first `h` days.
    """

    def __init__(self, variant: ProphetVariant, settings: ProphetSettings, calendar: Optional[pd.DataFrame] = None) -> None:
        if not isinstance(variant, ProphetVariant):
            raise TypeError("variant must be a ProphetVariant (use ProphetForecaster.from_config).")
        if not isinstance(settings, ProphetSettings):
            raise TypeError("settings must be a ProphetSettings (use ProphetForecaster.from_config).")
        if variant.use_regressors and calendar is None:
            raise ValueError(
                f"{variant.name} uses calendar regressors and needs the event calendar "
                "(pass calendar=load_event_calendar(...)); use the 'no-regressors' variant for a history-only model."
            )
        self.variant = variant
        self.settings = settings
        self.calendar = calendar if variant.use_regressors else None
        self.regressors: Tuple[str, ...] = settings.regressors if variant.use_regressors else ()
        self.name = variant.name
        self._reset()

    @classmethod
    def from_config(
        cls, variant_name: str, config: Optional[Mapping[str, Any]] = None, calendar: Optional[pd.DataFrame] = None
    ) -> "ProphetForecaster":
        """Build the variant called `variant_name` from config.yaml (`prophet` section + backtest horizon)."""
        config = load_config() if config is None else config
        variants = {v.name: v for v in variants_from_config(config)}
        if variant_name not in variants:
            raise ValueError(f"Unknown Prophet variant {variant_name!r}; declared variants: {list(variants)}.")
        return cls(variants[variant_name], ProphetSettings.from_config(config), calendar=calendar)

    # ------------------------------------------------------------------ state
    def _reset(self) -> None:
        self._model = None
        self._history_end: Optional[pd.Timestamp] = None
        self._frame: Optional[pd.DataFrame] = None            # Prophet forecast frame for the future days
        self._diagnostics: Dict[str, Any] = {}

    @property
    def is_fitted(self) -> bool:
        return self._model is not None

    def _require_fitted(self) -> None:
        if self._model is None:
            raise RuntimeError(f"{self.name} has not been fitted: call fit(history) first.")

    # ------------------------------------------------------------------ fit
    def _new_prophet(self, Prophet):
        s = self.settings
        model = Prophet(
            growth="linear",
            n_changepoints=s.n_changepoints,
            changepoint_range=s.changepoint_range,
            yearly_seasonality=s.yearly_seasonality,
            weekly_seasonality=s.weekly_seasonality,
            daily_seasonality=s.daily_seasonality,
            seasonality_mode=self.variant.seasonality_mode,
            seasonality_prior_scale=s.seasonality_prior_scale,
            changepoint_prior_scale=s.changepoint_prior_scale,
            mcmc_samples=s.mcmc_samples,
            interval_width=s.interval_width,
            uncertainty_samples=s.uncertainty_samples,
        )
        for name in self.regressors:
            model.add_regressor(name)
        return model

    def _predict_frame(self, model, history_end: pd.Timestamp, horizon: int) -> pd.DataFrame:
        future = build_future_frame(history_end, horizon, self.calendar, self.regressors)
        with _seeded_numpy(self.settings.random_seed), _quiet_stan():
            return model.predict(future)

    def fit(self, history: pd.Series) -> "ProphetForecaster":
        """Estimate the model on `history` (the data available at the forecast origin) and forecast ahead."""
        Prophet = _require_prophet()
        check_history(history)
        if len(history) < MIN_HISTORY_DAYS:
            raise ValueError(f"{self.name}: a history of {len(history)} days is too short; Prophet needs at least {MIN_HISTORY_DAYS}.")
        self._reset()
        started = time.perf_counter()
        history_end = history.index[-1]
        training = build_training_frame(history, self.calendar, self.regressors)
        # Both frames are built BEFORE fitting so a calendar gap fails fast, before any Stan work.
        build_future_frame(history_end, self.settings.horizon, self.calendar, self.regressors)

        model = self._new_prophet(Prophet)
        try:
            with _quiet_stan():
                model.fit(training, seed=self.settings.random_seed)
        except Exception as exc:  # noqa: BLE001 -- turned into an explanatory, non-substituting error
            raise _explain_backend_failure(exc) from exc

        frame = self._predict_frame(model, history_end, self.settings.horizon)
        yhat = frame["yhat"].to_numpy(dtype=float)
        lower = frame["yhat_lower"].to_numpy(dtype=float)
        upper = frame["yhat_upper"].to_numpy(dtype=float)
        if not (np.all(np.isfinite(yhat)) and np.all(np.isfinite(lower)) and np.all(np.isfinite(upper))):
            raise ProphetBackendError(f"{self.name} produced non-finite forecasts; the fit is not usable. No other method is substituted.")
        n_clipped = int(np.sum(yhat < 0) + np.sum(lower < 0) + np.sum(upper < 0))
        frame = frame.assign(yhat=np.clip(yhat, 0.0, None), yhat_lower=np.clip(lower, 0.0, None), yhat_upper=np.clip(upper, 0.0, None))

        stan_fit = getattr(getattr(model, "stan_backend", None), "stan_fit", None)
        converged = getattr(stan_fit, "converged", None)
        self._model, self._history_end, self._frame = model, history_end, frame
        self._diagnostics = {
            "model": self.name,
            "seasonality_mode": self.variant.seasonality_mode,
            "uses_regressors": self.variant.use_regressors,
            "regressors": list(self.regressors),
            "n_obs": int(len(history)),
            "history_start": str(history.index[0].date()),
            "history_end": str(history_end.date()),
            "horizon": self.settings.horizon,
            "n_changepoints_requested": self.settings.n_changepoints,
            "n_changepoints_fitted": int(len(model.changepoints)),
            "changepoint_range": self.settings.changepoint_range,
            "changepoint_prior_scale": self.settings.changepoint_prior_scale,
            "seasonality_prior_scale": self.settings.seasonality_prior_scale,
            "yearly_seasonality": self.settings.yearly_seasonality,
            "weekly_seasonality": self.settings.weekly_seasonality,
            "daily_seasonality": self.settings.daily_seasonality,
            "interval_width": self.settings.interval_width,
            "uncertainty_samples": self.settings.uncertainty_samples,
            "estimation": "MAP (cmdstanpy optimize)",
            "optimizer_converged": None if converged is None else bool(converged),
            "log_posterior": _log_posterior(stan_fit),
            "n_negative_values_clipped": n_clipped,
            "random_seed": self.settings.random_seed,
            "fit_seconds": float(time.perf_counter() - started),
            "prophet_version": _package_version("prophet"),
            "cmdstanpy_version": _package_version("cmdstanpy"),
        }
        logger.info("[%s] fitted on %d days ending %s in %.1fs (converged=%s)", self.name, len(history), history_end.date(),
                    self._diagnostics["fit_seconds"], self._diagnostics["optimizer_converged"])
        return self

    # ------------------------------------------------------------------ predict
    def _check_horizon(self, horizon: int) -> int:
        if not _is_int(horizon) or horizon < 1:
            raise ValueError(f"horizon must be a positive integer (got {horizon!r}).")
        return int(horizon)

    def _frame_for(self, horizon: int) -> pd.DataFrame:
        self._require_fitted()
        horizon = self._check_horizon(horizon)
        if horizon <= len(self._frame):
            return self._frame.iloc[:horizon]
        longer = self._predict_frame(self._model, self._history_end, horizon)   # raises if the calendar does not reach that far
        for column in ("yhat", "yhat_lower", "yhat_upper"):
            longer[column] = np.clip(longer[column].to_numpy(dtype=float), 0.0, None)
        return longer

    def predict(self, horizon: int) -> np.ndarray:
        """Point forecast (zero-clipped) for the `horizon` days after the fitted history."""
        return self._frame_for(horizon)["yhat"].to_numpy(dtype=float).copy()

    def predict_interval(self, horizon: int) -> Tuple[np.ndarray, np.ndarray]:
        """(lower, upper) bounds of the `interval_width` uncertainty interval, clipped at zero."""
        frame = self._frame_for(horizon)
        return frame["yhat_lower"].to_numpy(dtype=float).copy(), frame["yhat_upper"].to_numpy(dtype=float).copy()

    def forecast_index(self, horizon: Optional[int] = None) -> pd.DatetimeIndex:
        """The calendar days that `predict(horizon)` refers to."""
        self._require_fitted()
        return future_dates(self._history_end, self.settings.horizon if horizon is None else self._check_horizon(horizon))

    def predict_frame(self, dates: pd.DatetimeIndex) -> pd.DataFrame:
        """Prophet's full component frame (trend, weekly, yearly, regressors, yhat) for arbitrary `dates`.

        Used for the components plot on days inside the fitted history. Regressor values come from
        the calendar; nothing here reads sales. The frame is NOT clipped (it shows the raw components).
        """
        self._require_fitted()
        dates = pd.DatetimeIndex(dates)
        frame = pd.DataFrame({"ds": dates.to_numpy()})
        if self.regressors:
            frame = pd.concat([frame, regressor_frame(dates, self.calendar, self.regressors)], axis=1)
        with _seeded_numpy(self.settings.random_seed), _quiet_stan():
            return self._model.predict(frame)

    # ------------------------------------------------------------------ reporting
    def fit_diagnostics(self) -> dict:
        """Facts about the most recent fit (JSON-friendly): settings, changepoints, convergence, seconds, versions."""
        return dict(self._diagnostics)


def _package_version(name: str) -> Optional[str]:
    from importlib import metadata

    try:
        return metadata.version(name)
    except metadata.PackageNotFoundError:
        return None


def _log_posterior(stan_fit: Any) -> Optional[float]:
    try:
        return float(stan_fit.optimized_params_dict["lp__"])
    except Exception:  # noqa: BLE001 -- purely informational
        return None
