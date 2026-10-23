"""XGBoost forecasters for item-level retail demand (Phase 3, Step 1).

Two thin wrappers share one core implementation:

  PooledXGBoostForecaster
      ONE global `XGBRegressor` fit on the training windows of all items
      at once (item identity and a demand-class label are features).
      This is the primary design: ~1,800 daily rows per item is little
      for a tree model, and half of the 20 selected items (the
      Intermittent and Lumpy classes) are sparse, so pooling lets items
      share weekly/calendar structure instead of each learning it from
      scratch. Its `fit()` takes a mapping
      `{item_id: date-indexed sales Series}` -- the same per-item Series
      `src.evaluation.item_level.item_development_series` already builds
      -- and `predict()` returns `{item_id: np.ndarray}`.

  XGBoostForecaster (a `src.models.base.BaseForecaster`)
      The same features and hyper-parameters fit on a single item's
      Series, so it runs through the existing, unmodified
      `src.evaluation.backtesting.run_backtest` like every Phase 2 model.
      It exists as the empirical check on the pooling decision, not as
      the primary model.

Features, and when each is known (every feature of the row for target
date t is available at the forecast origin, which is the last training
date T < t):

  lag_1, lag_7, lag_14, lag_28       sales at t-1, t-7, t-14, t-28. Beyond T
                                      these are the model's OWN earlier
                                      predictions (recursive forecasting),
                                      never unseen actual sales.
  roll_mean_7 / _14 / _28             mean of sales over t-w .. t-1. The
                                      window ends at t-1, so a row's features
                                      can never contain its own target-day
                                      sales (the series is shifted before it
                                      is averaged; see
                                      `lag_roll_feature_matrix`).
  dayofweek, month, dayofmonth        functions of the date alone.
  snap_ca, event_cultural,            M5 calendar facts. They are published
  event_national, event_religious,    in advance and are identical for every
  event_sporting, is_christmas        item on a date. Only the calendar
                                      columns are read from the panel file
                                      (never `sales`), and only up to the
                                      development end date.
  item_id                             static identity (pooled model only).
  demand_class                        Syntetos-Boylan-Croston class computed
                                      FROM THE FOLD'S TRAINING WINDOW ONLY
                                      (pooled model only), via the existing
                                      `classify_series`. The Phase 1
                                      `demand_class` label in
                                      selected_items_daily_sales.parquet was
                                      computed on all 1,941 days -- including
                                      every fold's forecast window and the
                                      final holdout -- so using it as a
                                      feature would leak future demand
                                      behaviour (project limitation L7). It
                                      is therefore used only to GROUP
                                      results, exactly as in Phase 2 Step 7.

Deliberately NOT used in this first baseline: `sell_price`. A price at a
date inside the forecast horizon is only legitimate if it is a planned,
known-in-advance covariate, which the dataset cannot verify, and an
absent weekly price in M5 also encodes "item not on sale" (a structural
zero). It is left for a later step with an explicit availability
assumption.

Training rows. A row needs 28 days of history (`cfg.max_lookback`), so
the first 28 rows of every item are EXCLUDED from the training matrix
rather than filled in; their sales still serve as history for later rows.
No lag or rolling value is ever imputed, and `fit_diagnostics()` reports
how many rows were dropped.

Recursive forecasting. For each of the `horizon` days after the last
training date, features are built from the history (actual up to T, then
predictions), the model predicts all items for that day, predictions are
clipped to be >= 0, and the clipped values are appended to the history
before the next day is predicted. `predict()` receives nothing but the
fitted state, so no actual sales from the forecast window can reach it.
Recursion makes forecasts smoother than real sales (a prediction is an
expected value, not a draw) and lets errors compound; this is a known
cost of the approach, reported in the Phase 3 Step 1 notes.

Non-negative demand. Every prediction is clipped at 0 -- inside the
recursion (so a negative value is never fed back as a lag) and in the
returned forecast. The rule is applied identically for every objective.
A NaN or infinite prediction is never clipped or replaced: it raises, so
the backtest records a failed fold instead of a fabricated number.

Reproducibility. Hyper-parameters come from config.yaml (`xgboost:`),
`random_state` is fixed, and `n_jobs=1` keeps the histogram algorithm's
floating-point summation order independent of the machine's core count.
No hyper-parameter search, early stopping or feature selection is done
in this step.
"""

from __future__ import annotations

import time
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Mapping, Optional, Tuple, Union

import numpy as np
import pandas as pd

from src.analysis.demand_classification import (
    INSUFFICIENT_DATA_LABEL,
    NO_DEMAND_LABEL,
    STANDARD_CATEGORIES,
    classify_series,
)
from src.models.base import BaseForecaster
from src.utils.logging import get_logger

logger = get_logger(__name__)

#: All demand-class levels the categorical feature can take. Fixed up front
#: so the categorical dtype is identical at fit and predict time.
DEMAND_CLASS_LEVELS: Tuple[str, ...] = (*STANDARD_CATEGORIES, NO_DEMAND_LABEL, INSUFFICIENT_DATA_LABEL)

CALENDAR_BASE_COLUMNS: Tuple[str, ...] = ("dayofweek", "month", "dayofmonth")
EVENT_FLAG_COLUMNS: Tuple[str, ...] = (
    "snap_ca",
    "event_cultural",
    "event_national",
    "event_religious",
    "event_sporting",
    "is_christmas",
)

# Calendar columns read from data/interim/panel_ca1_foods3.parquet. `sales` is
# intentionally absent: the calendar loader must never touch target values.
_EVENT_PANEL_COLUMNS: List[str] = ["date", "event_name_1", "event_type_1", "event_name_2", "event_type_2", "snap_CA"]
_EVENT_TYPES: Tuple[str, ...] = ("Cultural", "National", "Religious", "Sporting")
_CLOSURE_EVENT_NAME = "Christmas"  # M5 CA_1 stores report zero sales on every 25 December

_ALLOWED_HYPERPARAMETERS = {
    "objective", "n_estimators", "learning_rate", "max_depth", "min_child_weight",
    "subsample", "colsample_bytree", "reg_lambda", "random_state", "n_jobs", "tree_method",
}


# --------------------------------------------------------------------------- configuration

@dataclass(frozen=True)
class XGBoostConfig:
    """Feature definition and fixed hyper-parameters. Defaults match config.yaml's `xgboost:` section."""

    lags: Tuple[int, ...] = (1, 7, 14, 28)
    rolling_mean_windows: Tuple[int, ...] = (7, 14, 28)
    use_event_calendar: bool = True
    objective: str = "reg:squarederror"
    n_estimators: int = 300
    learning_rate: float = 0.05
    max_depth: int = 6
    min_child_weight: float = 1.0
    subsample: float = 0.8
    colsample_bytree: float = 0.8
    reg_lambda: float = 1.0
    random_state: int = 42
    n_jobs: int = 1
    tree_method: str = "hist"

    def __post_init__(self) -> None:
        for field_name in ("lags", "rolling_mean_windows"):
            values = tuple(int(v) for v in getattr(self, field_name))
            if not values:
                raise ValueError(f"{field_name} must not be empty.")
            if any(v < 1 for v in values) or len(set(values)) != len(values):
                raise ValueError(f"{field_name} must be unique positive integers, got {values}.")
            object.__setattr__(self, field_name, tuple(sorted(values)))
        if self.n_estimators < 1:
            raise ValueError(f"n_estimators must be >= 1, got {self.n_estimators}.")
        if not (self.learning_rate > 0):
            raise ValueError(f"learning_rate must be > 0, got {self.learning_rate}.")
        if self.max_depth < 1:
            raise ValueError(f"max_depth must be >= 1, got {self.max_depth}.")
        if self.n_jobs < 1:
            raise ValueError(f"n_jobs must be >= 1, got {self.n_jobs}.")

    @property
    def max_lookback(self) -> int:
        """History length a row needs: the largest lag or rolling window."""
        return max(max(self.lags), max(self.rolling_mean_windows))

    @property
    def lag_roll_names(self) -> List[str]:
        return [f"lag_{k}" for k in self.lags] + [f"roll_mean_{w}" for w in self.rolling_mean_windows]

    def numeric_feature_names(self) -> List[str]:
        names = self.lag_roll_names + list(CALENDAR_BASE_COLUMNS)
        if self.use_event_calendar:
            names += list(EVENT_FLAG_COLUMNS)
        return names

    def feature_names(self, pooled: bool) -> List[str]:
        """Ordered model-input columns. The pooled model adds item identity and demand class."""
        return self.numeric_feature_names() + (["item_id", "demand_class"] if pooled else [])

    def regressor_params(self) -> Dict[str, Any]:
        return {
            "objective": self.objective,
            "n_estimators": self.n_estimators,
            "learning_rate": self.learning_rate,
            "max_depth": self.max_depth,
            "min_child_weight": self.min_child_weight,
            "subsample": self.subsample,
            "colsample_bytree": self.colsample_bytree,
            "reg_lambda": self.reg_lambda,
            "random_state": self.random_state,
            "n_jobs": self.n_jobs,
            "tree_method": self.tree_method,
        }

    @classmethod
    def from_config(cls, config: Mapping[str, Any]) -> "XGBoostConfig":
        """Build from the loaded config.yaml dict (its `xgboost:` section).

        Unknown hyper-parameter keys raise instead of being ignored, so a
        typo in config.yaml cannot silently fall back to a default.
        """
        if "xgboost" not in config:
            raise KeyError("config has no 'xgboost' section.")
        section = config["xgboost"]
        unknown_keys = set(section) - {"lags", "rolling_mean_windows", "use_event_calendar", "hyperparameters"}
        if unknown_keys:
            raise ValueError(f"Unknown xgboost config key(s): {sorted(unknown_keys)}.")
        params = dict(section.get("hyperparameters", {}))
        unknown = set(params) - _ALLOWED_HYPERPARAMETERS
        if unknown:
            raise ValueError(f"Unknown xgboost.hyperparameters key(s): {sorted(unknown)}.")
        kwargs: Dict[str, Any] = dict(params)
        if "lags" in section:
            kwargs["lags"] = tuple(section["lags"])
        if "rolling_mean_windows" in section:
            kwargs["rolling_mean_windows"] = tuple(section["rolling_mean_windows"])
        if "use_event_calendar" in section:
            kwargs["use_event_calendar"] = bool(section["use_event_calendar"])
        return cls(**kwargs)


# --------------------------------------------------------------------------- calendar features

def build_event_calendar(raw: pd.DataFrame) -> pd.DataFrame:
    """Turn M5 calendar columns into a date-indexed table of 0/1 flags (`EVENT_FLAG_COLUMNS`).

    `raw` needs `date, event_name_1, event_type_1, event_name_2, event_type_2,
    snap_CA`; the panel repeats those values once per item, so identical rows
    collapse to one row per date. Two different rows for the same date mean
    the calendar is inconsistent and raise.
    """
    missing = set(_EVENT_PANEL_COLUMNS) - set(raw.columns)
    if missing:
        raise ValueError(f"raw calendar is missing column(s): {sorted(missing)}.")
    cal = raw[_EVENT_PANEL_COLUMNS].drop_duplicates()
    if cal["date"].duplicated().any():
        raise ValueError("Calendar columns are inconsistent: the same date appears with different values.")
    cal = cal.sort_values("date").reset_index(drop=True)

    out = pd.DataFrame(index=pd.DatetimeIndex(pd.to_datetime(cal["date"]), name="date"))
    out["snap_ca"] = cal["snap_CA"].to_numpy(dtype=int)
    for event_type in _EVENT_TYPES:
        flag = cal["event_type_1"].eq(event_type) | cal["event_type_2"].eq(event_type)
        out[f"event_{event_type.lower()}"] = flag.fillna(False).astype(bool).astype(int).to_numpy()
    closure = cal["event_name_1"].eq(_CLOSURE_EVENT_NAME) | cal["event_name_2"].eq(_CLOSURE_EVENT_NAME)
    out["is_christmas"] = closure.fillna(False).astype(bool).astype(int).to_numpy()
    return out[list(EVENT_FLAG_COLUMNS)]


def load_event_calendar(
    panel_path: Union[str, Path], last_date: Optional[Union[str, pd.Timestamp]] = None
) -> pd.DataFrame:
    """Read the M5 calendar facts from the panel parquet, restricted to `last_date` and earlier.

    Only the calendar columns are read -- `sales` is never loaded. Pass the last
    DEVELOPMENT date as `last_date`: the date filter is applied while reading the
    parquet file, so rows after it (the final holdout days) are never loaded into
    memory during development experiments.
    """
    path = Path(panel_path)
    if not path.exists():
        raise FileNotFoundError(f"Panel file '{path}' not found; it supplies the calendar/event columns.")
    filters = None if last_date is None else [("date", "<=", pd.Timestamp(last_date))]
    raw = pd.read_parquet(path, columns=_EVENT_PANEL_COLUMNS, filters=filters)
    calendar = build_event_calendar(raw)
    if last_date is not None and len(calendar) and calendar.index.max() > pd.Timestamp(last_date):
        raise AssertionError("Calendar read returned dates after last_date; the read-time filter failed.")
    return calendar


def calendar_features(
    index: pd.DatetimeIndex, event_calendar: Optional[pd.DataFrame], use_events: bool = True
) -> pd.DataFrame:
    """Calendar columns for `index`: day-of-week/month/day-of-month, plus event flags if `use_events`.

    Raises if `use_events` is set and the event calendar does not cover every
    requested date -- a missing date is never filled with zeros.
    """
    out = pd.DataFrame(
        {
            "dayofweek": np.asarray(index.dayofweek, dtype=int),
            "month": np.asarray(index.month, dtype=int),
            "dayofmonth": np.asarray(index.day, dtype=int),
        },
        index=index,
    )
    if not use_events:
        return out
    if event_calendar is None:
        raise ValueError(
            "An event calendar is required when use_event_calendar=True "
            "(see load_event_calendar); pass use_event_calendar=False for calendar-only features."
        )
    uncovered = index.difference(event_calendar.index)
    if len(uncovered):
        raise ValueError(
            f"Event calendar does not cover {len(uncovered)} requested date(s); first: {uncovered[0].date()}."
        )
    return pd.concat([out, event_calendar.loc[index, list(EVENT_FLAG_COLUMNS)]], axis=1)


# --------------------------------------------------------------------------- lag / rolling features

def lag_roll_feature_matrix(
    values: np.ndarray, lags: Tuple[int, ...], windows: Tuple[int, ...]
) -> np.ndarray:
    """Vectorised lag and rolling-mean features for ONE item, shape (T, len(lags)+len(windows)).

    Row t holds sales at t-k for each lag k, then the mean of sales over
    t-w .. t-1 for each window w. A window always ENDS at t-1, which is the
    same as shifting the series by one day before taking a rolling mean, so
    `values[t]` never appears in row t. Entries without a full history are
    NaN (never imputed); the caller drops those warm-up rows explicitly.
    """
    y = np.asarray(values, dtype=float)
    n = len(y)
    columns = []
    for k in lags:
        col = np.full(n, np.nan)
        if n > k:
            col[k:] = y[: n - k]
        columns.append(col)
    for w in windows:
        col = np.full(n, np.nan)
        if n > w:
            col[w:] = np.lib.stride_tricks.sliding_window_view(y, w)[: n - w].mean(axis=1)
        columns.append(col)
    return np.column_stack(columns)


def lag_roll_features_for_next_step(
    history: np.ndarray, lags: Tuple[int, ...], windows: Tuple[int, ...]
) -> np.ndarray:
    """The same features for the NEXT day, from `history` of shape (n_items, L), newest value last.

    Used by recursive forecasting; `lag_roll_feature_matrix` and this function
    must agree wherever both are defined (covered by a parity test).
    """
    h = np.asarray(history, dtype=float)
    if h.ndim != 2:
        raise ValueError(f"history must be 2-D (n_items, L), got shape {h.shape}.")
    need = max(max(lags), max(windows))
    if h.shape[1] < need:
        raise ValueError(f"history has {h.shape[1]} columns but {need} are needed.")
    columns = [h[:, -k] for k in lags] + [h[:, -w:].mean(axis=1) for w in windows]
    return np.column_stack(columns)


# --------------------------------------------------------------------------- history validation / training frame

def _validate_history(item_id: str, series: pd.Series) -> pd.Series:
    if not isinstance(series, pd.Series):
        raise TypeError(f"History for {item_id!r} must be a pandas Series, got {type(series).__name__}.")
    if not isinstance(series.index, pd.DatetimeIndex):
        raise ValueError(f"History for {item_id!r} must be indexed by date (a pd.DatetimeIndex).")
    if len(series) == 0:
        raise ValueError(f"History for {item_id!r} is empty.")
    index = series.index
    if not index.is_monotonic_increasing or index.has_duplicates:
        raise ValueError(f"History for {item_id!r} must be sorted ascending with no duplicate dates.")
    if not index.equals(pd.date_range(index[0], periods=len(index), freq="D")):
        raise ValueError(
            f"History for {item_id!r} must be a continuous daily series (positional lags need no gaps)."
        )
    values = series.to_numpy(dtype=float)
    if np.any(np.isnan(values)):
        raise ValueError(f"History for {item_id!r} contains NaN value(s).")
    if np.any(values < 0):
        raise ValueError(f"History for {item_id!r} contains negative value(s); sales cannot be negative.")
    return series


def fold_local_demand_classes(histories: Mapping[str, pd.Series]) -> pd.DataFrame:
    """ADI / CV^2 / class per item from exactly the given (training) histories.

    Thin wrapper over the existing `classify_series`; nothing outside
    `histories` is read.
    """
    rows = []
    for item_id in sorted(histories):
        s = histories[item_id]
        result = classify_series(pd.Series(s.to_numpy(dtype=float)))
        rows.append(
            {
                "item_id": item_id,
                "ADI": result["ADI"],
                "CV2": result["CV2"],
                "demand_class": result["demand_class"],
                "n_train_obs": int(len(s)),
            }
        )
    return pd.DataFrame(rows)


def build_training_frame(
    histories: Mapping[str, pd.Series],
    cfg: XGBoostConfig,
    event_calendar: Optional[pd.DataFrame] = None,
) -> Tuple[pd.DataFrame, int]:
    """Supervised rows (one per item and target date) from the training histories only.

    Returns `(frame, n_warmup_rows_dropped)`. `frame` has `item_id`, `date`, `y`
    (that day's sales) and the numeric feature columns. The first
    `cfg.max_lookback` days of every item are excluded because they lack a full
    history; they remain available as history for later rows. No value is
    imputed, and the finished frame is checked to contain no NaN.
    """
    warmup = cfg.max_lookback
    frames: List[pd.DataFrame] = []
    n_dropped = 0
    for item_id in sorted(histories):
        series = _validate_history(item_id, histories[item_id])
        values = series.to_numpy(dtype=float)
        if len(values) <= warmup:
            raise ValueError(
                f"History for {item_id!r} has {len(values)} observation(s); more than {warmup} are needed."
            )
        keep = np.arange(len(values)) >= warmup
        n_dropped += int((~keep).sum())
        features = pd.DataFrame(
            lag_roll_feature_matrix(values, cfg.lags, cfg.rolling_mean_windows)[keep], columns=cfg.lag_roll_names
        )
        calendar = calendar_features(series.index[keep], event_calendar, cfg.use_event_calendar).reset_index(drop=True)
        frame = pd.concat([features, calendar], axis=1)
        frame.insert(0, "y", values[keep])
        frame.insert(0, "date", series.index[keep])
        frame.insert(0, "item_id", item_id)
        frames.append(frame)
    out = pd.concat(frames, ignore_index=True)
    if out[cfg.numeric_feature_names()].isna().any().any():
        raise AssertionError("Training frame still contains NaN features after excluding warm-up rows.")
    return out, n_dropped


# --------------------------------------------------------------------------- regressor construction

def _xgboost_version() -> Optional[str]:
    try:
        import xgboost

        return str(xgboost.__version__)
    except ImportError:
        return None


def _default_regressor(cfg: XGBoostConfig):
    try:
        from xgboost import XGBRegressor
    except ImportError as exc:  # pragma: no cover - exercised only without the dependency
        raise ImportError(
            "xgboost is required for XGBoost forecasting. Install it with `pip install -r requirements.txt`."
        ) from exc
    return XGBRegressor(enable_categorical=True, verbosity=0, **cfg.regressor_params())


def _default_name(kind: str, cfg: XGBoostConfig) -> str:
    if kind == "per-item":
        return "XGBoost(per-item)"
    if cfg.objective == "reg:squarederror":
        return "XGBoost(pooled)"
    return f"XGBoost(pooled,{cfg.objective.split(':')[-1]})"


# --------------------------------------------------------------------------- shared core

class _XGBoostCore:
    """Feature building, fitting and recursive prediction shared by both public forecasters."""

    def __init__(
        self,
        cfg: XGBoostConfig,
        event_calendar: Optional[pd.DataFrame],
        regressor_factory: Optional[Callable[[], Any]],
        pooled: bool,
        model_name: str,
    ):
        self.cfg = cfg
        self.event_calendar = event_calendar
        self.pooled = pooled
        self.model_name = model_name
        self._regressor_factory = regressor_factory or (lambda: _default_regressor(cfg))
        self._regressor: Any = None
        self._items: List[str] = []
        self._tail: Optional[np.ndarray] = None
        self._train_end: Optional[pd.Timestamp] = None
        self._item_class_table: Optional[pd.DataFrame] = None
        self._numeric_dtypes: Optional[pd.Series] = None
        self._diagnostics: Dict[str, Any] = {}

    @property
    def feature_names(self) -> List[str]:
        return self.cfg.feature_names(self.pooled)

    def _design_matrix(
        self, numeric: pd.DataFrame, item_ids: List[str], classes: List[str]
    ) -> pd.DataFrame:
        X = numeric[self.cfg.numeric_feature_names()].copy()
        if self.pooled:
            X["item_id"] = pd.Categorical(item_ids, categories=self._items)
            X["demand_class"] = pd.Categorical(classes, categories=list(DEMAND_CLASS_LEVELS))
        return X

    def fit(self, histories: Mapping[str, pd.Series]) -> None:
        started = time.perf_counter()
        if len(histories) == 0:
            raise ValueError("fit() needs at least one item history.")
        items = sorted(histories)
        validated = {item: _validate_history(item, histories[item]) for item in items}
        ends = {s.index[-1] for s in validated.values()}
        if len(ends) != 1:
            raise ValueError("All item histories must end on the same date (one forecast origin).")
        train_end = next(iter(ends))

        frame, n_warmup = build_training_frame(validated, self.cfg, self.event_calendar)

        class_table = fold_local_demand_classes(validated)
        class_by_item = dict(zip(class_table["item_id"], class_table["demand_class"]))
        self._items = items
        X = self._design_matrix(
            frame, frame["item_id"].tolist(), [class_by_item[i] for i in frame["item_id"]]
        )
        y = frame["y"].to_numpy(dtype=float)

        regressor = self._regressor_factory()
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            regressor.fit(X, y)

        L = self.cfg.max_lookback
        self._regressor = regressor
        self._train_end = train_end
        self._tail = np.vstack([validated[i].to_numpy(dtype=float)[-L:] for i in items])
        self._item_class_table = class_table
        self._numeric_dtypes = frame[self.cfg.numeric_feature_names()].dtypes
        messages = sorted({str(w.message) for w in caught})
        self._diagnostics = {
            "model": self.model_name,
            "pooled": self.pooled,
            "n_items": len(items),
            "n_train_rows": int(len(frame)),
            "n_warmup_rows_dropped": int(n_warmup),
            "n_features": len(self.feature_names),
            "features": ",".join(self.feature_names),
            "train_start": str(min(s.index[0] for s in validated.values()).date()),
            "train_end": str(train_end.date()),
            "objective": self.cfg.objective,
            "n_estimators": self.cfg.n_estimators,
            "learning_rate": self.cfg.learning_rate,
            "max_depth": self.cfg.max_depth,
            "random_state": self.cfg.random_state,
            "n_jobs": self.cfg.n_jobs,
            "xgboost_version": _xgboost_version(),
            "fit_seconds": round(time.perf_counter() - started, 3),
            "n_warnings": len(caught),
            "warnings": "; ".join(messages[:3]),
        }

    def _require_fitted(self) -> None:
        if self._regressor is None:
            raise RuntimeError(f"{self.model_name}.predict() called before fit().")

    def forecast_dates(self, horizon: int) -> pd.DatetimeIndex:
        self._require_fitted()
        if horizon < 1:
            raise ValueError(f"horizon must be positive, got {horizon}.")
        return pd.date_range(self._train_end + pd.Timedelta(days=1), periods=horizon, freq="D")

    def predict(self, horizon: int) -> Dict[str, np.ndarray]:
        self._require_fitted()
        dates = self.forecast_dates(horizon)
        calendar = calendar_features(dates, self.event_calendar, self.cfg.use_event_calendar)
        calendar_values = calendar[list(CALENDAR_BASE_COLUMNS) + (list(EVENT_FLAG_COLUMNS) if self.cfg.use_event_calendar else [])]

        items = self._items
        n_items = len(items)
        class_by_item = dict(zip(self._item_class_table["item_id"], self._item_class_table["demand_class"]))
        classes = [class_by_item[i] for i in items]
        lr_names = self.cfg.lag_roll_names

        history = self._tail.copy()  # actual sales up to the forecast origin, then predictions only
        predictions = np.empty((n_items, horizon), dtype=float)
        raw_all = np.empty((n_items, horizon), dtype=float)
        for step in range(horizon):
            lag_roll = lag_roll_features_for_next_step(history, self.cfg.lags, self.cfg.rolling_mean_windows)
            numeric = pd.DataFrame(lag_roll, columns=lr_names)
            for column in calendar_values.columns:
                numeric[column] = np.repeat(calendar_values[column].iloc[step], n_items)
            numeric = numeric.astype(self._numeric_dtypes.to_dict())
            X = self._design_matrix(numeric, list(items), classes)

            raw = np.asarray(self._regressor.predict(X), dtype=float)
            if raw.shape != (n_items,):
                raise ValueError(f"Regressor returned shape {raw.shape}; expected {(n_items,)}.")
            if not np.all(np.isfinite(raw)):
                raise ValueError(
                    f"{self.model_name} produced non-finite prediction(s) at horizon step {step + 1} "
                    f"({dates[step].date()})."
                )
            clipped = np.clip(raw, 0.0, None)  # non-negative demand, applied before the value is reused as a lag
            raw_all[:, step] = raw
            predictions[:, step] = clipped
            history = np.concatenate([history[:, 1:], clipped[:, None]], axis=1)
        # Reported (not acted on) so the effect of the non-negativity rule is visible in the saved diagnostics.
        self._diagnostics.update(
            {
                "n_predictions": int(predictions.size),
                "n_predictions_clipped_at_zero": int((raw_all < 0).sum()),
                "min_raw_prediction": float(raw_all.min()),
                "max_prediction": float(predictions.max()),
            }
        )
        return {item: predictions[i] for i, item in enumerate(items)}

    def feature_importance(self, importance_type: str = "gain") -> pd.Series:
        self._require_fitted()
        get_booster = getattr(self._regressor, "get_booster", None)
        if get_booster is None:
            raise RuntimeError("The fitted regressor has no booster to read importances from.")
        scores = get_booster().get_score(importance_type=importance_type)
        return pd.Series(scores, dtype=float).reindex(self.feature_names).fillna(0.0)

    def diagnostics(self) -> Dict[str, Any]:
        return dict(self._diagnostics)


# --------------------------------------------------------------------------- public forecasters

class PooledXGBoostForecaster:
    """One global XGBoost model across items. See the module docstring for the design."""

    def __init__(
        self,
        config: Optional[XGBoostConfig] = None,
        event_calendar: Optional[pd.DataFrame] = None,
        regressor_factory: Optional[Callable[[], Any]] = None,
        name: Optional[str] = None,
    ):
        self.config = config or XGBoostConfig()
        self.name = name or _default_name("pooled", self.config)
        self._core = _XGBoostCore(self.config, event_calendar, regressor_factory, pooled=True, model_name=self.name)

    def fit(self, histories: Mapping[str, pd.Series]) -> "PooledXGBoostForecaster":
        """Fit on `{item_id: date-indexed sales Series}`; all Series must end on the same date."""
        self._core.fit(histories)
        return self

    def predict(self, horizon: int) -> Dict[str, np.ndarray]:
        """`horizon` recursive forecasts per item for the days after the last training date."""
        return self._core.predict(horizon)

    def forecast_dates(self, horizon: int) -> pd.DatetimeIndex:
        return self._core.forecast_dates(horizon)

    def fit_diagnostics(self) -> Dict[str, Any]:
        return self._core.diagnostics()

    def item_class_table(self) -> pd.DataFrame:
        """Fold-local ADI / CV^2 / demand class per item (computed from the training window only)."""
        self._core._require_fitted()
        return self._core._item_class_table.copy()

    def feature_importance(self, importance_type: str = "gain") -> pd.Series:
        return self._core.feature_importance(importance_type)


class XGBoostForecaster(BaseForecaster):
    """The same features and hyper-parameters fit on ONE item's history (the per-item comparison model)."""

    name = "XGBoost(per-item)"

    def __init__(
        self,
        config: Optional[XGBoostConfig] = None,
        event_calendar: Optional[pd.DataFrame] = None,
        regressor_factory: Optional[Callable[[], Any]] = None,
    ):
        self.config = config or XGBoostConfig()
        self._core = _XGBoostCore(
            self.config, event_calendar, regressor_factory, pooled=False, model_name=self.name
        )
        self._item: Optional[str] = None

    def fit(self, history: pd.Series) -> "XGBoostForecaster":
        self._item = str(history.name) if getattr(history, "name", None) is not None else "series"
        self._core.fit({self._item: history})
        return self

    def predict(self, horizon: int) -> np.ndarray:
        return self._core.predict(horizon)[self._item]

    def fit_diagnostics(self) -> Dict[str, Any]:
        return self._core.diagnostics()

    def feature_importance(self, importance_type: str = "gain") -> pd.Series:
        return self._core.feature_importance(importance_type)
