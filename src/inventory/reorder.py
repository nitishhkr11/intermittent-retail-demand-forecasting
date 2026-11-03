"""Reorder point and safety stock from forecasts and their saved forecast errors.

This module turns a demand FORECAST into an inventory DECISION INPUT, and is explicit about the gap between
the two:

* A **forecast** says how much demand to expect (for example, 12 units over the next 7 days).
* A **decision** also needs *how wrong the forecast tends to be* and *how much risk of running out is acceptable*.
  The safety stock is the extra stock that covers the forecast's typical shortfall at a chosen service level.

Method (all of it explicit; nothing is assumed Gaussian or independent):

1. The **protection interval** is the number of days a replenishment decision has to cover:
   `lead time + review period` (days). With continuous review the review period is 0.
2. **Expected demand** over that interval = the sum of the model's forecast for those days.
3. **Cumulative forecast errors** over windows of that length are read from SAVED backtest forecasts: for every run of
   `protection_days` consecutive forecast days inside a fold, `error = sum(actual) - sum(forecast)`. A positive error is a
   shortfall (demand above forecast). Overlapping windows are all kept, so the strong day-to-day dependence of errors is
   preserved instead of being assumed away (the project's residual diagnostics found autocorrelated errors).
4. **Safety stock** = the empirical quantile of those cumulative errors at the target service level, floored at zero.
   No normal-distribution `z * sigma * sqrt(L)` formula is used: errors of intermittent demand are skewed and not independent.
5. **Reorder point** = expected demand over the protection interval + safety stock.

Service level here is a **cycle service level**: the target probability that demand over one protection interval does not
exceed the stock held for it (no stock-out in a replenishment cycle). It is NOT the fill rate (the share of demand units
filled from stock); a 95 % cycle service level can correspond to a very different fill rate.

Everything operational (lead time, review period, service level, stock on hand) is a user-supplied or ILLUSTRATIVE input.
The routines estimate stock levels; they do not estimate costs or savings, and nothing here claims any.

Supported horizon: the saved backtests contain 28-day forecasts, so `lead time + review period` must be at most 28 days.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from src.utils.config import load_config

#: Bounds mirrored from `config.yaml` (`inventory:`); `assumptions_from_config` re-checks the config's own values.
MIN_SERVICE_LEVEL = 0.50
MAX_SERVICE_LEVEL = 0.999
MAX_PROTECTION_DAYS = 28
DEFAULT_MIN_ERROR_WINDOWS = 20

FORECAST_VS_DECISION_TEXT = (
    "A forecast is an estimate of demand; a stocking decision also needs the forecast's typical error and an acceptable "
    "risk of running out. The reorder point below = expected demand over the protection interval (from the forecast) + "
    "safety stock (from the saved forecast errors at the chosen service level)."
)
SERVICE_LEVEL_TEXT = (
    "The service level used here is a cycle service level: the target probability that demand during one replenishment "
    "cycle does not exceed the stock held for it. It is not the fill rate (the share of demand units delivered from stock), "
    "which is a different quantity and cannot be read off a cycle service level."
)
ASSUMPTIONS_TEXT = (
    "Lead time, review period, service level and stock on hand are user-supplied or illustrative inputs, not facts about "
    "any real supplier. No cost, holding or shortage-penalty model is used, so no savings are claimed."
)


# --------------------------------------------------------------------------------------------- validation helpers
def _is_integer(value: Any) -> bool:
    return isinstance(value, (int, np.integer)) and not isinstance(value, (bool, np.bool_))


def _require_int(name: str, value: Any, minimum: int) -> int:
    if not _is_integer(value):
        raise ValueError(f"{name} must be a whole number of days, got {value!r}.")
    if int(value) < minimum:
        raise ValueError(f"{name} must be at least {minimum}, got {int(value)}.")
    return int(value)


def _require_finite_number(name: str, value: Any, minimum: float = 0.0) -> float:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, float, np.integer, np.floating)):
        raise ValueError(f"{name} must be a number, got {value!r}.")
    number = float(value)
    if not np.isfinite(number):
        raise ValueError(f"{name} must be finite, got {value!r}.")
    if number < minimum:
        raise ValueError(f"{name} must be at least {minimum}, got {number}.")
    return number


def _check_service_level(value: Any, minimum: float = MIN_SERVICE_LEVEL, maximum: float = MAX_SERVICE_LEVEL) -> float:
    level = _require_finite_number("service_level", value, minimum=0.0)
    if not (minimum <= level <= maximum):
        raise ValueError(
            f"service_level must be between {minimum} and {maximum} (a cycle service level), got {level}. "
            "100 % would require unbounded stock; below the minimum the empirical quantile is not meaningful."
        )
    return level


def _as_nonneg_vector(name: str, values: Sequence[float]) -> np.ndarray:
    array = np.asarray(values, dtype=float)
    if array.ndim != 1:
        raise ValueError(f"{name} must be a one-dimensional sequence of numbers.")
    if array.size == 0:
        raise ValueError(f"{name} is empty.")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} contains NaN or infinite value(s).")
    return array


def _as_error_vector(error_samples: Sequence[float]) -> np.ndarray:
    return _as_nonneg_vector("error_samples", error_samples)


# --------------------------------------------------------------------------------------------- assumptions & result
@dataclass(frozen=True)
class InventoryAssumptions:
    """The operational inputs of one stocking decision. All are user-supplied or illustrative.

    `lead_time_days` and `review_period_days` are whole days; `service_level` is the target cycle service level;
    `on_hand` / `on_order` (units) are optional and only used to compute the inventory position and a suggested order.
    """

    lead_time_days: int
    review_period_days: int = 0
    service_level: float = 0.95
    on_hand: Optional[float] = None
    on_order: float = 0.0

    def __post_init__(self) -> None:
        lead = _require_int("lead_time_days", self.lead_time_days, 1)
        review = _require_int("review_period_days", self.review_period_days, 0)
        _check_service_level(self.service_level)
        if self.on_hand is not None:
            _require_finite_number("on_hand", self.on_hand, 0.0)
        _require_finite_number("on_order", self.on_order, 0.0)
        if lead + review > MAX_PROTECTION_DAYS:
            raise ValueError(
                f"lead_time_days + review_period_days = {lead + review} exceeds the {MAX_PROTECTION_DAYS}-day horizon "
                "of the saved backtest forecasts; longer protection intervals are not supported."
            )

    @property
    def protection_days(self) -> int:
        """Days one replenishment decision must cover: lead time + review period."""
        return int(self.lead_time_days) + int(self.review_period_days)


@dataclass(frozen=True)
class ReorderPointResult:
    """Everything `compute_reorder_point` decided, in units of the forecast (for example, units of one item)."""

    protection_days: int
    expected_demand: float
    safety_stock: float
    reorder_point: float
    service_level: float
    error_quantile_level: float
    n_error_samples: int
    order_up_to_level: Optional[float] = None
    inventory_position: Optional[float] = None
    suggested_order_quantity: Optional[float] = None
    warnings: Tuple[str, ...] = field(default_factory=tuple)

    def to_dict(self) -> Dict[str, Any]:
        record = asdict(self)
        record["warnings"] = list(self.warnings)
        return record


# --------------------------------------------------------------------------------------------- building blocks
def protection_interval_days(lead_time_days: int, review_period_days: int = 0) -> int:
    """`lead time + review period`, validated (whole days, lead time >= 1, total <= 28)."""
    lead = _require_int("lead_time_days", lead_time_days, 1)
    review = _require_int("review_period_days", review_period_days, 0)
    total = lead + review
    if total > MAX_PROTECTION_DAYS:
        raise ValueError(
            f"lead_time_days + review_period_days = {total} exceeds the {MAX_PROTECTION_DAYS}-day horizon of the saved forecasts."
        )
    return total


def expected_demand_over_interval(forecast: Sequence[float], days: int) -> float:
    """Sum of the first `days` forecast values. Never extrapolates: a shorter forecast is an error."""
    days = _require_int("days", days, 1)
    values = _as_nonneg_vector("forecast", forecast)
    if values.size < days:
        raise ValueError(f"forecast has {values.size} day(s) but the protection interval needs {days}.")
    window = values[:days]
    if np.any(window < 0):
        raise ValueError("forecast contains negative demand; unit demand cannot be negative.")
    return float(window.sum())


def safety_stock_from_errors(error_samples: Sequence[float], service_level: float) -> float:
    """Empirical quantile of the cumulative forecast errors at `service_level`, floored at zero.

    Uses the "higher" empirical quantile, so at least `service_level` of the samples are covered (the conservative
    reading with few samples). A positive error means demand exceeded the forecast.
    """
    level = _check_service_level(service_level)
    errors = _as_error_vector(error_samples)
    return float(max(0.0, np.quantile(errors, level, method="higher")))


def _validated_error_table(
    table: pd.DataFrame, model: Optional[str], item_id: Optional[str]
) -> pd.DataFrame:
    required = {"model", "fold", "date", "actual", "predicted"}
    missing = required - set(table.columns)
    if missing:
        raise ValueError(f"forecast table is missing column(s): {sorted(missing)}")
    subset = table
    if model is not None:
        subset = subset[subset["model"] == model]
    elif subset["model"].nunique() > 1:
        raise ValueError(f"The table holds several models ({sorted(subset['model'].unique())[:4]}...); pass model=...")
    if "item_id" in subset.columns:
        if item_id is not None:
            subset = subset[subset["item_id"] == item_id]
        elif subset["item_id"].nunique() > 1:
            raise ValueError(
                "The table holds several items; pass item_id=... (errors of different items are on different scales "
                "and are never pooled here)."
            )
    elif item_id is not None:
        raise ValueError("item_id was given but the table has no item_id column (aggregate table).")
    if subset.empty:
        raise ValueError(f"No rows left for model={model!r}, item_id={item_id!r}.")
    return subset


def _fold_windows(group: pd.DataFrame, window_days: int) -> np.ndarray:
    """Cumulative errors (actual - predicted) over every run of `window_days` consecutive days of ONE fold."""
    ordered = group.sort_values("date")
    dates = pd.to_datetime(ordered["date"])
    if len(dates) > 1 and not (dates.diff().dropna() == pd.Timedelta(days=1)).all():
        raise ValueError("Forecast days of a fold are not consecutive calendar days; cannot build cumulative errors.")
    if len(ordered) < window_days:
        raise ValueError(f"A fold has {len(ordered)} day(s), fewer than the {window_days}-day window.")
    errors = ordered["actual"].to_numpy(dtype=float) - ordered["predicted"].to_numpy(dtype=float)
    cumulative = np.concatenate(([0.0], np.cumsum(errors)))
    return cumulative[window_days:] - cumulative[:-window_days]


def cumulative_error_samples(
    table: pd.DataFrame,
    window_days: int,
    *,
    model: Optional[str] = None,
    item_id: Optional[str] = None,
    folds: Optional[Sequence[int]] = None,
) -> np.ndarray:
    """Cumulative forecast errors over every `window_days`-day run inside each selected fold.

    `table` is a saved item-level or aggregate forecast table. For each fold the error of a window is
    `sum(actual) - sum(predicted)` (positive = demand above forecast). All overlapping windows are kept, so
    a 28-day fold yields `28 - window_days + 1` samples that are strongly dependent; their number is NOT an
    independent sample size. Folds with a failed or missing forecast are skipped, never filled.
    """
    window_days = _require_int("window_days", window_days, 1)
    if window_days > MAX_PROTECTION_DAYS:
        raise ValueError(f"window_days must be at most {MAX_PROTECTION_DAYS}, got {window_days}.")
    subset = _validated_error_table(table, model, item_id)
    if folds is not None:
        subset = subset[subset["fold"].isin(list(folds))]
    samples: List[np.ndarray] = []
    for _, group in subset.groupby("fold", sort=True):
        usable = bool(group["success"].astype(bool).all()) if "success" in group.columns else True
        if not usable or not np.all(np.isfinite(group["predicted"].to_numpy(dtype=float))):
            continue
        samples.append(_fold_windows(group, window_days))
    return np.concatenate(samples) if samples else np.empty(0, dtype=float)


def compute_reorder_point(
    forecast: Sequence[float],
    error_samples: Sequence[float],
    assumptions: InventoryAssumptions,
    *,
    min_error_windows_warning: int = DEFAULT_MIN_ERROR_WINDOWS,
) -> ReorderPointResult:
    """Reorder point = expected demand over the protection interval + empirical-quantile safety stock.

    `forecast` must start on the first day the replenishment has to cover and be at least
    `assumptions.protection_days` long. `error_samples` are cumulative errors over windows of the SAME length
    (see `cumulative_error_samples`). With a periodic review (`review_period_days > 0`) the same number is also
    reported as the order-up-to level. If `on_hand` is given, the inventory position (`on_hand + on_order`)
    and the quantity that lifts it back to the target are computed; lot sizes and minimum order quantities are not modelled.
    """
    if not isinstance(assumptions, InventoryAssumptions):
        raise ValueError("assumptions must be an InventoryAssumptions instance.")
    protection = assumptions.protection_days
    expected = expected_demand_over_interval(forecast, protection)
    errors = _as_error_vector(error_samples)
    safety = safety_stock_from_errors(errors, assumptions.service_level)
    reorder_point = max(0.0, expected + safety)

    notes: List[str] = []
    if errors.size < min_error_windows_warning:
        notes.append(
            f"Only {errors.size} cumulative-error sample(s) (fewer than {min_error_windows_warning}); the safety stock is a "
            "rough estimate. Overlapping windows are dependent, so the effective sample size is smaller still."
        )
    if expected == 0.0:
        notes.append("The forecast is zero over the whole protection interval; the reorder point comes from the saved errors alone.")
    if safety == 0.0:
        notes.append("Safety stock is zero: at this service level the saved errors never required stock above the forecast.")

    review = assumptions.review_period_days
    order_up_to = reorder_point if review > 0 else None
    position = suggestion = None
    if assumptions.on_hand is not None:
        position = float(assumptions.on_hand) + float(assumptions.on_order)
        target = order_up_to if order_up_to is not None else reorder_point
        suggestion = max(0.0, target - position)
    return ReorderPointResult(
        protection_days=protection,
        expected_demand=expected,
        safety_stock=safety,
        reorder_point=reorder_point,
        service_level=float(assumptions.service_level),
        error_quantile_level=float(assumptions.service_level),
        n_error_samples=int(errors.size),
        order_up_to_level=order_up_to,
        inventory_position=position,
        suggested_order_quantity=suggestion,
        warnings=tuple(notes),
    )


# --------------------------------------------------------------------------------------------- uncertainty
def _resample(rng: np.random.Generator, values: np.ndarray, block_size: int) -> np.ndarray:
    n = values.size
    if block_size <= 1 or n <= 1:
        return values[rng.integers(0, n, size=n)]
    n_blocks = int(np.ceil(n / block_size))
    starts = rng.integers(0, n, size=n_blocks)
    positions = (starts[:, None] + np.arange(block_size)[None, :]) % n  # circular blocks keep short-range dependence
    return values[positions.ravel()[:n]]


def bootstrap_safety_stock_interval(
    error_samples: Sequence[float],
    service_level: float,
    *,
    n_resamples: int = 1000,
    seed: int = 42,
    confidence: float = 0.90,
    block_size: int = 1,
) -> Tuple[float, float]:
    """Percentile interval of the safety stock under resampling of the saved errors (seeded, reproducible).

    `block_size` > 1 resamples circular blocks, which keeps the dependence of overlapping windows; pass the window
    length. It uses its own random generator, so the global NumPy random state is never touched. A short error
    history makes this interval wide; it describes sampling noise only, not model error.
    """
    errors = _as_error_vector(error_samples)
    level = _check_service_level(service_level)
    n_resamples = _require_int("n_resamples", n_resamples, 1)
    block_size = _require_int("block_size", block_size, 1)
    if not (0.0 < confidence < 1.0):
        raise ValueError(f"confidence must be strictly between 0 and 1, got {confidence}.")
    rng = np.random.default_rng(seed)
    stocks = np.empty(n_resamples)
    for i in range(n_resamples):
        stocks[i] = max(0.0, float(np.quantile(_resample(rng, errors, block_size), level, method="higher")))
    tail = (1.0 - confidence) / 2.0
    low, high = np.quantile(stocks, [tail, 1.0 - tail])
    return float(low), float(high)


# --------------------------------------------------------------------------------------------- backtest of the rule
def service_level_backtest(
    table: pd.DataFrame,
    window_days: int,
    service_levels: Sequence[float] = (0.80, 0.90, 0.95),
    *,
    model: Optional[str] = None,
    item_ids: Optional[Sequence[str]] = None,
    min_calibration_folds: int = 2,
) -> pd.DataFrame:
    """Did the safety-stock rule deliver its target service level on folds it was NOT calibrated on?

    For each item and each test fold `k` that has at least `min_calibration_folds` EARLIER folds, the safety stock is
    the empirical quantile of cumulative errors from folds `< k` only. It is then applied to every `window_days`-day
    window inside fold `k`: a window is a stock-out when `sum(actual) > sum(forecast for those days) + safety stock`.
    The window's expected demand is the saved forecast for those very days (the forecast issued at the fold origin).
    One row per (item, test fold, target level); `achieved_cycle_service` = share of windows without a stock-out.
    Cells that cannot be evaluated are kept with a `status` explaining why.
    """
    window_days = _require_int("window_days", window_days, 1)
    min_calibration_folds = _require_int("min_calibration_folds", min_calibration_folds, 1)
    levels = [_check_service_level(level) for level in service_levels]
    required = {"model", "fold", "date", "actual", "predicted"}
    missing = required - set(table.columns)
    if missing:
        raise ValueError(f"forecast table is missing column(s): {sorted(missing)}")
    subset = table
    if model is not None:
        subset = subset[subset["model"] == model]
    elif subset["model"].nunique() > 1:
        raise ValueError("The table holds several models; pass model=...")
    if subset.empty:
        raise ValueError(f"No rows for model={model!r}.")
    model_name = model if model is not None else str(subset["model"].iloc[0])
    has_items = "item_id" in subset.columns
    items = [None] if not has_items else (list(item_ids) if item_ids is not None else sorted(subset["item_id"].unique()))

    rows: List[Dict[str, Any]] = []
    for item in items:
        part = subset if item is None else subset[subset["item_id"] == item]
        folds = sorted(part["fold"].unique())
        for fold in folds:
            base = {"model": model_name, "item_id": item, "fold": int(fold), "window_days": window_days}
            earlier = [f for f in folds if f < fold]
            if len(earlier) < min_calibration_folds:
                for level in levels:
                    rows.append({**base, "target_cycle_service": level, "status": f"fewer than {min_calibration_folds} earlier folds"})
                continue
            test = part[part["fold"] == fold]
            usable = bool(test["success"].astype(bool).all()) if "success" in test.columns else True
            if not usable or not np.all(np.isfinite(test["predicted"].to_numpy(dtype=float))):
                for level in levels:
                    rows.append({**base, "target_cycle_service": level, "status": "test fold has failed or missing forecasts"})
                continue
            calibration = cumulative_error_samples(part, window_days, model=model, item_id=item, folds=earlier)
            if calibration.size == 0:
                for level in levels:
                    rows.append({**base, "target_cycle_service": level, "status": "no usable calibration folds"})
                continue
            test_errors = _fold_windows(test, window_days)
            for level in levels:
                safety = safety_stock_from_errors(calibration, level)
                stockouts = int((test_errors > safety).sum())
                rows.append(
                    {
                        **base,
                        "target_cycle_service": level,
                        "status": "ok",
                        "n_calibration_samples": int(calibration.size),
                        "safety_stock": safety,
                        "n_test_windows": int(test_errors.size),
                        "n_stockout_windows": stockouts,
                        "achieved_cycle_service": 1.0 - stockouts / test_errors.size,
                    }
                )
    return pd.DataFrame(rows)


def summarize_service_level_backtest(
    results: pd.DataFrame,
    *,
    n_resamples: int = 1000,
    seed: int = 42,
    confidence: float = 0.90,
    min_cells_warning: int = 20,
) -> pd.DataFrame:
    """Pool `service_level_backtest` cells per target level; the interval comes from a cluster bootstrap.

    Cells (one item in one test fold) are the resampling units, because windows inside a cell overlap and depend on
    each other. `achieved_cycle_service` pools all windows; `gap` = achieved - target (negative = fell short).
    """
    ok = results[results["status"] == "ok"]
    rows: List[Dict[str, Any]] = []
    rng = np.random.default_rng(seed)
    tail = (1.0 - confidence) / 2.0
    for level, group in ok.groupby("target_cycle_service"):
        windows = group["n_test_windows"].to_numpy(dtype=float)
        stockouts = group["n_stockout_windows"].to_numpy(dtype=float)
        achieved = 1.0 - stockouts.sum() / windows.sum()
        n = len(group)
        draws = np.empty(n_resamples)
        for i in range(n_resamples):
            pick = rng.integers(0, n, size=n)
            draws[i] = 1.0 - stockouts[pick].sum() / windows[pick].sum()
        low, high = np.quantile(draws, [tail, 1.0 - tail])
        note = "" if n >= min_cells_warning else f"only {n} (item, fold) cell(s); wide uncertainty, read as indicative"
        rows.append(
            {
                "target_cycle_service": float(level),
                "n_cells": int(n),
                "n_windows": int(windows.sum()),
                "achieved_cycle_service": float(achieved),
                "ci_low": float(low),
                "ci_high": float(high),
                "gap": float(achieved - level),
                "note": note,
            }
        )
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------------------------- demonstration helper
def reorder_point_at_origin(
    table: pd.DataFrame,
    *,
    model: str,
    item_id: Optional[str],
    fold: int,
    assumptions: InventoryAssumptions,
    min_error_windows_warning: int = DEFAULT_MIN_ERROR_WINDOWS,
) -> Tuple[ReorderPointResult, Dict[str, Any]]:
    """A reorder point as it would have been computed at the START of a past fold (no live forecasts).

    The forecast is the saved forecast of that fold; the error samples come from EARLIER folds only. The realised
    demand over the protection interval is returned in the context for comparison only; it was not known at the order time.
    """
    folds = sorted(int(f) for f in table["fold"].unique())
    if fold not in folds:
        raise ValueError(f"fold {fold} is not in the table (available folds: {folds}).")
    earlier = [f for f in folds if f < fold]
    if not earlier:
        raise ValueError(f"fold {fold} has no earlier fold to calibrate the safety stock on; choose a later fold.")
    subset = _validated_error_table(table, model, item_id)
    current = subset[subset["fold"] == fold].sort_values("date")
    if current.empty:
        raise ValueError(f"No rows for model={model!r}, item_id={item_id!r}, fold={fold}.")
    if "success" in current.columns and not current["success"].astype(bool).all():
        raise ValueError(f"The saved forecast of fold {fold} failed for this model/item; nothing to compute.")
    forecast = current["predicted"].to_numpy(dtype=float)
    errors = cumulative_error_samples(subset, assumptions.protection_days, model=model, item_id=item_id, folds=earlier)
    if errors.size == 0:
        raise ValueError("No usable earlier folds to calibrate the safety stock on.")
    result = compute_reorder_point(forecast, errors, assumptions, min_error_windows_warning=min_error_windows_warning)
    days = assumptions.protection_days
    context = {
        "fold": int(fold),
        "forecast_start": pd.Timestamp(current["date"].iloc[0]).date().isoformat(),
        "forecast_end": pd.Timestamp(current["date"].iloc[-1]).date().isoformat(),
        "calibration_folds": earlier,
        "forecast_values": [float(v) for v in forecast],
        "realised_demand_over_protection": float(current["actual"].to_numpy(dtype=float)[:days].sum()),
    }
    return result, context


# --------------------------------------------------------------------------------------------- explanation
def _number(value: float) -> str:
    return f"{value:,.1f}" if abs(value) >= 10 else f"{value:,.2f}"


def explain_result(
    result: ReorderPointResult,
    assumptions: Optional[InventoryAssumptions] = None,
    *,
    item_label: Optional[str] = None,
    unit: str = "units",
) -> str:
    """Plain-English reading of one result, including what it does NOT say."""
    subject = f" for {item_label}" if item_label else ""
    lines = [
        f"Protection interval{subject}: {result.protection_days} day(s)"
        + (
            f" (lead time {assumptions.lead_time_days} + review period {assumptions.review_period_days})."
            if assumptions is not None
            else "."
        ),
        f"Expected demand over it (from the forecast): {_number(result.expected_demand)} {unit}.",
        f"Safety stock (the {result.error_quantile_level:.0%} quantile of {result.n_error_samples} saved cumulative forecast errors, "
        f"floored at 0): {_number(result.safety_stock)} {unit}.",
        f"Reorder point = expected demand + safety stock = {_number(result.reorder_point)} {unit}.",
    ]
    if result.order_up_to_level is not None:
        lines.append(f"With a periodic review the same figure is the order-up-to level: {_number(result.order_up_to_level)} {unit}.")
    if result.inventory_position is not None:
        lines.append(f"Inventory position (on hand + on order): {_number(result.inventory_position)} {unit}.")
        if result.suggested_order_quantity is not None:
            action = (
                f"At least {_number(result.suggested_order_quantity)} {unit} would lift the position back to the target."
                if result.suggested_order_quantity > 0
                else "The position is already at or above the target; no order is needed to protect this interval."
            )
            lines.append(action + " Lot sizes and minimum order quantities are not modelled.")
    lines.extend(["", FORECAST_VS_DECISION_TEXT, SERVICE_LEVEL_TEXT, ASSUMPTIONS_TEXT])
    if result.warnings:
        lines.extend(["", "Warnings:"] + [f"- {w}" for w in result.warnings])
    return "\n".join(lines)


def assumptions_from_config(config: Optional[Mapping[str, Any]] = None, **overrides: Any) -> InventoryAssumptions:
    """Illustrative defaults from `config.yaml` (`inventory:`), optionally overridden field by field.

    The config's own bounds are checked first so that a bad config fails with the offending key named.
    """
    config = load_config() if config is None else config
    section = config.get("inventory")
    if not isinstance(section, Mapping):
        raise ValueError("config.yaml has no 'inventory' section.")
    for key in ("default_lead_time_days", "default_review_period_days", "default_service_level"):
        if key not in section:
            raise ValueError(f"config.yaml inventory section is missing '{key}'.")
    low = float(section.get("min_service_level", MIN_SERVICE_LEVEL))
    high = float(section.get("max_service_level", MAX_SERVICE_LEVEL))
    if not (0.0 < low < high < 1.0):
        raise ValueError(f"config inventory min/max service level must satisfy 0 < min < max < 1, got {low}, {high}.")
    values: Dict[str, Any] = {
        "lead_time_days": section["default_lead_time_days"],
        "review_period_days": section["default_review_period_days"],
        "service_level": section["default_service_level"],
    }
    unknown = set(overrides) - {"lead_time_days", "review_period_days", "service_level", "on_hand", "on_order"}
    if unknown:
        raise ValueError(f"Unknown assumption override(s): {sorted(unknown)}")
    values.update(overrides)
    _check_service_level(values["service_level"], low, high)
    return InventoryAssumptions(**values)
