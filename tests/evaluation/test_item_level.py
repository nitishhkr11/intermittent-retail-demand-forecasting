"""Unit tests for src/evaluation/item_level.py (Phase 2 Step 7 orchestration).

Small, fully deterministic synthetic multi-item data only -- not the
real 20-item/1941-day project data (that's what the notebook-execution
test exercises). Series are built from fixed repeating blocks (never
`np.random`), so there is zero chance of a flaky test hitting
Croston's all-zero-training-window edge case on some random seed. A
shorter horizon/n_folds than the real project's (28, 5) keeps these
fast; the fold-generation logic itself
(src.evaluation.backtesting.generate_rolling_origins) is already
covered by tests/evaluation/test_backtesting.py and is not re-derived
here -- these tests are about the per-item orchestration layer on top
of it.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.evaluation.backtesting import run_backtest
from src.evaluation.item_level import (
    category_macro_average_summary,
    fold_results_to_long_table,
    item_development_series,
    per_item_summary,
    run_item_level_backtests,
)
from src.models.base import BaseForecaster
from src.models.baseline import SeasonalNaive
from src.models.croston import CrostonForecaster
from src.models.tsb import TSBForecaster

# Deterministic repeating blocks -- no randomness anywhere in this file.
INTERMITTENT_BLOCK = [0, 0, 3, 0, 0, 2, 0, 0, 0, 4]  # 3 occurrences / 10 days
SMOOTH_BLOCK = [1, 2, 1, 3, 2, 1, 2, 3, 1, 2]           # no zeros at all


def _pattern(n_days: int, block) -> np.ndarray:
    reps = int(np.ceil(n_days / len(block)))
    return np.asarray((list(block) * reps)[:n_days], dtype=float)


def _daily_sales(item_specs, n_days=150, start="2020-01-01") -> pd.DataFrame:
    """Build a selected_items_daily_sales-shaped frame: {item_id: (values, demand_class)}."""
    dates = pd.date_range(start, periods=n_days, freq="D")
    frames = []
    for item_id, (values, demand_class) in item_specs.items():
        assert len(values) == n_days
        frames.append(pd.DataFrame({
            "item_id": item_id, "date": dates, "sales": values, "demand_class": demand_class,
        }))
    return pd.concat(frames, ignore_index=True)


# ---- item_development_series: correct fold boundaries / no holdout leakage --------

def test_item_development_series_excludes_the_final_holdout():
    daily_sales = _daily_sales({"A": (_pattern(150, INTERMITTENT_BLOCK), "Intermittent")}, n_days=150)
    series = item_development_series(daily_sales, "A", holdout_days=10)
    assert len(series) == 140
    assert series.index.max() == pd.Timestamp("2020-01-01") + pd.Timedelta(days=139)
    assert series.index.min() == pd.Timestamp("2020-01-01")


def test_item_development_series_raises_for_unknown_item():
    daily_sales = _daily_sales({"A": (_pattern(150, INTERMITTENT_BLOCK), "Intermittent")}, n_days=150)
    with pytest.raises(ValueError):
        item_development_series(daily_sales, "DOES_NOT_EXIST", holdout_days=10)


# ---- fold_results_to_long_table ----------------------------------------------------

def test_fold_results_to_long_table_shape_and_columns():
    series = pd.Series(_pattern(150, SMOOTH_BLOCK), index=pd.date_range("2020-01-01", periods=150, freq="D"))
    results = run_backtest(series, lambda: SeasonalNaive(season_length=7), horizon=10, n_folds=3)
    long_table = fold_results_to_long_table(results, item_id="A", demand_class="Smooth")
    assert len(long_table) == 3 * 10  # n_folds x horizon
    for col in ["item_id", "demand_class", "model", "fold", "date", "actual", "predicted", "success"]:
        assert col in long_table.columns
    assert (long_table["item_id"] == "A").all()
    assert (long_table["demand_class"] == "Smooth").all()
    assert set(long_table["fold"]) == {1, 2, 3}


def test_fold_results_to_long_table_marks_failed_folds_with_nan_predicted():
    class _AlwaysFails(BaseForecaster):
        name = "AlwaysFails"

        def fit(self, history):
            raise RuntimeError("deliberate failure for testing")

        def predict(self, horizon):  # pragma: no cover -- never reached
            raise RuntimeError("unreachable")

    series = pd.Series(_pattern(150, INTERMITTENT_BLOCK), index=pd.date_range("2020-01-01", periods=150, freq="D"))
    results = run_backtest(series, _AlwaysFails, horizon=10, n_folds=3)
    long_table = fold_results_to_long_table(results, item_id="B", demand_class="Lumpy")
    assert len(long_table) == 3 * 10
    assert long_table["predicted"].isna().all()
    assert (~long_table["success"]).all()
    assert long_table["actual"].notna().all()  # actuals still populated for failed folds


# ---- run_item_level_backtests: multi-item orchestration, holdout boundary ---------

def test_run_item_level_backtests_produces_expected_shapes():
    item_specs = {
        "A": (_pattern(150, INTERMITTENT_BLOCK), "Intermittent"),
        "B": (_pattern(150, SMOOTH_BLOCK), "Smooth"),
    }
    daily_sales = _daily_sales(item_specs, n_days=150)
    model_factories = {
        "SeasonalNaive": lambda: SeasonalNaive(season_length=7),
        "Croston": lambda: CrostonForecaster(alpha=0.1),
        "TSB": lambda: TSBForecaster(alpha=0.1, beta=0.1),
    }
    out = run_item_level_backtests(
        daily_sales, ["A", "B"], model_factories, horizon=10, n_folds=3, holdout_days=10
    )
    long_table, fold_summary = out["long"], out["fold_summary"]

    assert len(long_table) == 2 * 3 * 3 * 10   # items x models x folds x horizon
    assert len(fold_summary) == 2 * 3 * 3        # items x models x folds
    assert set(fold_summary["item_id"]) == {"A", "B"}
    # "model" is each forecaster's own .name (e.g. "Croston(alpha=0.1)"),
    # the same convention every other model in this project uses -- not
    # the model_factories dict key.
    expected_names = {factory().name for factory in model_factories.values()}
    assert set(fold_summary["model"]) == expected_names
    assert (fold_summary["success"]).all()  # deterministic non-degenerate series -- nothing should fail
    # demand_class correctly attached per item, not mixed up
    assert set(fold_summary.loc[fold_summary["item_id"] == "A", "demand_class"]) == {"Intermittent"}
    assert set(fold_summary.loc[fold_summary["item_id"] == "B", "demand_class"]) == {"Smooth"}


def test_run_item_level_backtests_never_sees_the_final_holdout():
    # The final 10 days are an unmistakable spike -- if the holdout ever
    # leaked into a training or test window, it would show up either as
    # a date >= the holdout boundary, or as the spike value itself.
    values = _pattern(150, INTERMITTENT_BLOCK)
    values[-10:] = 9999.0
    daily_sales = _daily_sales({"A": (values, "Erratic")}, n_days=150)
    model_factories = {"SeasonalNaive": lambda: SeasonalNaive(season_length=7)}

    out = run_item_level_backtests(
        daily_sales, ["A"], model_factories, horizon=10, n_folds=3, holdout_days=10
    )
    holdout_start = pd.Timestamp("2020-01-01") + pd.Timedelta(days=140)
    assert pd.to_datetime(out["long"]["date"]).max() < holdout_start
    assert pd.to_datetime(out["fold_summary"]["forecast_end"]).max() < holdout_start
    assert (out["long"]["actual"] != 9999.0).all()  # the spike itself never appears


def test_run_item_level_backtests_uses_identical_fold_boundaries_across_models():
    # Every model for a given item must see the exact same fold
    # boundaries -- the Step 7 brief's "same development window, five
    # expanding folds" applied per item, not re-derived per model.
    daily_sales = _daily_sales({"A": (_pattern(150, INTERMITTENT_BLOCK), "Lumpy")}, n_days=150)
    model_factories = {
        "SeasonalNaive": lambda: SeasonalNaive(season_length=7),
        "Croston": lambda: CrostonForecaster(alpha=0.1),
    }
    out = run_item_level_backtests(
        daily_sales, ["A"], model_factories, horizon=10, n_folds=3, holdout_days=10
    )
    by_model = out["fold_summary"].groupby("model")
    sn_name = model_factories["SeasonalNaive"]().name
    cr_name = model_factories["Croston"]().name
    sn_ranges = by_model.get_group(sn_name)[["forecast_start", "forecast_end"]].reset_index(drop=True)
    cr_ranges = by_model.get_group(cr_name)[["forecast_start", "forecast_end"]].reset_index(drop=True)
    assert sn_ranges.equals(cr_ranges)


# ---- per_item_summary: correct metric calculations ---------------------------------

def test_per_item_summary_hand_computed_mean():
    fold_summary = pd.DataFrame({
        "item_id": ["A", "A", "B"],
        "demand_class": ["Smooth", "Smooth", "Lumpy"],
        "model": ["Croston", "Croston", "Croston"],
        "MAE": [2.0, 4.0, 10.0],
        "RMSE": [3.0, 5.0, 12.0],
        "MAPE": [20.0, 30.0, None],
        "success": [True, True, True],
    })
    summary = per_item_summary(fold_summary).set_index("item_id")
    assert summary.loc["A", "MAE_mean"] == pytest.approx(3.0)    # (2+4)/2
    assert summary.loc["A", "RMSE_mean"] == pytest.approx(4.0)   # (3+5)/2
    assert summary.loc["A", "MAPE_mean"] == pytest.approx(25.0)  # (20+30)/2
    assert summary.loc["A", "MAPE_n_folds_valid"] == 2
    assert summary.loc["B", "MAPE_n_folds_valid"] == 0            # MAPE undefined that fold
    assert pd.isna(summary.loc["B", "MAPE_mean"])


def test_per_item_summary_excludes_failed_folds_from_the_mean():
    fold_summary = pd.DataFrame({
        "item_id": ["A", "A"],
        "demand_class": ["Smooth", "Smooth"],
        "model": ["Croston", "Croston"],
        "MAE": [2.0, None],
        "RMSE": [3.0, None],
        "MAPE": [20.0, None],
        "success": [True, False],
    })
    summary = per_item_summary(fold_summary).set_index("item_id")
    assert summary.loc["A", "n_folds"] == 2
    assert summary.loc["A", "n_success"] == 1
    assert summary.loc["A", "MAE_mean"] == pytest.approx(2.0)  # only the successful fold


def test_per_item_summary_raises_on_missing_columns():
    with pytest.raises(ValueError):
        per_item_summary(pd.DataFrame({"item_id": ["A"]}))


# ---- category_macro_average_summary: unweighted-across-items, not volume-weighted -

def test_category_macro_average_is_unweighted_across_items():
    # Item A has a far larger MAE than item B; macro-averaging must just
    # average the two items' own means equally (50/50) rather than being
    # dominated by either one's sales volume.
    item_summary = pd.DataFrame({
        "item_id": ["A", "B"],
        "demand_class": ["Smooth", "Smooth"],
        "model": ["Croston", "Croston"],
        "MAE_mean": [100.0, 0.0],
        "RMSE_mean": [100.0, 0.0],
        "MAPE_mean": [50.0, 10.0],
    })
    cat_summary = category_macro_average_summary(item_summary).set_index("demand_class")
    assert cat_summary.loc["Smooth", "MAE_macro_mean"] == pytest.approx(50.0)
    assert cat_summary.loc["Smooth", "n_items"] == 2


def test_category_macro_average_handles_multiple_categories_and_models():
    item_summary = pd.DataFrame({
        "item_id": ["A", "B", "C"],
        "demand_class": ["Smooth", "Smooth", "Lumpy"],
        "model": ["Croston", "Croston", "Croston"],
        "MAE_mean": [10.0, 20.0, 5.0],
        "RMSE_mean": [10.0, 20.0, 5.0],
        "MAPE_mean": [10.0, 20.0, 5.0],
    })
    cat_summary = category_macro_average_summary(item_summary).set_index("demand_class")
    assert cat_summary.loc["Smooth", "MAE_macro_mean"] == pytest.approx(15.0)
    assert cat_summary.loc["Lumpy", "MAE_macro_mean"] == pytest.approx(5.0)
    assert cat_summary.loc["Lumpy", "n_items"] == 1


def test_category_macro_average_raises_on_missing_columns():
    with pytest.raises(ValueError):
        category_macro_average_summary(pd.DataFrame({"item_id": ["A"]}))
