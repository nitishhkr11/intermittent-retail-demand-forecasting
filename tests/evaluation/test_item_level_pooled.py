"""Unit tests for the Phase 3 Step 1 additions to src/evaluation/item_level.py.

Covers `run_pooled_item_level_backtest` (a walk-forward runner for one model
fit on all items together) and `check_same_evaluation_grid`. The completed
Phase 2 functions in that module are tested, unmodified, in
tests/evaluation/test_item_level.py; this file only adds tests.

Small deterministic synthetic data (fixed repeating blocks, no `np.random`),
three items, 150 days, horizon 10, three folds. Protocol tests use either
the real pooled XGBoost model with a handful of trees or a tiny fake model
whose behaviour is fully controlled, so a protocol violation (wrong dates,
wrong shape, a fold that fails) can be provoked on purpose.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.evaluation.backtesting import generate_rolling_origins
from src.evaluation.item_level import (
    category_macro_average_summary,
    check_same_evaluation_grid,
    item_development_series,
    per_item_summary,
    run_item_level_backtests,
    run_pooled_item_level_backtest,
)
from src.models.baseline import SeasonalNaive
from src.models.xgboost_model import PooledXGBoostForecaster, XGBoostConfig

HORIZON, N_FOLDS, HOLDOUT_DAYS, N_DAYS = 10, 3, 10, 150
START = pd.Timestamp("2020-01-01")
DEV_END = START + pd.Timedelta(days=N_DAYS - HOLDOUT_DAYS - 1)         # last development day
CFG = XGBoostConfig(n_estimators=6, max_depth=3, use_event_calendar=False)

BLOCKS = {
    "A": ([0, 2, 0, 0, 5, 1, 0, 3, 0, 0, 4, 0, 1], "Intermittent"),
    "B": ([3, 4, 2, 5, 3, 4, 6, 3, 2, 4], "Smooth"),
    "C": ([0, 0, 9, 0, 1, 0, 0, 7, 2, 0, 0], "Lumpy"),
}


def _pattern(n: int, block) -> np.ndarray:
    reps = int(np.ceil(n / len(block)))
    return np.asarray((list(block) * reps)[:n], dtype=float)


def _daily_sales(n_days: int = N_DAYS) -> pd.DataFrame:
    dates = pd.date_range(START, periods=n_days, freq="D")
    frames = [
        pd.DataFrame({"item_id": item, "date": dates, "sales": _pattern(n_days, block), "demand_class": label})
        for item, (block, label) in BLOCKS.items()
    ]
    return pd.concat(frames, ignore_index=True)


ITEMS = sorted(BLOCKS)


def _run(daily_sales: pd.DataFrame, factory=None):
    factory = factory or (lambda: PooledXGBoostForecaster(CFG))
    return run_pooled_item_level_backtest(
        daily_sales, ITEMS, factory, horizon=HORIZON, n_folds=N_FOLDS, holdout_days=HOLDOUT_DAYS
    )


class _FakePooled:
    """A pooled 'model' with controllable misbehaviour; records what it was given."""

    name = "FakePooled"

    def __init__(self, fail=False, date_shift=0, length=None, items=None, value=1.0, value_by_item=None):
        self.fail, self.date_shift, self.length, self.items, self.value = fail, date_shift, length, items, value
        self.value_by_item = value_by_item
        self.seen = {}

    def fit(self, histories):
        if self.fail:
            raise RuntimeError("boom")
        self.seen = {item: (s.index[0], s.index[-1], len(s)) for item, s in histories.items()}
        self._items = sorted(histories)
        self._end = max(s.index[-1] for s in histories.values())
        return self

    def predict(self, horizon):
        n = horizon if self.length is None else self.length
        items = self._items if self.items is None else self.items
        values = self.value_by_item or {}
        return {item: np.full(n, values.get(item, self.value)) for item in items}

    def forecast_dates(self, horizon):
        return pd.date_range(self._end + pd.Timedelta(days=1 + self.date_shift), periods=horizon, freq="D")

    def fit_diagnostics(self):
        return {"note": "fake"}


# ---- shape, schema and fold layout ---------------------------------------------------

def test_output_schema_and_sizes_match_the_phase2_item_level_runner():
    ds = _daily_sales()
    pooled = _run(ds)
    baseline = run_item_level_backtests(
        ds, ITEMS, {"SeasonalNaive": lambda: SeasonalNaive(season_length=7)},
        horizon=HORIZON, n_folds=N_FOLDS, holdout_days=HOLDOUT_DAYS,
    )
    assert list(pooled["long"].columns) == list(baseline["long"].columns)
    assert list(pooled["fold_summary"].columns) == list(baseline["fold_summary"].columns)
    assert len(pooled["long"]) == len(ITEMS) * N_FOLDS * HORIZON
    assert len(pooled["fold_summary"]) == len(ITEMS) * N_FOLDS
    assert pooled["fold_summary"]["success"].all()
    assert (pooled["long"]["model"] == "XGBoost(pooled)").all()
    assert not pooled["long"]["predicted"].isna().any()


def test_folds_dates_and_actuals_are_identical_to_the_existing_runner():
    ds = _daily_sales()
    pooled = _run(ds)
    baseline = run_item_level_backtests(
        ds, ITEMS, {"SeasonalNaive": lambda: SeasonalNaive(season_length=7)},
        horizon=HORIZON, n_folds=N_FOLDS, holdout_days=HOLDOUT_DAYS,
    )
    grid = check_same_evaluation_grid(baseline["long"], pooled["long"])  # raises on any difference
    assert grid["n_cells"] == len(ITEMS) * N_FOLDS * HORIZON and grid["max_abs_actual_diff"] == 0.0

    keys = ["item_id", "fold"]
    merged = baseline["fold_summary"].merge(pooled["fold_summary"], on=keys, suffixes=("_base", "_pool"))
    assert len(merged) == len(ITEMS) * N_FOLDS
    assert (merged["forecast_start_base"] == merged["forecast_start_pool"]).all()
    assert (merged["forecast_end_base"] == merged["forecast_end_pool"]).all()

    # and they are the folds generate_rolling_origins defines on the development length
    dev_len = N_DAYS - HOLDOUT_DAYS
    for origin in generate_rolling_origins(dev_len, horizon=HORIZON, n_folds=N_FOLDS):
        row = pooled["fold_summary"][(pooled["fold_summary"]["fold"] == origin.fold)].iloc[0]
        assert row["forecast_start"] == (START + pd.Timedelta(days=origin.train_end)).date()
        assert row["forecast_end"] == (START + pd.Timedelta(days=origin.test_end - 1)).date()


def test_the_last_fold_ends_on_the_last_development_day_never_in_the_holdout():
    out = _run(_daily_sales())
    assert pd.to_datetime(out["long"]["date"]).max() == DEV_END
    assert pd.to_datetime(out["fold_summary"]["forecast_end"]).max() == DEV_END


def test_every_fit_receives_only_the_training_prefix_of_every_item():
    created = []

    def factory():
        created.append(_FakePooled())
        return created[-1]

    out = _run(_daily_sales(), factory)
    origins = generate_rolling_origins(N_DAYS - HOLDOUT_DAYS, horizon=HORIZON, n_folds=N_FOLDS)
    assert len(created) == N_FOLDS                     # a fresh model for every fold
    for model, origin in zip(created, origins):
        for item in ITEMS:
            first, last, n = model.seen[item]
            assert first == START and n == origin.train_end
            assert last == START + pd.Timedelta(days=origin.train_end - 1)
            assert last < START + pd.Timedelta(days=origin.test_start)   # strictly before the forecast window
    diag = out["fold_diagnostics"]
    assert (pd.to_datetime(diag["origin_date"]) + pd.Timedelta(days=1) == pd.to_datetime(diag["forecast_start"])).all()


# ---- information boundary: final holdout and each fold's own actuals -----------------

def test_final_holdout_values_never_reach_any_forecast():
    ds = _daily_sales()
    base = _run(ds)
    sentinel = ds.copy()
    in_holdout = sentinel["date"] > DEV_END
    assert int(in_holdout.sum()) == len(ITEMS) * HOLDOUT_DAYS
    sentinel.loc[in_holdout, "sales"] = 1_000_000.0
    changed = _run(sentinel)
    np.testing.assert_array_equal(base["long"]["predicted"].to_numpy(), changed["long"]["predicted"].to_numpy())
    np.testing.assert_array_equal(base["long"]["actual"].to_numpy(), changed["long"]["actual"].to_numpy())
    assert changed["long"]["actual"].max() < 1_000_000.0
    assert pd.to_datetime(changed["long"]["date"]).max() == DEV_END


def test_forecasts_never_depend_on_their_own_fold_actuals_but_later_folds_do():
    ds = _daily_sales()
    base = _run(ds)
    for fold in range(1, N_FOLDS + 1):
        window = pd.to_datetime(base["long"].loc[base["long"]["fold"] == fold, "date"].unique())
        perturbed = ds.copy()
        perturbed.loc[perturbed["date"].isin(window), "sales"] += 50.0
        changed = _run(perturbed)
        for earlier in range(1, fold + 1):
            same = base["long"]["fold"] == earlier
            np.testing.assert_array_equal(
                base["long"].loc[same, "predicted"].to_numpy(),
                changed["long"].loc[same, "predicted"].to_numpy(),
                err_msg=f"fold {earlier} forecasts moved when fold {fold}'s actuals were changed",
            )
        if fold < N_FOLDS:   # control: the changed days become training data for the next fold
            nxt = base["long"]["fold"] == fold + 1
            assert not np.array_equal(
                base["long"].loc[nxt, "predicted"].to_numpy(), changed["long"].loc[nxt, "predicted"].to_numpy()
            ), "the perturbation had no effect on later folds, so this test cannot detect leakage"
        assert not np.array_equal(
            base["long"].loc[base["long"]["fold"] == fold, "actual"].to_numpy(),
            changed["long"].loc[changed["long"]["fold"] == fold, "actual"].to_numpy(),
        )


def test_each_items_forecasts_and_metrics_are_attributed_to_that_item():
    ds = _daily_sales()
    constants = {"A": 1.0, "B": 2.0, "C": 3.0}
    out = _run(ds, lambda: _FakePooled(value_by_item=constants))
    long, fs = out["long"], out["fold_summary"]
    for item, value in constants.items():
        rows = long[long["item_id"] == item]
        assert (rows["predicted"] == value).all()
        expected_mae = rows.groupby("fold").apply(lambda g: (g["actual"] - value).abs().mean(), include_groups=False)
        got = fs[fs["item_id"] == item].set_index("fold")["MAE"]
        np.testing.assert_allclose(got.to_numpy(), expected_mae.reindex(got.index).to_numpy())


def test_the_runner_is_deterministic():
    ds = _daily_sales()
    first, second = _run(ds), _run(ds)
    np.testing.assert_array_equal(first["long"]["predicted"].to_numpy(), second["long"]["predicted"].to_numpy())


# ---- failures are recorded; protocol errors raise ------------------------------------

def test_a_failed_fold_is_recorded_for_every_item_not_dropped():
    calls = {"n": 0}

    def factory():
        calls["n"] += 1
        return _FakePooled(fail=(calls["n"] == 2))

    out = _run(_daily_sales(), factory)
    fs, long = out["fold_summary"], out["long"]
    failed = fs[fs["fold"] == 2]
    assert len(failed) == len(ITEMS) and (~failed["success"]).all()
    assert failed["error"].str.startswith("RuntimeError: boom").all()
    assert failed[["MAE", "RMSE", "MAPE"]].isna().all().all()
    assert fs[fs["fold"] != 2]["success"].all()
    failed_rows = long[long["fold"] == 2]
    assert len(failed_rows) == len(ITEMS) * HORIZON
    assert failed_rows["predicted"].isna().all() and failed_rows["actual"].notna().all()
    diag = out["fold_diagnostics"].set_index("fold")
    assert diag.loc[2, "success"] == False and "boom" in diag.loc[2, "error"]  # noqa: E712
    assert bool(diag.loc[1, "success"]) and bool(diag.loc[3, "success"])

    summary = per_item_summary(fs)
    assert (summary["n_folds"] == N_FOLDS).all() and (summary["n_success"] == N_FOLDS - 1).all()
    assert summary["MAE_mean"].notna().all()           # averaged over the folds that succeeded, not zero-filled


def test_forecast_dates_that_do_not_match_the_fold_raise():
    with pytest.raises(ValueError, match="forecast dates"):
        _run(_daily_sales(), lambda: _FakePooled(date_shift=1))


def test_a_wrong_length_forecast_raises():
    with pytest.raises(ValueError, match="length"):
        _run(_daily_sales(), lambda: _FakePooled(length=HORIZON - 1))


def test_forecasts_for_the_wrong_set_of_items_raise():
    with pytest.raises(ValueError, match="different set of items"):
        _run(_daily_sales(), lambda: _FakePooled(items=["A", "B"]))


def test_items_on_different_calendars_cannot_be_pooled():
    ds = _daily_sales()
    last_c = ds[(ds["item_id"] == "C")]["date"].max()
    ds = ds[~((ds["item_id"] == "C") & (ds["date"] == last_c))]
    with pytest.raises(ValueError, match="shared"):
        _run(ds)


def test_invalid_item_lists_raise():
    ds = _daily_sales()
    factory = lambda: _FakePooled()  # noqa: E731
    with pytest.raises(ValueError):
        run_pooled_item_level_backtest(ds, [], factory, HORIZON, N_FOLDS, HOLDOUT_DAYS)
    with pytest.raises(ValueError):
        run_pooled_item_level_backtest(ds, ["A", "A"], factory, HORIZON, N_FOLDS, HOLDOUT_DAYS)
    with pytest.raises(ValueError):
        run_pooled_item_level_backtest(ds, ["A", "NOT_AN_ITEM"], factory, HORIZON, N_FOLDS, HOLDOUT_DAYS)


# ---- extra tables ---------------------------------------------------------------------

def test_demand_class_columns_keep_the_phase1_label_and_fold_classes_are_separate():
    out = _run(_daily_sales())
    labels = {item: label for item, (_, label) in BLOCKS.items()}
    assert (out["long"]["demand_class"] == out["long"]["item_id"].map(labels)).all()
    assert (out["fold_summary"]["demand_class"] == out["fold_summary"]["item_id"].map(labels)).all()

    classes = out["fold_item_classes"]
    assert len(classes) == len(ITEMS) * N_FOLDS
    assert {"fold", "item_id", "ADI", "CV2", "fold_local_class", "n_train_obs", "phase1_class"} <= set(classes.columns)
    assert (classes["phase1_class"] == classes["item_id"].map(labels)).all()
    origins = generate_rolling_origins(N_DAYS - HOLDOUT_DAYS, horizon=HORIZON, n_folds=N_FOLDS)
    expected_obs = {o.fold: o.train_end for o in origins}
    assert (classes["n_train_obs"] == classes["fold"].map(expected_obs)).all()


def test_fold_diagnostics_and_importance_tables_have_the_documented_content():
    out = _run(_daily_sales())
    diag = out["fold_diagnostics"]
    assert len(diag) == N_FOLDS and diag["success"].all()
    for column in ["model", "fold", "origin_date", "forecast_start", "forecast_end", "n_train_rows",
                   "n_warmup_rows_dropped", "n_predictions_clipped_at_zero", "random_state", "xgboost_version"]:
        assert column in diag.columns
    assert (diag["n_warmup_rows_dropped"] == len(ITEMS) * 28).all()

    importance = out["feature_importance"]
    assert set(importance["fold"]) == set(range(1, N_FOLDS + 1))
    assert importance.groupby("fold")["feature"].nunique().eq(CFG.feature_names(pooled=True).__len__()).all()
    assert (importance["gain"] >= 0).all()


def test_a_model_without_class_or_importance_methods_still_runs():
    out = _run(_daily_sales(), lambda: _FakePooled())
    assert out["fold_item_classes"].empty and out["feature_importance"].empty
    assert len(out["fold_diagnostics"]) == N_FOLDS


# ---- reuse of the existing summaries -------------------------------------------------

def test_pooled_results_flow_through_the_existing_summaries_next_to_baselines():
    ds = _daily_sales()
    baseline = run_item_level_backtests(
        ds, ITEMS, {"SeasonalNaive": lambda: SeasonalNaive(season_length=7)},
        horizon=HORIZON, n_folds=N_FOLDS, holdout_days=HOLDOUT_DAYS,
    )
    pooled = _run(ds)
    combined = pd.concat([baseline["fold_summary"], pooled["fold_summary"]], ignore_index=True)
    per_item = per_item_summary(combined)
    assert set(per_item["model"]) == {"SeasonalNaive", "XGBoost(pooled)"}
    assert len(per_item) == 2 * len(ITEMS)
    category = category_macro_average_summary(per_item)
    assert set(category["demand_class"]) == {"Intermittent", "Smooth", "Lumpy"}
    assert (category["n_items"] == 1).all()
    assert category[["MAE_macro_mean", "RMSE_macro_mean"]].notna().all().all()


# ---- model-keyed bar colours -------------------------------------------------------------

def test_category_bars_are_coloured_by_model_name_not_by_position():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import to_hex

    from src.evaluation.item_level import _MODEL_PLOT_COLORS, plot_category_comparison_by_model_color

    rows = []
    for category, offset in zip(["Smooth", "Intermittent", "Erratic", "Lumpy"], [1.0, 2.0, 3.0, 4.0]):
        for model, bump in [("TSB(alpha=0.1,beta=0.1)", 0.0), ("XGBoost(pooled)", 0.5), ("SomethingElse", 1.0)]:
            rows.append({"demand_class": category, "model": model, "MAE_macro_mean": offset + bump})
    summary = pd.DataFrame(rows)
    ax = plot_category_comparison_by_model_color(summary, metric="MAE_macro_mean")
    try:
        assert len(ax.patches) == 4 * 3
        by_label = {}
        for container in ax.containers:
            by_label[container.get_label()] = {to_hex(p.get_facecolor()) for p in container.patches}
        assert by_label["TSB(alpha=0.1,beta=0.1)"] == {_MODEL_PLOT_COLORS["TSB"].lower()}
        assert by_label["XGBoost(pooled)"] == {_MODEL_PLOT_COLORS["XGBoost"].lower()}
        assert by_label["SomethingElse"] == {"#4c4c4c"}                    # neutral fallback for unknown models
        assert [t.get_text() for t in ax.get_xticklabels()] == ["Smooth", "Intermittent", "Erratic", "Lumpy"]
        assert ax.get_ylabel() == "MAE macro mean"
        # the same model keeps its colour when other models are dropped from the chart
        ax2 = plot_category_comparison_by_model_color(summary[summary["model"] == "XGBoost(pooled)"])
        assert {to_hex(p.get_facecolor()) for p in ax2.patches} == {_MODEL_PLOT_COLORS["XGBoost"].lower()}
    finally:
        plt.close("all")


# ---- check_same_evaluation_grid ------------------------------------------------------

def _grid(items=("A", "B"), folds=(1, 2), days=3, model="m", shift=0.0) -> pd.DataFrame:
    rows = []
    for item in items:
        for fold in folds:
            for d in range(days):
                rows.append(
                    {
                        "item_id": item, "fold": fold, "model": model, "predicted": 0.0,
                        "date": (START + pd.Timedelta(days=(fold - 1) * days + d)).date(),
                        "actual": float(d) + shift,
                    }
                )
    return pd.DataFrame(rows)


def test_identical_grids_pass_and_report_their_size():
    info = check_same_evaluation_grid(_grid(), _grid(model="other"))
    assert info == {"n_cells": 12, "n_items": 2, "n_folds": 2, "max_abs_actual_diff": 0.0}


def test_dates_may_be_strings_timestamps_or_dates_and_tables_may_hold_several_models():
    as_text = _grid()
    as_text["date"] = as_text["date"].astype(str)                      # what a CSV read-back gives
    as_ts = _grid()
    as_ts["date"] = pd.to_datetime(as_ts["date"])
    many_models = pd.concat([_grid(model="m1"), _grid(model="m2")], ignore_index=True)
    check_same_evaluation_grid(as_text, as_ts)
    check_same_evaluation_grid(many_models, as_text)


def test_a_missing_or_extra_cell_is_rejected():
    full = _grid()
    with pytest.raises(ValueError, match="differ"):
        check_same_evaluation_grid(full, full.iloc[1:])
    with pytest.raises(ValueError, match="differ"):
        check_same_evaluation_grid(full.iloc[1:], full)
    with pytest.raises(ValueError, match="differ"):
        check_same_evaluation_grid(_grid(items=("A",)), _grid(items=("A", "B")))


def test_different_actual_values_are_rejected_unless_within_the_tolerance():
    with pytest.raises(ValueError, match="Actual values differ"):
        check_same_evaluation_grid(_grid(), _grid(shift=0.5))
    info = check_same_evaluation_grid(_grid(), _grid(shift=0.5), atol=0.5)
    assert info["max_abs_actual_diff"] == 0.5


def test_conflicting_actuals_for_one_cell_inside_a_table_are_rejected():
    clash = pd.concat([_grid(), _grid(shift=1.0)], ignore_index=True)
    with pytest.raises(ValueError, match="more than one actual"):
        check_same_evaluation_grid(_grid(), clash)


def test_missing_columns_are_reported():
    with pytest.raises(ValueError, match="missing column"):
        check_same_evaluation_grid(_grid().drop(columns=["actual"]), _grid())
