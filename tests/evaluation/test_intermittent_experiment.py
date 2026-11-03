"""Tests for src/evaluation/intermittent_experiment.py (notebook 05's engine), on synthetic data."""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import pytest  # noqa: E402

from src.evaluation import intermittent_experiment as ix  # noqa: E402
from src.utils.config import load_config  # noqa: E402

CONFIG = load_config()


def _series(values, start="2020-01-01"):
    return pd.Series(np.asarray(values, dtype=float), index=pd.date_range(start, periods=len(values), freq="D"))


def _intermittent(n=90, every=3, size=2.0, offset=0):
    values = np.zeros(n)
    values[offset::every] = size
    return _series(values)


# ----------------------------------------------------------------------------------- patterns
def test_zero_run_lengths_basic_cases():
    assert ix.zero_run_lengths([0, 0, 1, 0, 3, 0, 0, 0]).tolist() == [2, 1, 3]
    assert ix.zero_run_lengths([0, 0, 0]).tolist() == [3]
    assert ix.zero_run_lengths([1, 2, 3]).tolist() == []
    assert ix.zero_run_lengths([5]).tolist() == []


@pytest.mark.parametrize("bad", [[], [1, np.nan], [1, -1], [np.inf, 1]])
def test_zero_run_lengths_rejects_invalid_input(bad):
    with pytest.raises(ValueError):
        ix.zero_run_lengths(bad)


def test_demand_pattern_summary_matches_hand_computation():
    summary = ix.demand_pattern_summary(_series([0, 0, 2, 0, 4, 0, 0, 0, 0, 6]))
    assert summary["n_days"] == 10 and summary["n_demand_days"] == 3
    assert summary["zero_share"] == pytest.approx(0.7)
    assert summary["occurrence_probability"] == pytest.approx(0.3)
    assert summary["ADI"] == pytest.approx(10 / 3)
    assert summary["mean_interval"] == pytest.approx((3 + 2 + 5) / 3)
    assert summary["mean_size"] == pytest.approx(4.0)
    assert summary["median_size"] == pytest.approx(4.0)
    assert summary["max_size"] == pytest.approx(6.0)
    assert summary["n_zero_runs"] == 3 and summary["longest_zero_run"] == 4
    assert summary["mean_zero_run"] == pytest.approx((2 + 1 + 4) / 3)
    assert summary["mean_daily_demand"] == pytest.approx(1.2)
    assert summary["leading_zero_days"] == 2 and summary["trailing_zero_days"] == 0


def test_leading_and_trailing_zero_days():
    summary = ix.demand_pattern_summary(_series([0, 0, 0, 5, 0, 1, 0, 0]))
    assert summary["leading_zero_days"] == 3 and summary["trailing_zero_days"] == 2
    assert summary["longest_zero_run"] == 3
    never = ix.demand_pattern_summary(_series([0, 0, 0]))
    assert never["leading_zero_days"] == 3 and never["trailing_zero_days"] == 3
    always = ix.demand_pattern_summary(_series([1, 2]))
    assert always["leading_zero_days"] == 0 and always["trailing_zero_days"] == 0


def test_demand_pattern_summary_of_a_series_with_no_demand_is_nan_not_an_error():
    summary = ix.demand_pattern_summary(_series([0, 0, 0, 0]))
    assert summary["n_demand_days"] == 0 and summary["zero_share"] == 1.0
    assert np.isnan(summary["ADI"]) and np.isnan(summary["mean_size"]) and np.isnan(summary["mean_interval"])
    assert summary["longest_zero_run"] == 4


def test_demand_pattern_summary_of_a_series_that_always_sells():
    summary = ix.demand_pattern_summary(_series([3, 3, 3, 3]))
    assert summary["zero_share"] == 0.0 and summary["ADI"] == 1.0 and summary["n_zero_runs"] == 0
    assert summary["longest_zero_run"] == 0 and summary["mean_zero_run"] == 0.0


def test_pattern_and_run_tables_are_ordered_by_class_then_item():
    items = {"B": _intermittent(), "A": _series([4, 5, 4, 6] * 10), "C": _intermittent(every=5)}
    classes = {"B": "Intermittent", "A": "Smooth", "C": "Lumpy"}
    table = ix.demand_pattern_table(items, classes)
    assert table["item_id"].tolist() == ["A", "B", "C"]
    runs = ix.zero_run_table(items, classes)
    assert set(runs["item_id"]) == {"B", "C"} and (runs["run_length"] > 0).all()
    assert list(runs.columns) == ["item_id", "demand_class", "run_length"]


# --------------------------------------------------------------------------- models and folds
def test_baseline_factories_use_the_phase2_settings_and_names():
    factories = ix.baseline_factories(CONFIG)
    assert list(factories) == ["SeasonalNaive", "Croston(alpha=0.1)", "TSB(alpha=0.1,beta=0.1)"]
    assert all(factory().name == name for name, factory in factories.items())


def test_baseline_factories_follow_the_config_values():
    config = {
        "backtest": {"season_length": 7},
        "models": {"croston_alpha": 0.2, "tsb_alpha": 0.3, "tsb_beta": 0.4},
    }
    assert list(ix.baseline_factories(config)) == ["SeasonalNaive", "Croston(alpha=0.2)", "TSB(alpha=0.3,beta=0.4)"]


def test_mase_scale_uses_only_the_training_window_of_each_fold():
    values = np.arange(160, dtype=float) % 7 + np.arange(160) * 0.1
    scales = ix.item_mase_scales({"X": _series(values)}, horizon=7, n_folds=3)
    assert scales["fold"].tolist() == [1, 2, 3]
    for row in scales.itertuples():
        train_end = 160 - (3 - row.fold + 1) * 7
        expected = np.mean(np.abs(values[7:train_end] - values[: train_end - 7]))
        assert row.mase_scale == pytest.approx(expected)
    assert scales["mase_scale"].is_monotonic_increasing or scales["mase_scale"].nunique() >= 1


def test_run_item_baselines_covers_every_cell_and_keeps_failed_folds():
    late_start = np.zeros(60)
    late_start[50:] = [1, 0, 2, 0, 1, 0, 3, 0, 0, 1]
    items = {"early": _intermittent(60), "late": _series(late_start)}
    classes = {"early": "Intermittent", "late": "Lumpy"}
    factories = ix.baseline_factories(CONFIG)
    long, summary = ix.run_item_baselines(items, classes, factories, horizon=7, n_folds=3)
    assert len(long) == 2 * 3 * 3 * 7 and len(summary) == 2 * 3 * 3
    late_croston = summary[(summary["item_id"] == "late") & (summary["model"].str.startswith("Croston"))]
    assert late_croston.sort_values("fold")["success"].tolist() == [False, False, True]
    failed = long[(long["item_id"] == "late") & long["model"].str.startswith("Croston") & ~long["success"]]
    assert len(failed) == 14 and failed["predicted"].isna().all() and failed["error"].str.len().gt(0).all()
    assert summary[(summary["model"] == "SeasonalNaive") | summary["model"].str.startswith("TSB")]["success"].all()


def test_zero_forecast_is_defined_everywhere_and_uses_the_same_cells():
    items = {"a": _intermittent(60)}
    long, _ = ix.run_item_baselines(items, {"a": "Intermittent"}, ix.baseline_factories(CONFIG), 7, 3)
    zero = ix.zero_forecast_long_table(long)
    assert zero["model"].unique().tolist() == [ix.ZERO_MODEL_NAME]
    assert (zero["predicted"] == 0.0).all() and zero["success"].all()
    assert len(zero) == len(long.drop_duplicates(["item_id", "fold", "date"]))
    pd.testing.assert_series_equal(
        zero.sort_values(["fold", "date"])["actual"].reset_index(drop=True),
        long[long["model"] == "SeasonalNaive"].sort_values(["fold", "date"])["actual"].reset_index(drop=True),
    )


def test_score_long_table_known_values_and_mase():
    items = {"a": _series(([0, 0, 4, 0, 0, 2, 0] * 12)[:84])}
    classes = {"a": "Intermittent"}
    long, _ = ix.run_item_baselines(items, classes, ix.baseline_factories(CONFIG), horizon=7, n_folds=3)
    long = pd.concat([long, ix.zero_forecast_long_table(long)], ignore_index=True)
    scales = ix.item_mase_scales(items, 7, 3)
    scored = ix.score_long_table(long, scales)
    sn = scored[scored["model"] == "SeasonalNaive"]
    assert (sn["MAE"] == 0).all()  # strictly periodic series -> perfect forecasts
    assert sn["MASE"].isna().all()  # ... but its seasonal-difference scale is 0, so MASE is undefined, not 0/inf
    zero = scored[scored["model"] == ix.ZERO_MODEL_NAME].sort_values("fold")
    window_sums = [6.0, 6.0, 6.0]
    assert zero["WAPE"].tolist() == pytest.approx([1.0, 1.0, 1.0])  # forecasting 0 misses all the demand
    assert zero["MAE"].tolist() == pytest.approx([s / 7 for s in window_sums])
    assert (zero["bias"] < 0).all() and {"forecast_start", "forecast_end"} <= set(scored.columns)


def test_mase_is_mae_over_the_training_scale_when_the_scale_is_defined():
    values = np.array(([0, 0, 4, 0, 0, 2, 0] * 12)[:84], dtype=float)
    values[[10, 40, 55]] = [3.0, 5.0, 1.0]  # break the exact periodicity
    items = {"a": _series(values)}
    long, _ = ix.run_item_baselines(items, {"a": "Intermittent"}, ix.baseline_factories(CONFIG), 7, 3)
    long = pd.concat([long, ix.zero_forecast_long_table(long)], ignore_index=True)
    scales = ix.item_mase_scales(items, 7, 3).set_index("fold")["mase_scale"]
    scored = ix.score_long_table(long, ix.item_mase_scales(items, 7, 3))
    assert scales.notna().all() and (scales > 0).all()
    for row in scored.itertuples():
        assert row.MASE == pytest.approx(row.MAE / scales.loc[row.fold])


def _fold_metrics(rows):
    columns = ["item_id", "demand_class", "model", "fold", "success", "MAE", "RMSE", "MAPE", "WAPE", "MASE", "bias"]
    return pd.DataFrame(rows, columns=columns)


def test_item_summary_skips_failed_folds_and_counts_defined_metrics():
    n = np.nan
    table = _fold_metrics(
        [
            ["a", "Smooth", "M", 1, True, 1.0, 2.0, 50.0, 0.5, 0.8, 0.1],
            ["a", "Smooth", "M", 2, True, 3.0, 4.0, n, n, n, -0.1],
            ["a", "Smooth", "M", 3, False, n, n, n, n, n, n],
        ]
    )
    summary = ix.item_summary(table).iloc[0]
    assert summary["n_folds"] == 3 and summary["n_success"] == 2
    assert summary["MAE_mean"] == pytest.approx(2.0) and summary["RMSE_mean"] == pytest.approx(3.0)
    assert summary["MAPE_mean"] == pytest.approx(50.0) and summary["MAPE_n_folds_valid"] == 1
    assert summary["WAPE_n_folds_valid"] == 1 and summary["MASE_mean"] == pytest.approx(0.8)
    assert summary["MAE_std"] == pytest.approx(np.std([1.0, 3.0], ddof=1))


def test_item_summary_of_an_item_with_no_successful_fold_is_nan_not_zero():
    table = _fold_metrics([["a", "Smooth", "M", 1, False] + [np.nan] * 6])
    summary = ix.item_summary(table).iloc[0]
    assert summary["n_success"] == 0 and np.isnan(summary["MAE_mean"]) and np.isnan(summary["MAE_std"])


def test_item_summary_rejects_missing_columns():
    with pytest.raises(ValueError, match="missing column"):
        ix.item_summary(pd.DataFrame({"item_id": ["a"]}))


def test_class_summary_is_an_equal_weight_macro_average_in_class_order():
    items = pd.DataFrame(
        {
            "item_id": ["small", "big", "x"],
            "demand_class": ["Lumpy", "Lumpy", "Smooth"],
            "model": ["M", "M", "M"],
            **{f"{m}_mean": [1.0, 100.0, 5.0] for m in ["MAE", "RMSE", "MAPE", "WAPE", "MASE", "bias"]},
        }
    )
    summary = ix.class_summary(items)
    assert summary["demand_class"].tolist() == ["Smooth", "Lumpy"]
    lumpy = summary[summary["demand_class"] == "Lumpy"].iloc[0]
    assert lumpy["MAE_macro_mean"] == pytest.approx(50.5)  # not volume weighted
    assert lumpy["MAE_item_min"] == 1.0 and lumpy["MAE_item_max"] == 100.0 and lumpy["n_items"] == 2


def test_reproduction_check_detects_identical_changed_and_missing_rows():
    base = pd.DataFrame(
        {
            "item_id": ["a", "a"], "model": ["M", "M"], "fold": [1, 2],
            "MAE": [1.0, 2.0], "RMSE": [1.5, 2.5], "MAPE": [np.nan, 40.0],
        }
    )
    assert ix.reproduction_check(base.copy(), base)["reproduced"] is True
    changed = base.copy()
    changed.loc[0, "MAE"] += 1e-3
    verdict = ix.reproduction_check(changed, base)
    assert verdict["reproduced"] is False and verdict["max_abs_diff_MAE"] == pytest.approx(1e-3)
    verdict = ix.reproduction_check(base.iloc[:1], base)
    assert verdict["reproduced"] is False and verdict["n_only_in_saved_file"] == 1
    nan_flip = base.copy()
    nan_flip.loc[0, "MAPE"] = 10.0
    assert ix.reproduction_check(nan_flip, base)["reproduced"] is False


# ---------------------------------------------------------------------------------- states
def test_smoothing_states_of_constant_demand():
    table = ix.smoothing_state_table({"c": _series([2.0] * 60)}, {"c": "Smooth"}, 7, 3, 0.1, 0.1, 0.1)
    assert len(table) == 3
    assert (table["croston_status"] == "ok").all()
    assert table["croston_size"].tolist() == pytest.approx([2.0] * 3)
    assert table["croston_interval"].tolist() == pytest.approx([1.0] * 3)
    assert table["croston_forecast"].tolist() == pytest.approx([2.0] * 3)
    assert table["train_probability"].tolist() == [1.0] * 3
    assert table["window_mean_actual"].tolist() == [2.0] * 3
    # TSB seeds both states at 0, so they climb towards 2 / 1 from below and have not reached them yet.
    assert ((table["tsb_size"] > 1.9) & (table["tsb_size"] < 2.0)).all()
    assert (table["tsb_forecast"] < table["tsb_size"]).all() and table["tsb_forecast"].is_monotonic_increasing


def test_smoothing_states_when_the_training_window_has_no_demand():
    values = np.zeros(60)
    values[55:] = 1.0
    table = ix.smoothing_state_table({"z": _series(values)}, {"z": "Lumpy"}, 7, 3, 0.1, 0.1, 0.1)
    first = table[table["fold"] == 1].iloc[0]
    assert first["croston_status"].startswith("failed") and np.isnan(first["croston_forecast"])
    assert first["tsb_forecast"] == 0.0 and first["train_demand_days"] == 0 and np.isnan(first["train_mean_size"])


def test_smoothing_states_only_describe_the_training_window():
    base = _intermittent(80)
    changed = base.copy()
    changed.iloc[-7:] = 99.0  # the last fold's forecast window
    a = ix.smoothing_state_table({"x": base}, {"x": "Intermittent"}, 7, 3, 0.1, 0.1, 0.1)
    b = ix.smoothing_state_table({"x": changed}, {"x": "Intermittent"}, 7, 3, 0.1, 0.1, 0.1)
    for column in ["train_probability", "croston_forecast", "tsb_forecast", "train_mean_demand"]:
        pd.testing.assert_series_equal(a[column], b[column])
    assert a["window_mean_actual"].iloc[-1] != b["window_mean_actual"].iloc[-1]


# ---------------------------------------------------------------- all-item run, fold-local classes
def _wide(n=140, seed=0):
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2020-01-01", periods=n, freq="D")
    columns = {
        "I1": rng.integers(4, 7, n).astype(float),
        "I2": _intermittent(n, every=3, size=1.0).to_numpy(),
        "I3": np.where(rng.random(n) < 0.1, rng.integers(1, 30, n), 0).astype(float),
        "I4": np.where(rng.random(n) < 0.6, rng.integers(1, 4, n), 0).astype(float),
    }
    return pd.DataFrame(columns, index=dates)


def test_fold_local_classes_ignore_everything_after_the_fold_origin():
    wide = _wide()
    base = ix.fold_local_class_table(wide, horizon=7, n_folds=5)
    assert len(base) == 5 * wide.shape[1]
    perturbed = wide.copy()
    perturbed.iloc[-7:] = 500.0  # fold 5's forecast window is used by no training prefix
    assert ix.fold_local_class_table(perturbed, 7, 5).equals(base)
    perturbed2 = wide.copy()
    perturbed2.iloc[-14:-7] = 500.0  # fold 4's window: only fold 5's training prefix may see it
    after = ix.fold_local_class_table(perturbed2, 7, 5)
    assert after[after["fold"] <= 4].equals(base[base["fold"] <= 4])
    assert (after[after["fold"] == 5]["CV2"].to_numpy() != base[base["fold"] == 5]["CV2"].to_numpy()).any()


def test_fold_local_classes_follow_the_sbc_rule_on_clear_cases():
    wide = _wide()
    table = ix.fold_local_class_table(wide, 7, 5, phase1_classes={"I1": "Smooth", "I2": "Intermittent"})
    assert (table[table["item_id"] == "I1"]["fold_local_class"] == "Smooth").all()
    assert (table[table["item_id"] == "I2"]["fold_local_class"] == "Intermittent").all()
    agreement = ix.class_agreement(table)
    assert agreement["n_items"].tolist() == [2] * 5 and agreement["share_same_label"].tolist() == [1.0] * 5


def test_class_agreement_requires_the_phase1_column():
    with pytest.raises(ValueError, match="phase1_class"):
        ix.class_agreement(pd.DataFrame({"fold": [1]}))


def test_all_items_run_end_to_end_on_synthetic_data():
    wide = _wide()
    factories = ix.baseline_factories(CONFIG)
    metrics, classes = ix.run_all_items_baselines(wide, factories, 7, 5, {"I1": "Smooth"})
    assert set(metrics["model"]) == set(factories) | {ix.ZERO_MODEL_NAME}
    assert len(metrics) == wide.shape[1] * 4 * 5 and metrics["fold_local_class"].notna().all()
    assert len(classes) == wide.shape[1] * 5
    per_fold = ix.fold_class_summary(metrics)
    assert {"fold", "fold_local_class", "model", "n_items", "MAE_macro_mean"} <= set(per_fold.columns)
    overall = ix.all_items_class_summary(per_fold)
    assert (overall["n_folds"] >= 1).all() and "MAE_across_folds_std" in overall.columns
    zero = metrics[metrics["model"] == ix.ZERO_MODEL_NAME]
    assert zero["success"].all()


def test_load_all_item_development_sales_cuts_the_holdout_before_pivoting(tmp_path):
    dates = pd.date_range("2020-01-01", periods=100, freq="D")
    rows = [
        {"item_id": item, "date": d, "sales": (1e9 if i >= 72 else float(j))}
        for j, item in enumerate(["B_item", "A_item"])
        for i, d in enumerate(dates)
    ]
    panel = pd.DataFrame(rows)
    panel["item_id"] = panel["item_id"].astype("category")
    (tmp_path / "interim").mkdir()
    panel.to_parquet(tmp_path / "interim" / "panel.parquet")
    config = {
        "data": {"interim_dir": str(tmp_path / "interim")},
        "output": {"panel_filename": "panel.parquet"},
        "backtest": {"holdout_days": 28},
    }
    wide, window = ix.load_all_item_development_sales(config)
    assert wide.shape == (72, 2) and list(wide.columns) == ["A_item", "B_item"]
    assert wide.index.max() == window.dev_end == pd.Timestamp("2020-03-12")
    assert wide.to_numpy().max() < 1e6  # none of the reserved days leaked in
    assert window.n_holdout_days == 28


def test_load_all_item_development_sales_names_the_missing_file(tmp_path):
    config = {
        "data": {"interim_dir": str(tmp_path)},
        "output": {"panel_filename": "absent.parquet"},
        "backtest": {"holdout_days": 28},
    }
    with pytest.raises(FileNotFoundError, match="absent.parquet"):
        ix.load_all_item_development_sales(config)


def test_load_all_item_development_sales_rejects_gaps(tmp_path):
    dates = pd.date_range("2020-01-01", periods=60, freq="D")
    panel = pd.DataFrame({"item_id": "A", "date": dates, "sales": 1})
    panel = panel.drop(index=10)
    panel.to_parquet(tmp_path / "panel.parquet")
    config = {
        "data": {"interim_dir": str(tmp_path)},
        "output": {"panel_filename": "panel.parquet"},
        "backtest": {"holdout_days": 28},
    }
    with pytest.raises(ValueError):
        ix.load_all_item_development_sales(config)


# --------------------------------------------------------------------------------------- plots
def test_plots_build_and_use_more_than_colour_to_distinguish_classes():
    items = {
        "S": _series([4, 5, 6, 5] * 40), "I": _intermittent(160), "E": _series(([0, 9, 1, 0, 25] * 32)[:160]),
        "L": _intermittent(160, every=11, size=7.0),
    }
    classes = {"S": "Smooth", "I": "Intermittent", "E": "Erratic", "L": "Lumpy"}
    table = ix.demand_pattern_table(items, classes)
    runs = ix.zero_run_table(items, classes)
    figures = [ix.plot_demand_patterns(items, table, days=60), ix.plot_zero_runs(runs)]
    summary = pd.DataFrame(
        {
            "demand_class": ["Smooth", "Smooth", "Lumpy", "Lumpy"],
            "model": ["SeasonalNaive", ix.ZERO_MODEL_NAME] * 2,
            "MAE_macro_mean": [1.0, 2.0, 3.0, 1.5],
        }
    )
    bar_fig = ix.plot_class_metric(summary, "MAE")
    figures.append(bar_fig)
    hatches = {patch.get_hatch() for patch in bar_fig.axes[0].patches}
    assert len({h for h in hatches if h}) >= 2
    with pytest.raises(ValueError, match="no column"):
        ix.plot_class_metric(summary, "RMSE")
    for fig in figures:
        plt.close(fig)
