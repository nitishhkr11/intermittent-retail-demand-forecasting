"""Tests for src/evaluation/forecast_artifacts.py: reading, auditing and scoring saved forecasts."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.evaluation.forecast_artifacts import (
    ArtifactError,
    MissingArtifactError,
    add_horizon_step,
    audit_forecast_table,
    check_same_grid,
    expected_fold_windows,
    fold_training_scales,
    forecasts_dir,
    load_aggregate_forecast_tables,
    load_item_forecast_tables,
    metrics_by_group,
    metrics_dir,
    notebook_command,
    producer_hint,
    producer_notebook,
    read_forecast_table,
    seasonal_naive_scale,
    truncate_horizon,
    validate_forecast_table,
)
from src.utils.config import load_config

N_DEV = 200
HORIZON, FOLDS = 28, 5


@pytest.fixture()
def dev_series():
    dates = pd.date_range("2020-01-01", periods=N_DEV, freq="D")
    rng = np.random.default_rng(0)
    values = 100 + 20 * np.sin(np.arange(N_DEV) * 2 * np.pi / 7) + rng.integers(0, 5, N_DEV)
    return pd.Series(values.astype(float), index=dates, name="aggregate_sales")


def _aggregate_table(dev_series, model="M", offset=1.0):
    windows = expected_fold_windows(dev_series.index, HORIZON, FOLDS)
    rows = []
    for w in windows.itertuples(index=False):
        for day in pd.date_range(w.forecast_start, periods=HORIZON, freq="D"):
            rows.append((model, w.fold, day, dev_series.loc[day], dev_series.loc[day] + offset))
    return pd.DataFrame(rows, columns=["model", "fold", "date", "actual", "predicted"]).assign(success=True, error=None)


def _failed(results):
    return {r.name for r in results if not r.passed}


# --------------------------------------------------------------------------- fold grid

def test_expected_fold_windows_are_contiguous_and_end_at_the_last_development_day(dev_series):
    windows = expected_fold_windows(dev_series.index, HORIZON, FOLDS)
    assert windows["fold"].tolist() == [1, 2, 3, 4, 5]
    assert windows["forecast_end"].iloc[-1] == dev_series.index[-1]
    assert (windows["forecast_end"] - windows["forecast_start"]).eq(pd.Timedelta(days=HORIZON - 1)).all()
    gaps = windows["forecast_start"].iloc[1:].to_numpy() - windows["forecast_end"].iloc[:-1].to_numpy()
    assert (gaps == np.timedelta64(1, "D")).all()
    assert (windows["origin_date"] + pd.Timedelta(days=1) == windows["forecast_start"]).all()
    assert windows["n_train"].tolist() == [N_DEV - 28 * k for k in (5, 4, 3, 2, 1)]


def test_expected_fold_windows_reproduce_the_project_calendar():
    dates = pd.date_range("2011-01-29", "2016-04-24", freq="D")
    windows = expected_fold_windows(dates, 28, 5)
    assert windows["forecast_start"].dt.strftime("%Y-%m-%d").tolist() == [
        "2015-12-07", "2016-01-04", "2016-02-01", "2016-02-29", "2016-03-28",
    ]
    assert windows["n_train"].tolist() == [1773, 1801, 1829, 1857, 1885]


def test_expected_fold_windows_need_enough_history():
    with pytest.raises(ValueError):
        expected_fold_windows(pd.date_range("2020-01-01", periods=100, freq="D"), 28, 5)


# --------------------------------------------------------------------------- reading

def test_missing_file_raises_with_the_producer_command_and_no_absolute_path(tmp_path):
    with pytest.raises(MissingArtifactError) as excinfo:
        read_forecast_table(tmp_path / "lstm_gru_backtest_forecasts.csv")
    message = str(excinfo.value)
    assert "lstm_gru_backtest_forecasts.csv" in message
    assert "python -m jupyter nbconvert --to notebook --execute --inplace notebooks/07_lstm_gru.ipynb" in message
    assert str(tmp_path) not in message
    assert isinstance(excinfo.value, FileNotFoundError)


@pytest.mark.parametrize(
    "filename, notebook",
    [
        ("lstm_gru_backtest_forecasts.csv", "07_lstm_gru.ipynb"),
        ("prophet_backtest_metrics.csv", "08_prophet.ipynb"),
        ("xgboost_item_level_backtest_long.csv", "06_xgboost.ipynb"),
        ("intermittent_item_level_backtest_long.csv", "05_intermittent_models.ipynb"),
        ("comparison_aggregate_summary.csv", "09_model_comparison.ipynb"),
        ("inventory_demo_reorder_points.csv", "10_inventory_decision_support.ipynb"),
        ("item_level_backtest_long.csv", "04_classical_models.ipynb"),
        ("arima_backtest_forecasts.csv", "04_classical_models.ipynb"),
    ],
)
def test_every_artifact_family_maps_to_its_notebook(filename, notebook):
    assert producer_notebook(filename) == notebook
    assert notebook_command(notebook) in producer_hint(filename)


def test_unknown_artifact_gets_a_generic_hint():
    assert producer_notebook("mystery.csv") is None
    assert "Re-run the notebook" in producer_hint("mystery.csv")


def test_read_forecast_table_types_sorting_and_default_success(tmp_path, dev_series):
    table = _aggregate_table(dev_series).drop(columns=["success", "error"])
    shuffled = table.sample(frac=1.0, random_state=3)
    path = tmp_path / "f.csv"
    shuffled.to_csv(path, index=False)
    loaded = read_forecast_table(path, "aggregate")
    assert loaded["date"].dtype == "datetime64[ns]" and loaded["fold"].dtype.kind == "i"
    assert loaded["actual"].dtype == float and loaded["predicted"].dtype == float
    assert loaded["success"].all() and "error" in loaded.columns
    assert loaded.equals(loaded.sort_values(["model", "fold", "date"]).reset_index(drop=True))
    assert len(loaded) == HORIZON * FOLDS


def test_read_forecast_table_rejects_missing_columns_and_empty_files(tmp_path, dev_series):
    path = tmp_path / "bad.csv"
    _aggregate_table(dev_series).drop(columns=["predicted"]).to_csv(path, index=False)
    with pytest.raises(ArtifactError, match="missing column"):
        read_forecast_table(path, "aggregate")
    empty = tmp_path / "empty.csv"
    _aggregate_table(dev_series).iloc[:0].to_csv(empty, index=False)
    with pytest.raises(ArtifactError, match="no rows"):
        read_forecast_table(empty, "aggregate")
    with pytest.raises(ValueError, match="kind"):
        read_forecast_table(path, "weekly")


def test_item_tables_need_item_columns(tmp_path, dev_series):
    path = tmp_path / "agg_as_item.csv"
    _aggregate_table(dev_series).to_csv(path, index=False)
    with pytest.raises(ArtifactError, match="item_id"):
        read_forecast_table(path, "item")


def test_loaders_report_problems_instead_of_inventing_data(tmp_path, dev_series):
    config = {"phase3": {"forecasts_dir": str(tmp_path), "metrics_dir": str(tmp_path)}}
    _aggregate_table(dev_series).to_csv(tmp_path / "good.csv", index=False)
    (tmp_path / "broken.csv").write_text("model,fold\nx,1\n", encoding="utf-8")
    tables, problems = load_aggregate_forecast_tables(["good.csv", "broken.csv", "lstm_gru_backtest_forecasts.csv"], config)
    assert list(tables) == ["good.csv"]
    assert set(problems) == {"broken.csv", "lstm_gru_backtest_forecasts.csv"}
    assert "missing column" in problems["broken.csv"] and "07_lstm_gru.ipynb" in problems["lstm_gru_backtest_forecasts.csv"]
    item_tables, item_problems = load_item_forecast_tables(["good.csv"], config)
    assert item_tables == {} and "item_id" in item_problems["good.csv"]


def test_directory_helpers_follow_the_config(tmp_path):
    config = {"phase3": {"forecasts_dir": str(tmp_path / "f"), "metrics_dir": str(tmp_path / "m")}}
    assert forecasts_dir(config) == tmp_path / "f" and metrics_dir(config) == tmp_path / "m"


# --------------------------------------------------------------------------- audit

def test_a_clean_table_passes_every_check(dev_series):
    results = audit_forecast_table(_aggregate_table(dev_series), kind="aggregate", dev_series=dev_series)
    assert _failed(results) == set() and len(results) == 6
    assert validate_forecast_table(_aggregate_table(dev_series), kind="aggregate", dev_series=dev_series)


def test_audit_flags_a_date_inside_the_reserved_holdout(dev_series):
    table = _aggregate_table(dev_series)
    extra = table.iloc[[-1]].copy()
    extra["date"] = dev_series.index[-1] + pd.Timedelta(days=1)
    failed = _failed(audit_forecast_table(pd.concat([table, extra]), kind="aggregate", dev_series=dev_series))
    assert "no_dates_after_development_end" in failed


def test_audit_flags_a_wrong_window_a_missing_fold_and_duplicates(dev_series):
    table = _aggregate_table(dev_series)
    shifted = table.copy()
    shifted.loc[shifted["fold"] == 2, "date"] += pd.Timedelta(days=1)
    assert "folds_and_windows_match_protocol" in _failed(audit_forecast_table(shifted, kind="aggregate", dev_series=dev_series))
    missing_fold = table[table["fold"] != 4]
    assert "folds_and_windows_match_protocol" in _failed(audit_forecast_table(missing_fold, kind="aggregate", dev_series=dev_series))
    short = table.drop(index=table.index[3])
    assert "folds_and_windows_match_protocol" in _failed(audit_forecast_table(short, kind="aggregate", dev_series=dev_series))
    duplicated = pd.concat([table, table.iloc[[0]]])
    assert "no_duplicate_rows" in _failed(audit_forecast_table(duplicated, kind="aggregate", dev_series=dev_series))


def test_audit_flags_actuals_that_do_not_match_the_development_series(dev_series):
    table = _aggregate_table(dev_series)
    table.loc[10, "actual"] += 5.0
    assert "actuals_equal_development_series" in _failed(audit_forecast_table(table, kind="aggregate", dev_series=dev_series))


def test_audit_flags_negative_and_non_finite_predictions(dev_series):
    table = _aggregate_table(dev_series)
    table.loc[0, "predicted"] = -3.0
    assert _failed(audit_forecast_table(table, kind="aggregate", dev_series=dev_series)) == {"predictions_non_negative"}
    assert _failed(audit_forecast_table(table, kind="aggregate", dev_series=dev_series, require_non_negative=False)) == set()
    table.loc[1, "predicted"] = np.nan
    assert "predictions_finite_on_successful_rows" in _failed(audit_forecast_table(table, kind="aggregate", dev_series=dev_series))


def test_a_failed_row_may_carry_a_missing_prediction(dev_series):
    table = _aggregate_table(dev_series)
    table.loc[table["fold"] == 3, ["predicted", "success"]] = [np.nan, False]
    table["success"] = table["success"].astype(bool)
    assert _failed(audit_forecast_table(table, kind="aggregate", dev_series=dev_series)) == set()


def test_validate_raises_with_the_failed_check_names(dev_series):
    table = _aggregate_table(dev_series)
    table.loc[0, "predicted"] = -1.0
    with pytest.raises(ArtifactError, match="predictions_non_negative"):
        validate_forecast_table(table, kind="aggregate", dev_series=dev_series)


def test_audit_needs_a_reference_calendar():
    with pytest.raises(ValueError):
        audit_forecast_table(pd.DataFrame(), kind="aggregate")
    with pytest.raises(ValueError):
        audit_forecast_table(pd.DataFrame(), kind="daily", dev_dates=pd.date_range("2020-01-01", periods=200))


def _item_table(dev_series, items=("A", "B")):
    base = _aggregate_table(dev_series)
    parts = []
    for item in items:
        part = base.copy()
        part.insert(0, "item_id", item)
        part.insert(1, "demand_class", "smooth")
        parts.append(part)
    return pd.concat(parts, ignore_index=True)


def test_item_level_audit_checks_each_item_against_its_own_series(dev_series):
    table = _item_table(dev_series)
    item_series = {"A": dev_series, "B": dev_series}
    assert _failed(audit_forecast_table(table, kind="item", item_series=item_series, dev_dates=dev_series.index)) == set()
    bad = table.copy()
    bad.loc[bad["item_id"] == "B", "actual"] += 1.0
    assert "actuals_equal_development_series" in _failed(
        audit_forecast_table(bad, kind="item", item_series=item_series, dev_dates=dev_series.index)
    )
    unknown = {"A": dev_series}
    assert "actuals_equal_development_series" in _failed(
        audit_forecast_table(table, kind="item", item_series=unknown, dev_dates=dev_series.index)
    )
    skipped = audit_forecast_table(table, kind="item", dev_dates=dev_series.index)
    assert [r for r in skipped if r.name == "actuals_equal_development_series"][0].detail.startswith("skipped")


# --------------------------------------------------------------------------- same grid

def test_check_same_grid_accepts_identical_cells_for_different_models(dev_series):
    a, b = _aggregate_table(dev_series, "M1"), _aggregate_table(dev_series, "M2", offset=5.0)
    info = check_same_grid(a, b, ["fold", "date"])
    assert info == {"n_cells": HORIZON * FOLDS, "max_abs_actual_diff": 0.0}


def test_check_same_grid_rejects_different_dates_and_different_actuals(dev_series):
    a = _aggregate_table(dev_series, "M1")
    with pytest.raises(ArtifactError, match="grids differ"):
        check_same_grid(a, a.iloc[1:], ["fold", "date"])
    changed = a.copy()
    changed.loc[0, "actual"] += 1.0
    with pytest.raises(ArtifactError, match="Actual values differ"):
        check_same_grid(a, changed, ["fold", "date"])
    check_same_grid(a, changed, ["fold", "date"], atol=2.0)
    with pytest.raises(ArtifactError, match="missing column"):
        check_same_grid(a, a.drop(columns=["actual"]), ["fold", "date"])


# --------------------------------------------------------------------------- MASE scale

def test_seasonal_naive_scale_known_values_and_undefined_cases():
    alternating = [0, 1, 0, 1, 0, 1, 0, 1]
    assert seasonal_naive_scale(alternating, lag=1) == 1.0
    assert np.isnan(seasonal_naive_scale(alternating, lag=2))        # every seasonal difference is zero
    assert seasonal_naive_scale([1, 2, 4, 8], lag=1) == pytest.approx((1 + 2 + 4) / 3)
    assert np.isnan(seasonal_naive_scale([1, 2, 3], lag=3))          # not longer than the lag
    assert np.isnan(seasonal_naive_scale([1, np.nan, 3, 4], lag=1))
    with pytest.raises(ValueError):
        seasonal_naive_scale([1, 2, 3], lag=0)


def test_fold_training_scales_use_only_each_folds_training_window(dev_series):
    windows = expected_fold_windows(dev_series.index, HORIZON, FOLDS)
    scales = fold_training_scales(dev_series, windows, lag=7)
    for row in windows.itertuples(index=False):
        train = dev_series.iloc[: row.n_train].to_numpy()
        assert scales[row.fold] == pytest.approx(np.mean(np.abs(train[7:] - train[:-7])))
    # the scale of fold 1 must not change when later (test-window) values change
    changed = dev_series.copy()
    changed.iloc[-28:] = 1e6
    assert fold_training_scales(changed, windows, 7)[1] == scales[1]


# --------------------------------------------------------------------------- metrics

def _hand_table():
    return pd.DataFrame(
        {
            "model": ["M"] * 4 + ["N"] * 4,
            "fold": [1] * 8,
            "actual": [0.0, 2.0, 4.0, 4.0] * 2,
            "predicted": [1.0, 2.0, 2.0, 6.0, 0.0, 0.0, 0.0, 0.0],
            "success": True,
        }
    )


def test_metrics_by_group_matches_hand_calculation():
    scales = pd.DataFrame({"fold": [1], "mase_scale": [2.0]})
    result = metrics_by_group(_hand_table(), ["model", "fold"], scales).set_index("model")
    m = result.loc["M"]
    assert m["MAE"] == pytest.approx((1 + 0 + 2 + 2) / 4)
    assert m["RMSE"] == pytest.approx(np.sqrt((1 + 0 + 4 + 4) / 4))
    assert m["MAPE"] == pytest.approx(100 * np.mean([0 / 2, 2 / 4, 2 / 4]))      # the zero actual is excluded
    assert m["mape_n_excluded"] == 1
    assert m["WAPE"] == pytest.approx(5 / 10)
    assert m["bias"] == pytest.approx((1 + 0 - 2 + 2) / 4)
    assert m["MASE"] == pytest.approx(1.25 / 2.0)
    assert m["n"] == 4 and bool(m["success"])
    assert result.loc["N", "MAE"] == pytest.approx(10 / 4) and result.loc["N", "bias"] == pytest.approx(-2.5)


def test_metrics_by_group_handles_all_zero_actuals_and_undefined_scales():
    table = pd.DataFrame({"model": "Z", "fold": 1, "actual": [0.0, 0.0], "predicted": [1.0, 0.0], "success": True})
    row = metrics_by_group(table, ["model", "fold"], pd.DataFrame({"fold": [1], "mase_scale": [np.nan]})).iloc[0]
    assert row["MAE"] == pytest.approx(0.5) and row["RMSE"] == pytest.approx(np.sqrt(0.5))
    assert np.isnan(row["MAPE"]) and row["mape_n_excluded"] == 2
    assert np.isnan(row["WAPE"]) and np.isnan(row["MASE"])


def test_a_group_with_a_failed_or_missing_prediction_is_never_scored():
    table = pd.DataFrame(
        {
            "model": ["A", "A", "B", "B"],
            "fold": 1,
            "actual": [1.0, 2.0, 1.0, 2.0],
            "predicted": [1.0, 2.0, np.nan, 2.0],
            "success": [True, True, False, False],
        }
    )
    result = metrics_by_group(table, ["model", "fold"]).set_index("model")
    assert bool(result.loc["A", "success"]) and result.loc["A", "MAE"] == 0.0
    assert not bool(result.loc["B", "success"]) and np.isnan(result.loc["B", "MAE"]) and np.isnan(result.loc["B", "RMSE"])


def test_metrics_by_group_validates_its_inputs():
    table = _hand_table()
    with pytest.raises(ArtifactError, match="missing column"):
        metrics_by_group(table, ["model", "no_such_column"])
    with pytest.raises(ValueError, match="mase_scale"):
        metrics_by_group(table, ["model", "fold"], pd.DataFrame({"fold": [1]}))
    with pytest.raises(ValueError, match="not among the group columns"):
        metrics_by_group(table, ["model"], pd.DataFrame({"fold": [1], "mase_scale": [1.0]}))


def test_item_level_mase_scales_are_looked_up_per_item_and_fold():
    table = pd.DataFrame(
        {
            "item_id": ["A", "A", "B", "B"],
            "fold": [1, 1, 1, 1],
            "actual": [1.0, 3.0, 1.0, 3.0],
            "predicted": [2.0, 3.0, 1.0, 1.0],
            "success": True,
        }
    )
    scales = pd.DataFrame({"item_id": ["A", "B"], "fold": [1, 1], "mase_scale": [0.5, 2.0]})
    result = metrics_by_group(table, ["item_id", "fold"], scales).set_index("item_id")
    assert result.loc["A", "MASE"] == pytest.approx(0.5 / 0.5)
    assert result.loc["B", "MASE"] == pytest.approx(1.0 / 2.0)


# --------------------------------------------------------------------------- horizon handling

def test_truncate_horizon_keeps_the_first_days_of_every_window(dev_series):
    table = _aggregate_table(dev_series)
    short = truncate_horizon(table, 7, ["model"])
    assert len(short) == 7 * FOLDS and short["horizon_step"].max() == 7
    first_days = expected_fold_windows(dev_series.index, HORIZON, FOLDS)["forecast_start"].tolist()
    assert sorted(short.groupby("fold")["date"].min().tolist()) == sorted(first_days)
    assert len(truncate_horizon(table, 28, ["model"])) == len(table)
    assert len(truncate_horizon(table, 99, ["model"])) == len(table)
    with pytest.raises(ValueError):
        truncate_horizon(table, 0, ["model"])


def test_add_horizon_step_counts_from_one_within_each_window(dev_series):
    stepped = add_horizon_step(_aggregate_table(dev_series), ["model"])
    assert stepped.groupby("fold")["horizon_step"].apply(lambda s: s.tolist() == list(range(1, HORIZON + 1))).all()


# --------------------------------------------------------------------------- real project artifacts

REAL_AGGREGATE = [
    "seasonal_naive_backtest_forecasts.csv",
    "arima_backtest_forecasts.csv",
    "sarima_backtest_forecasts.csv",
    "holt_winters_backtest_forecasts.csv",
]


@pytest.fixture(scope="module")
def real_context():
    from src.data.aggregate_series import aggregate_series_path, load_aggregate_development

    config = load_config()
    if not aggregate_series_path(config).exists():
        pytest.skip("the processed aggregate parquet file is not present")
    series, _ = load_aggregate_development(config)
    return config, series


@pytest.mark.parametrize("filename", REAL_AGGREGATE)
def test_completed_phase2_aggregate_forecasts_pass_the_audit(real_context, filename):
    config, series = real_context
    path = forecasts_dir(config) / filename
    if not path.exists():
        pytest.skip(f"{filename} not present")
    table = read_forecast_table(path, "aggregate")
    assert _failed(audit_forecast_table(table, kind="aggregate", dev_series=series)) == set()


@pytest.mark.parametrize("stem", ["seasonal_naive", "arima", "sarima", "holt_winters"])
def test_recomputed_metrics_reproduce_the_saved_phase2_fold_metrics(real_context, stem):
    config, series = real_context
    fpath = forecasts_dir(config) / f"{stem}_backtest_forecasts.csv"
    mpath = metrics_dir(config) / f"{stem}_backtest_metrics.csv"
    if not (fpath.exists() and mpath.exists()):
        pytest.skip(f"{stem} artifacts not present")
    windows = expected_fold_windows(series.index, 28, 5)
    scales = fold_training_scales(series, windows, 7)
    recomputed = metrics_by_group(
        read_forecast_table(fpath, "aggregate"), ["model", "fold"],
        pd.DataFrame({"fold": list(scales), "mase_scale": list(scales.values())}),
    ).sort_values("fold")
    saved = pd.read_csv(mpath).sort_values("fold")
    np.testing.assert_allclose(recomputed["MAE"].to_numpy(), saved["MAE"].to_numpy(), rtol=0, atol=1e-8)
    np.testing.assert_allclose(recomputed["RMSE"].to_numpy(), saved["RMSE"].to_numpy(), rtol=0, atol=1e-8)
    np.testing.assert_array_equal(recomputed["mape_n_excluded"].to_numpy(), saved["mape_n_excluded"].to_numpy())


@pytest.mark.parametrize("filename", ["item_level_backtest_long.csv", "xgboost_item_level_backtest_long.csv"])
def test_completed_item_level_forecasts_pass_the_audit(real_context, filename):
    from src.evaluation.forecast_artifacts import load_selected_item_series

    config, series = real_context
    path = forecasts_dir(config) / filename
    if not path.exists():
        pytest.skip(f"{filename} not present")
    table = read_forecast_table(path, "item")
    results = audit_forecast_table(table, kind="item", item_series=load_selected_item_series(config), dev_dates=series.index)
    assert _failed(results) == set()
    assert table["item_id"].nunique() == 20


# --------------------------------------------------------------------------- writing

def _fake_results(dev_series, fail_fold=None):
    from src.evaluation.backtesting import BacktestFoldResult

    windows = expected_fold_windows(dev_series.index, HORIZON, FOLDS)
    results = []
    for w in windows.itertuples(index=False):
        days = pd.date_range(w.forecast_start, periods=HORIZON, freq="D")
        failed = w.fold == fail_fold
        results.append(
            BacktestFoldResult(
                model_name="Fake", fold=int(w.fold), origin_date=w.origin_date, forecast_start=days[0],
                forecast_end=days[-1], dates=days, actuals=dev_series.loc[days].to_numpy(),
                predictions=None if failed else dev_series.loc[days].to_numpy() + 2.0,
                success=not failed, error="RuntimeError: boom" if failed else None,
            )
        )
    return results


def test_backtest_results_become_a_valid_forecast_table(tmp_path, dev_series):
    from src.evaluation.forecast_artifacts import FORECAST_TABLE_COLUMNS, backtest_results_to_forecast_table, write_forecast_table

    table = backtest_results_to_forecast_table(_fake_results(dev_series))
    assert list(table.columns) == list(FORECAST_TABLE_COLUMNS) and len(table) == HORIZON * FOLDS
    path = write_forecast_table(table, tmp_path / "sub" / "out.csv")
    first_start = expected_fold_windows(dev_series.index, HORIZON, FOLDS)["forecast_start"].iloc[0]
    lines = path.read_text(encoding="utf-8").splitlines()
    assert lines[0] == ",".join(FORECAST_TABLE_COLUMNS)
    assert lines[1].split(",")[2] == first_start.strftime("%Y-%m-%d")          # ISO date, no time part
    reloaded = read_forecast_table(path, "aggregate")
    assert _failed(audit_forecast_table(reloaded, kind="aggregate", dev_series=dev_series)) == set()
    np.testing.assert_allclose(reloaded["predicted"] - reloaded["actual"], 2.0)
    assert "\r" not in path.read_text(encoding="utf-8")


def test_a_failed_fold_stays_in_the_table_without_a_substitute_forecast(tmp_path, dev_series):
    from src.evaluation.forecast_artifacts import backtest_results_to_forecast_table, write_forecast_table

    table = backtest_results_to_forecast_table(_fake_results(dev_series, fail_fold=3))
    failed = table[table["fold"] == 3]
    assert len(failed) == HORIZON and failed["predicted"].isna().all()
    assert not failed["success"].any() and (failed["error"] == "RuntimeError: boom").all()
    assert table[table["fold"] != 3]["success"].all()
    reloaded = read_forecast_table(write_forecast_table(table, tmp_path / "f.csv"), "aggregate")
    metrics = metrics_by_group(reloaded, ["model", "fold"]).set_index("fold")
    assert not bool(metrics.loc[3, "success"]) and np.isnan(metrics.loc[3, "MAE"])
    assert bool(metrics.loc[2, "success"]) and metrics.loc[2, "MAE"] == pytest.approx(2.0)


def test_a_prediction_count_that_does_not_match_the_dates_is_rejected(dev_series):
    from src.evaluation.forecast_artifacts import backtest_results_to_forecast_table

    results = _fake_results(dev_series)
    results[0].predictions = results[0].predictions[:-1]
    with pytest.raises(ArtifactError, match="predictions for"):
        backtest_results_to_forecast_table(results)
