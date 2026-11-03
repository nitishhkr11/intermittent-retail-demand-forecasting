"""Tests for src/evaluation/model_comparison.py on synthetic saved tables."""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import pytest  # noqa: E402

from src.evaluation import model_comparison as mc  # noqa: E402
from src.evaluation.forecast_artifacts import ArtifactError, write_forecast_table  # noqa: E402
from src.utils.config import load_config  # noqa: E402

H, F = 7, 3
N = 80
DATES = pd.date_range("2020-01-01", periods=N, freq="D")
rng = np.random.default_rng(0)
DEV = pd.Series(100 + 20 * np.sin(np.arange(N) * 2 * np.pi / 7) + rng.normal(0, 3, N), index=DATES)


def fold_dates(fold):
    test_end = N - (F - fold) * H
    return DATES[test_end - H : test_end]


def agg_table(models, offsets, noise=0.0, fail_fold=None):
    rows = []
    for model, offset in zip(models, offsets):
        for fold in range(1, F + 1):
            for d in fold_dates(fold):
                ok = fail_fold != (model, fold)
                rows.append({"model": model, "fold": fold, "date": d, "actual": DEV[d],
                             "predicted": DEV[d] + offset if ok else np.nan, "success": ok, "error": "" if ok else "boom"})
    return pd.DataFrame(rows)


def cfg(tmp_path):
    base = load_config()
    base = {k: (dict(v) if isinstance(v, dict) else v) for k, v in base.items()}
    base["phase3"]["forecasts_dir"] = str(tmp_path)
    return base


# ------------------------------------------------------------------------------------------ registry
def test_sources_come_from_config_and_cover_every_model_family():
    config = load_config()
    agg, items = mc.aggregate_sources(config), mc.item_sources(config)
    assert [s.scope for s in agg] == ["aggregate"] * 6 and [s.scope for s in items] == ["item"] * 3
    names = {s.filename for s in agg}
    assert config["phase3"]["neural_forecasts_filename"] in names and config["phase3"]["prophet_forecasts_filename"] in names
    assert config["phase2"]["arima_forecasts_filename"] in names
    assert any("07_lstm_gru.ipynb" in s.hint for s in agg) and any("06_xgboost.ipynb" in s.hint for s in items)


def test_model_family_prefixes():
    assert mc.model_family("SARIMA(6, 1, 1)x(0, 0, 2, 7)") == "classical (Phase 2)"
    assert mc.model_family("LSTM(calendar)") == "neural (Phase 3)"
    assert mc.model_family("Prophet(additive)") == "Prophet (Phase 3)"
    assert mc.model_family("XGBoost(pooled)", "item") == "XGBoost (Phase 3)"
    assert mc.model_family("Croston(alpha=0.1)", "item") == "baseline (Phase 2)"
    assert mc.model_family("Mystery") == "other"


# ------------------------------------------------------------------------------------------ loading and audit
def test_load_sources_reports_ok_missing_invalid_and_failed_audit(tmp_path):
    config = cfg(tmp_path)
    write_forecast_table(agg_table(["SeasonalNaive"], [0.0]), tmp_path / "good.csv")
    pd.DataFrame({"model": ["X"], "fold": [1]}).to_csv(tmp_path / "invalid.csv", index=False)
    wrong = agg_table(["ARIMA"], [1.0])
    wrong["actual"] = wrong["actual"] + 5  # actuals differ from the development series
    write_forecast_table(wrong, tmp_path / "wrong.csv")
    sources = [
        mc.ArtifactSource("good", "good.csv", "aggregate"),
        mc.ArtifactSource("absent", "prophet_backtest_forecasts.csv", "aggregate"),
        mc.ArtifactSource("invalid", "invalid.csv", "aggregate"),
        mc.ArtifactSource("wrong", "wrong.csv", "aggregate"),
    ]
    tables, audit = mc.load_sources(sources, dev_series=DEV, config=config, horizon=H, n_folds=F)
    status = dict(zip(audit["artifact"], audit["status"]))
    assert status == {"good.csv": "ok", "prophet_backtest_forecasts.csv": "missing", "invalid.csv": "invalid", "wrong.csv": "audit failed"}
    assert list(tables) == ["good.csv"]  # an artifact that failed its audit is never scored
    missing = audit[audit["status"] == "missing"]["detail"].iloc[0]
    assert "08_prophet.ipynb" in missing and "python -m jupyter nbconvert" in missing
    failed = audit[audit["status"] == "audit failed"].iloc[0]
    assert failed["failed_checks"] >= 1 and "actuals_equal_development_series" in failed["detail"]
    assert list(audit.columns) == mc.AUDIT_COLUMNS
    assert audit[audit["status"] == "ok"]["rows"].iloc[0] == F * H


def test_load_sources_never_fills_a_missing_artifact(tmp_path):
    tables, audit = mc.load_sources([mc.ArtifactSource("x", "lstm_gru_backtest_forecasts.csv", "aggregate")],
                                    dev_series=DEV, config=cfg(tmp_path), horizon=H, n_folds=F)
    assert tables == {} and audit["status"].tolist() == ["missing"] and "07_lstm_gru.ipynb" in audit["detail"].iloc[0]


# ------------------------------------------------------------------------------------------ combining
def test_combine_stacks_models_on_one_grid():
    a, b = agg_table(["M1"], [0.0]), agg_table(["M2"], [3.0])
    combined, notes = mc.combine_tables({"a.csv": a, "b.csv": b}, "aggregate")
    assert sorted(combined["model"].unique()) == ["M1", "M2"] and notes == [] and len(combined) == 2 * F * H


def test_combine_counts_an_identical_rerun_once_and_says_so():
    a, b = agg_table(["M1", "M2"], [0.0, 2.0]), agg_table(["M2"], [2.0])
    combined, notes = mc.combine_tables({"first.csv": a, "second.csv": b}, "aggregate")
    assert len(combined) == 2 * F * H and len(notes) == 1 and "identical" in notes[0] and "first.csv" in notes[0]


def test_combine_refuses_the_same_model_name_with_different_forecasts():
    with pytest.raises(ArtifactError, match="DIFFERENT forecasts"):
        mc.combine_tables({"a": agg_table(["M"], [0.0]), "b": agg_table(["M"], [1.0])}, "aggregate")


def test_combine_refuses_tables_scored_on_different_cells_or_actuals():
    a = agg_table(["M1"], [0.0])
    shifted = agg_table(["M2"], [0.0])
    shifted["actual"] = shifted["actual"] + 1
    with pytest.raises(ArtifactError):
        mc.combine_tables({"a": a, "b": shifted}, "aggregate")
    truncated = agg_table(["M3"], [0.0]).iloc[:-3]
    with pytest.raises(ArtifactError):
        mc.combine_tables({"a": a, "c": truncated}, "aggregate")


def test_aggregate_and_item_level_tables_are_never_mixed():
    item = agg_table(["M"], [0.0])
    item["item_id"] = "I1"
    item["demand_class"] = "Smooth"
    with pytest.raises(ArtifactError, match="never combined"):
        mc.combine_tables({"item": item}, "aggregate")
    with pytest.raises(ArtifactError, match="no item_id"):
        mc.combine_tables({"agg": agg_table(["M"], [0.0])}, "item")
    with pytest.raises(ValueError):
        mc.combine_tables({}, "aggregate")
    with pytest.raises(ValueError, match="scope"):
        mc.combine_tables({"x": agg_table(["M"], [0.0])}, "both")


# ------------------------------------------------------------------------------------------ metrics
def test_aggregate_fold_metrics_known_values():
    table = agg_table(["Perfect", "Plus5"], [0.0, 5.0])
    metrics = mc.aggregate_fold_metrics(table, DEV, H, F)
    perfect = metrics[metrics["model"] == "Perfect"]
    plus = metrics[metrics["model"] == "Plus5"]
    assert (perfect["MAE"] == 0).all() and (perfect["RMSE"] == 0).all() and (perfect["WAPE"] == 0).all()
    assert np.allclose(plus["MAE"], 5.0) and np.allclose(plus["RMSE"], 5.0) and np.allclose(plus["bias"], 5.0)
    assert (plus["MASE"] > 0).all() and (perfect["MASE"] == 0).all()
    assert set(metrics["fold"]) == {1, 2, 3} and metrics["success"].all()
    assert metrics["family"].eq("other").all() and not metrics["primary"].any()


def test_mase_uses_the_training_window_scale_of_each_fold():
    table = agg_table(["M"], [4.0])
    metrics = mc.aggregate_fold_metrics(table, DEV, H, F).sort_values("fold")
    for row in metrics.itertuples():
        train_end = N - (F - row.fold + 1) * H
        values = DEV.to_numpy()[:train_end]
        scale = np.mean(np.abs(values[7:] - values[:-7]))
        assert row.MASE == pytest.approx(4.0 / scale)


def test_failed_fold_gets_nan_metrics_and_success_false_not_a_partial_score():
    table = agg_table(["M"], [2.0], fail_fold=("M", 2))
    metrics = mc.aggregate_fold_metrics(table, DEV, H, F)
    failed = metrics[metrics["fold"] == 2].iloc[0]
    assert not failed["success"] and np.isnan(failed["MAE"]) and np.isnan(failed["MASE"])


def test_model_summary_mean_std_range_and_failed_folds():
    table = agg_table(["A", "B"], [1.0, 3.0], fail_fold=("B", 3))
    folds = mc.aggregate_fold_metrics(table, DEV, H, F)
    summary = mc.model_summary(folds).set_index("model")
    assert summary.loc["A", "folds_scored"] == 3 and summary.loc["A", "folds_failed"] == 0
    assert summary.loc["A", "MAE_mean"] == pytest.approx(1.0) and summary.loc["A", "MAE_std"] == pytest.approx(0.0)
    assert summary.loc["B", "folds_scored"] == 2 and summary.loc["B", "folds_failed"] == 1
    assert summary.loc["B", "MAE_mean"] == pytest.approx(3.0)
    assert summary.loc["A", "MAE_min"] <= summary.loc["A", "MAE_mean"] <= summary.loc["A", "MAE_max"]
    assert summary.loc["A", "mape_zero_days_excluded"] == 0
    with pytest.raises(ValueError, match="missing column"):
        mc.model_summary(folds.drop(columns="MAE"))


def test_model_summary_std_uses_the_sample_formula_and_counts_zero_days():
    folds = pd.DataFrame(
        {"model": ["M"] * 3, "family": ["other"] * 3, "fold": [1, 2, 3], "success": [True] * 3,
         "MAE": [1.0, 2.0, 6.0], "RMSE": [1.0, 2.0, 6.0], "MAPE": [np.nan, 10.0, 20.0], "WAPE": [0.1, 0.2, 0.3],
         "MASE": [np.nan] * 3, "bias": [0.0] * 3, "mape_n_excluded": [1, 0, 2]}
    )
    row = mc.model_summary(folds).iloc[0]
    assert row["MAE_std"] == pytest.approx(np.std([1.0, 2.0, 6.0], ddof=1))
    assert row["MAPE_mean"] == 15.0 and row["mape_zero_days_excluded"] == 3 and np.isnan(row["MASE_mean"])


def test_fold_wins_counts_folds_and_ignores_failed_ones():
    table = agg_table(["Ref", "Better", "Worse"], [4.0, 1.0, 9.0], fail_fold=("Better", 2))
    folds = mc.aggregate_fold_metrics(table, DEV, H, F)
    wins = mc.fold_wins(folds, "Ref").set_index("model")
    assert wins.loc["Better", "folds_compared"] == 2 and wins.loc["Better", "folds_lower"] == 2
    assert wins.loc["Better", "mean_difference"] == pytest.approx(-3.0)
    assert wins.loc["Worse", "folds_lower"] == 0 and wins.loc["Worse", "mean_difference"] == pytest.approx(5.0)
    assert "Ref" not in wins.index
    with pytest.raises(ValueError, match="not in the table"):
        mc.fold_wins(folds, "Nope")


def test_lowest_by_metric_handles_undefined_metrics():
    summary = pd.DataFrame({"model": ["A", "B"], "MAE_mean": [2.0, 1.0], "RMSE_mean": [1.0, 2.0],
                            "WAPE_mean": [np.nan, np.nan], "MASE_mean": [0.5, 0.7]})
    out = mc.lowest_by_metric(summary).set_index("metric")
    assert out.loc["MAE", "lowest_model"] == "B" and out.loc["RMSE", "lowest_model"] == "A"
    assert out.loc["WAPE", "lowest_model"] == "n/a" and np.isnan(out.loc["WAPE", "value"])


def item_table():
    rows = []
    for item, cls, level in (("I1", "Smooth", 5.0), ("I2", "Lumpy", 0.5)):
        for model, off in (("SeasonalNaive", 1.0), ("TSB(alpha=0.1,beta=0.1)", 0.2)):
            for fold in range(1, F + 1):
                for d in fold_dates(fold):
                    actual = level + (d.day % 3)
                    rows.append({"item_id": item, "demand_class": cls, "model": model, "fold": fold, "date": d,
                                 "actual": actual, "predicted": max(actual + off, 0.0), "success": True, "error": ""})
    return pd.DataFrame(rows)


def test_item_level_metrics_and_class_summary_stay_in_item_units():
    table = item_table()
    # the MASE scale needs each item's training history: a repeating pattern over the whole calendar
    full = {i: pd.Series(((np.arange(N) % 3) + (5.0 if i == "I1" else 0.5)), index=DATES) for i in ("I1", "I2")}
    folds = mc.item_fold_metrics(table, full, H, F)
    assert set(folds["family"]) == {"baseline (Phase 2)"} and len(folds) == 2 * 2 * F
    summary = mc.item_model_class_summary(folds)
    assert set(summary["demand_class"]) == {"Smooth", "Lumpy"}
    sn = summary[(summary["model"] == "SeasonalNaive") & (summary["demand_class"] == "Smooth")].iloc[0]
    assert sn["MAE_macro_mean"] == pytest.approx(1.0) and sn["n_items"] == 1
    assert sn["MAE_macro_mean"] < 100  # item-level scale, nothing like the aggregate numbers


# ------------------------------------------------------------------------------------------ plots
def test_plots_build_for_the_aggregate_comparison():
    table = agg_table(["SeasonalNaive", "LSTM(calendar)", "Prophet(additive)", "GRU(univariate)"], [3.0, 1.0, 2.0, 4.0])
    folds = mc.aggregate_fold_metrics(table, DEV, H, F)
    summary = mc.model_summary(folds)
    fig1 = mc.plot_aggregate_comparison(summary)
    fig2 = mc.plot_fold_variation(folds)
    hatches = {p.get_hatch() for p in fig1.axes[0].patches if p.get_hatch()}
    assert len(hatches) >= 2  # families are told apart by hatch, not colour alone
    assert len(fig2.axes[0].lines) == 4
    assert {line.get_marker() for line in fig2.axes[0].lines} >= {"o", "s"}
    plt.close(fig1)
    plt.close(fig2)
