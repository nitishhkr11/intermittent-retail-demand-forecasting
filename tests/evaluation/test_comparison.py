"""Unit tests for src/evaluation/comparison.py (Phase 2 Step 5 model comparison)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.evaluation.comparison import (
    combine_backtest_summaries,
    plot_model_comparison,
    summarize_model_means,
)


def _summary(model: str, maes, rmses, mapes, success=None) -> pd.DataFrame:
    n = len(maes)
    if success is None:
        success = [True] * n
    return pd.DataFrame(
        {
            "model": [model] * n,
            "fold": list(range(1, n + 1)),
            "forecast_start": [pd.Timestamp("2020-01-01")] * n,
            "forecast_end": [pd.Timestamp("2020-01-28")] * n,
            "MAE": maes,
            "RMSE": rmses,
            "MAPE": mapes,
            "mape_n_excluded": [0] * n,
            "n": [28] * n,
            "success": success,
            "error": [None] * n,
        }
    )


def test_combine_backtest_summaries_concatenates_all_rows():
    a = _summary("ModelA", [10, 20], [15, 25], [5, 6])
    b = _summary("ModelB", [30, 40], [35, 45], [7, 8])
    combined = combine_backtest_summaries([a, b])
    assert len(combined) == 4
    assert set(combined["model"]) == {"ModelA", "ModelB"}


def test_combine_backtest_summaries_raises_on_empty_list():
    with pytest.raises(ValueError):
        combine_backtest_summaries([])


def test_summarize_model_means_basic_mean_and_std():
    a = _summary("ModelA", [10.0, 20.0], [15.0, 25.0], [5.0, 7.0])
    combined = combine_backtest_summaries([a])
    means = summarize_model_means(combined)
    assert len(means) == 1
    row = means.iloc[0]
    assert row["model"] == "ModelA"
    assert row["n_folds"] == 2
    assert row["n_success"] == 2
    assert row["n_failed"] == 0
    assert row["MAE_mean"] == pytest.approx(15.0)
    assert row["RMSE_mean"] == pytest.approx(20.0)
    assert row["MAPE_mean"] == pytest.approx(6.0)
    assert row["MAPE_n_folds_valid"] == 2


def test_summarize_model_means_excludes_failed_folds_from_mean():
    # One real fold (MAE=10) and one failed fold (NaN) -- the mean must
    # be exactly 10, not NaN and not averaged-with-zero.
    summary = _summary(
        "ModelA",
        maes=[10.0, np.nan],
        rmses=[15.0, np.nan],
        mapes=[5.0, np.nan],
        success=[True, False],
    )
    combined = combine_backtest_summaries([summary])
    means = summarize_model_means(combined)
    row = means.iloc[0]
    assert row["n_success"] == 1
    assert row["n_failed"] == 1
    assert row["MAE_mean"] == pytest.approx(10.0)
    assert row["RMSE_mean"] == pytest.approx(15.0)


def test_summarize_model_means_all_failed_gives_nan_not_crash():
    summary = _summary("ModelA", [np.nan, np.nan], [np.nan, np.nan], [np.nan, np.nan], success=[False, False])
    combined = combine_backtest_summaries([summary])
    means = summarize_model_means(combined)
    row = means.iloc[0]
    assert row["n_success"] == 0
    assert np.isnan(row["MAE_mean"])
    assert np.isnan(row["RMSE_mean"])
    assert row["MAPE_n_folds_valid"] == 0


def test_summarize_model_means_mapes_with_some_undefined_folds():
    # A fold can succeed (valid MAE/RMSE) but still have an undefined
    # MAPE (all-zero actuals that fold) -- MAPE_mean must average only
    # over the folds where it's defined, matching mape()'s own contract.
    summary = _summary("ModelA", [10.0, 20.0], [15.0, 25.0], [5.0, np.nan])
    combined = combine_backtest_summaries([summary])
    means = summarize_model_means(combined)
    row = means.iloc[0]
    assert row["MAPE_mean"] == pytest.approx(5.0)
    assert row["MAPE_n_folds_valid"] == 1


def test_summarize_model_means_multiple_models_preserve_order_independence():
    a = _summary("ModelA", [10.0], [15.0], [5.0])
    b = _summary("ModelB", [1.0], [2.0], [3.0])
    combined = combine_backtest_summaries([a, b])
    means = summarize_model_means(combined)
    assert set(means["model"]) == {"ModelA", "ModelB"}
    a_row = means[means["model"] == "ModelA"].iloc[0]
    b_row = means[means["model"] == "ModelB"].iloc[0]
    assert a_row["MAE_mean"] == pytest.approx(10.0)
    assert b_row["MAE_mean"] == pytest.approx(1.0)


def test_plot_model_comparison_runs_and_returns_axes():
    a = _summary("ModelA", [10.0, 20.0], [15.0, 25.0], [5.0, 6.0])
    b = _summary("ModelB", [12.0, 18.0], [16.0, 24.0], [5.5, 6.5])
    means = summarize_model_means(combine_backtest_summaries([a, b]))
    ax = plot_model_comparison(means)
    assert ax is not None
    assert len(ax.get_xticks()) == 2
