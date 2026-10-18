"""Cross-model backtest comparison (Phase 2 Step 5).

Combines the per-fold summary tables that
`src.evaluation.backtesting.summarize_backtest` already produces for
each model (Seasonal Naive, ARIMA, SARIMA, Holt-Winters) into one table,
reduces it to a per-model mean/std view, and plots the mean fold-level
MAE/RMSE comparison the Step 5 brief asks for.

Per-demand-category model comparison -- this module's original mandate
-- is a separate, later feature: Step 5 compares models on the single
CA_1/FOODS_3 aggregate series only, the same series every other Phase 2
step has used.
"""

from __future__ import annotations

from typing import List, Optional

import numpy as np
import pandas as pd

from src.utils.logging import get_logger

logger = get_logger(__name__)


def combine_backtest_summaries(summaries: List[pd.DataFrame]) -> pd.DataFrame:
    """Stack several models' `summarize_backtest()` tables into one long table."""
    if not summaries:
        raise ValueError("combine_backtest_summaries() needs at least one summary table.")
    return pd.concat(summaries, ignore_index=True)


def summarize_model_means(combined: pd.DataFrame) -> pd.DataFrame:
    """Per-model mean/std MAE & RMSE, MAPE-with-caveat, and fold success counts.

    MAE/RMSE means and stds are computed only over folds where that
    model succeeded (`success == True`); a failed fold contributes
    nothing to them (its MAE/RMSE are already NaN from
    `summarize_backtest`). MAPE is averaged only over successful folds
    that additionally had a defined MAPE (not all-zero actuals).
    `n_folds`/`n_success`/`n_failed` are reported alongside so a reader
    can see how many of the folds each model's numbers actually rest on
    before comparing models.
    """
    required = {"model", "MAE", "RMSE", "MAPE", "success"}
    missing = required - set(combined.columns)
    if missing:
        raise ValueError(f"combined table is missing column(s): {sorted(missing)}")

    rows = []
    for model_name, grp in combined.groupby("model", sort=False):
        ok = grp[grp["success"].astype(bool)]
        n_folds = len(grp)
        n_success = len(ok)
        n_failed = n_folds - n_success
        mape_valid = ok["MAPE"].dropna()
        rows.append(
            {
                "model": model_name,
                "n_folds": n_folds,
                "n_success": n_success,
                "n_failed": n_failed,
                "MAE_mean": ok["MAE"].mean() if n_success else np.nan,
                "MAE_std": ok["MAE"].std() if n_success else np.nan,
                "RMSE_mean": ok["RMSE"].mean() if n_success else np.nan,
                "RMSE_std": ok["RMSE"].std() if n_success else np.nan,
                "MAPE_mean": mape_valid.mean() if len(mape_valid) else np.nan,
                "MAPE_n_folds_valid": int(len(mape_valid)),
            }
        )
    return pd.DataFrame(rows)


def plot_model_comparison(model_means: pd.DataFrame, ax=None):
    """Bar plot of each model's mean fold-level MAE and RMSE, with std-dev error bars.

    Monochrome by design (this project's established plot style):
    MAE in dark gray, RMSE in light gray, black error bars -- color is
    not what distinguishes the two series here, position/label is.
    """
    import matplotlib.pyplot as plt

    if ax is None:
        _, ax = plt.subplots(figsize=(9, 5))

    models = model_means["model"].tolist()
    x = np.arange(len(models))
    width = 0.35

    ax.bar(x - width / 2, model_means["MAE_mean"], width, label="MAE", color="#4C4C4C")
    ax.bar(x + width / 2, model_means["RMSE_mean"], width, label="RMSE", color="#B0B0B0")
    ax.errorbar(
        x - width / 2,
        model_means["MAE_mean"],
        yerr=model_means["MAE_std"].fillna(0.0),
        fmt="none",
        ecolor="black",
        capsize=3,
    )
    ax.errorbar(
        x + width / 2,
        model_means["RMSE_mean"],
        yerr=model_means["RMSE_std"].fillna(0.0),
        fmt="none",
        ecolor="black",
        capsize=3,
    )
    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=20, ha="right")
    ax.set_ylabel("Error (units)")
    ax.set_title("Mean fold-level MAE / RMSE by model (development backtesting, ±1 std)")
    ax.legend()
    ax.figure.tight_layout()
    return ax
