"""Generate matched-sample comparisons and ablation plots from audited XGBoost outputs."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from src.utils.paths import resolve_path


ROOT = Path("artifacts/expanded_xgboost")
FIG = Path("reports/phase3")


def main() -> None:
    root, fig = resolve_path(ROOT), resolve_path(FIG)
    metrics = pd.read_csv(root / "item_fold_metrics.csv")
    manifest = json.loads((root / "fold_manifest.json").read_text(encoding="utf-8"))
    sample_ids = {str(row["item_id"]) for row in manifest["per_item_sample"]}
    pooled = metrics[metrics["model"].str.startswith("XGBoost(pooled")]
    pooled_summary = pooled.groupby("model").agg(
        n_items=("item_id", "nunique"), folds=("fold", "nunique"),
        item_fold_cells=("success", "size"), MAE=("MAE", "mean"), MAE_sd=("MAE", "std"),
        RMSE=("RMSE", "mean"), WAPE=("WAPE", "mean"), MASE=("MASE", "mean"),
        bias=("bias", "mean"),
    ).reset_index()
    matched = metrics[metrics["item_id"].astype(str).isin(sample_ids)]
    matched_summary = matched.groupby("model").agg(
        n_items=("item_id", "nunique"), folds=("fold", "nunique"),
        item_fold_cells=("success", "size"), MAE=("MAE", "mean"), MAE_sd=("MAE", "std"),
        RMSE=("RMSE", "mean"), WAPE=("WAPE", "mean"), MASE=("MASE", "mean"),
        bias=("bias", "mean"),
    ).reset_index()
    pooled_summary.to_csv(root / "pooled_model_summary.csv", index=False, lineterminator="\n")
    matched_summary.to_csv(root / "matched_sample_model_summary.csv", index=False, lineterminator="\n")

    fold = pooled.groupby(["model", "fold"], as_index=False).agg(MAE=("MAE", "mean"), RMSE=("RMSE", "mean"))
    pivot = fold.pivot(index="fold", columns="model", values="MAE")
    comparisons = [
        ("XGBoost(pooled,lags_calendar)", "XGBoost(pooled,lags_only)", "calendar vs lags only"),
        ("XGBoost(pooled,lags_rolling)", "XGBoost(pooled,lags_only)", "rolling vs lags only"),
        ("XGBoost(pooled,lags_calendar_events)", "XGBoost(pooled,lags_calendar)", "events vs calendar"),
        ("XGBoost(pooled,lags_calendar_events_poisson,poisson)", "XGBoost(pooled,lags_calendar_events)", "Poisson vs square error"),
    ]
    rows = []
    for candidate, reference, label in comparisons:
        delta = pivot[reference] - pivot[candidate]
        rows.append({"comparison": label, "candidate": candidate, "reference": reference,
                     "folds_candidate_lower_mae": int((delta > 0).sum()),
                     "mean_mae_reduction": float(delta.mean()), "sd_paired_fold_reduction": float(delta.std(ddof=1)),
                     "fold_reductions": ";".join(f"{int(k)}:{v:.6f}" for k, v in delta.items())})
    pd.DataFrame(rows).to_csv(root / "paired_fold_comparisons.csv", index=False, lineterminator="\n")
    fold.to_csv(root / "pooled_fold_metrics.csv", index=False, lineterminator="\n")

    fig.mkdir(parents=True, exist_ok=True)
    ordered = pooled_summary.sort_values("MAE")
    xlabels = [x.split(",", 1)[1].replace(")", "") for x in ordered["model"]]
    fig_obj, ax = plt.subplots(figsize=(10, 5.5), constrained_layout=True)
    ax.barh(xlabels[::-1], ordered["MAE"].to_numpy()[::-1], color="#2563EB")
    ax.set_xlabel("Mean item-fold MAE (units/item/day)")
    ax.set_title("Pooled XGBoost feature ablations · 823 products · 16 development folds")
    ax.grid(axis="x", alpha=.2)
    fig_obj.savefig(fig / "expanded_xgboost_pooled_ablation_mae.png", dpi=180)
    plt.close(fig_obj)

    fig_obj, ax = plt.subplots(figsize=(10, 5.5), constrained_layout=True)
    for model in pivot.columns:
        ax.plot(pivot.index, pivot[model], marker="o", linewidth=1.6, label=model.replace("XGBoost(pooled,", "").replace(")", ""))
    ax.set_xlabel("Shared seasonal fold")
    ax.set_ylabel("Mean item-fold MAE")
    ax.set_title("Pooled XGBoost fold stability · same products and forecast dates")
    ax.grid(alpha=.2)
    ax.legend(fontsize=7, ncol=2)
    fig_obj.savefig(fig / "expanded_xgboost_pooled_fold_stability.png", dpi=180)
    plt.close(fig_obj)


if __name__ == "__main__":
    main()
