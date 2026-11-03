"""Page 3 - Model Comparison: aggregate-series and item-level results side by side but never ranked together."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
APP_DIR = Path(__file__).resolve().parents[1]
for _p in (str(ROOT), str(APP_DIR)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import streamlit as st  # noqa: E402

import data_access as da  # noqa: E402
from components import charts  # noqa: E402
from components.cards import badge, callout, empty_state, loaded_error, page_header, provenance_panel  # noqa: E402
from components.tables import render_table  # noqa: E402
from src.evaluation.forecast_artifacts import expected_fold_windows, forecasts_dir, notebook_command, producer_notebook  # noqa: E402
from src.evaluation.model_comparison import aggregate_sources, item_sources, lowest_by_metric, model_summary  # noqa: E402

page_header("Model Comparison", "Development-fold results for every model with saved forecasts. Metrics are recomputed from the saved forecast tables.")
callout("warn", "Aggregate and item-level errors are not comparable",
        "The aggregate series adds up all 823 items (thousands of units per day). Item-level errors are for single items (a few units per day). "
        "They are shown in separate tabs and are never ranked against each other.")

config = da.get_config()
bundle = da.load_bundle()
if not bundle.ok:
    loaded_error(bundle, "Saved forecasts are not available")
    st.stop()
b = bundle.value
METRICS = {"MAE": "MAE", "RMSE": "RMSE", "WAPE": "WAPE", "MASE": "MASE", "bias": "bias"}
SUMMARY_COLUMNS = ["MAE", "RMSE", "WAPE", "MASE"]
horizon, n_folds = int(config["backtest"]["horizon"]), int(config["backtest"]["n_folds"])
windows = expected_fold_windows(b.dev_series.index, horizon, n_folds)

st.markdown("## Evaluation window")
fold_table = windows.assign(forecast_window=lambda d: d["forecast_start"].dt.strftime("%Y-%m-%d") + " → " + d["forecast_end"].dt.strftime("%Y-%m-%d"),
                            training_days=lambda d: d["n_train"], cutoff=lambda d: d["origin_date"].dt.strftime("%Y-%m-%d"))
render_table(fold_table[["fold", "cutoff", "training_days", "forecast_window"]].rename(columns={"fold": "Fold", "cutoff": "Training cutoff", "training_days": "Training days", "forecast_window": "Forecast window (28 days)"}),
             caption=f"Walk-forward backtest: {n_folds} folds, {horizon}-day horizon, expanding training window", max_height=None)
st.caption(f"All folds lie inside the development window ({b.window.dev_start.date()} to {b.window.dev_end.date()}). The reserved final {b.window.n_holdout_days} days "
           f"({b.window.holdout_start.date()} to {b.window.holdout_end.date()}) appear nowhere on this page.")

tab_agg, tab_item, tab_final = st.tabs(["Aggregate series", "Item level (20 items)", "Development vs final holdout"])

# ------------------------------------------------------------------------------------------------ aggregate
with tab_agg:
    metrics = da.load_metrics("aggregate")
    if not metrics.ok:
        loaded_error(metrics, "Aggregate-series results are not available")
    else:
        folds = metrics.value["folds"]
        summary = model_summary(folds)
        metric = st.selectbox("Metric to plot", list(METRICS), key="mc_agg_metric", help="MAE and RMSE are in units per day for the whole department.")
        charts.show(charts.comparison_bar_chart(summary, metric, scope="aggregate", title=f"Aggregate series: mean {metric} over {n_folds} folds", unit="units/day" if metric in ("MAE", "RMSE", "bias") else "ratio"))
        best = lowest_by_metric(summary)
        st.markdown(" ".join(badge("info", f"{r.metric}: lowest is {r.lowest_model}") for r in best.itertuples()), unsafe_allow_html=True)
        st.caption("'Lowest' describes these five folds only. Five overlapping-history folds are too few for a significance claim, and nothing here tests one.")
        show = summary[["model", "family", "folds_scored"] + [f"{m}_{s}" for m in SUMMARY_COLUMNS for s in ("mean", "std")] + ["MAPE_mean", "mape_zero_days_excluded"]]
        show = show.rename(columns={"model": "Model", "family": "Family", "folds_scored": "Folds scored", "MAPE_mean": "MAPE mean (%)", "mape_zero_days_excluded": "Zero-sales days left out of MAPE"})
        render_table(show, caption="Mean and standard deviation across folds (units per day for MAE/RMSE)", best={"MAE_mean": "min"}, download_name="aggregate_model_summary.csv", key="dl_agg_sum")
        charts.show(charts.fold_lines_chart(folds, metric, scope="aggregate", title=f"Aggregate series: {metric} by fold", unit="units/day" if metric in ("MAE", "RMSE", "bias") else "ratio"))
        failed = folds[~folds["success"].astype(bool)]
        if len(failed):
            callout("error", "Some model-fold fits failed", "; ".join(f"{r.model} fold {r.fold}" for r in failed.itertuples()) + ". They are excluded from the means and never replaced.")
        by_fold = folds[["model", "fold", "MAE", "RMSE", "WAPE", "MASE", "bias", "success"]].rename(columns={"model": "Model", "fold": "Fold"})
        render_table(by_fold, caption="Every model and fold (the values behind the chart)", download_name="aggregate_fold_metrics.csv", key="dl_agg_fold", max_height=380)
        st.markdown("**Definitions.** MAE = mean |actual − forecast|. RMSE = √mean (actual − forecast)². WAPE = Σ|actual − forecast| / Σ actual. MASE = MAE / mean |yₜ − yₜ₋₇| on the fold's training window. "
                    "bias = mean (forecast − actual); positive means over-forecasting. MAPE is shown only with zero-sales days excluded, and their count.")
    missing = b.audit[(b.audit["scope"] == "aggregate") & (b.audit["status"] != "ok")]
    if len(missing):
        empty_state("Some aggregate models are not in this comparison", "Their forecast files are missing or failed the audit.", sorted({notebook_command(producer_notebook(a) or "") for a in missing["artifact"]}), list(missing["artifact"]))

# ------------------------------------------------------------------------------------------------ item level
with tab_item:
    metrics = da.load_metrics("item")
    if not metrics.ok:
        loaded_error(metrics, "Item-level results are not available")
    else:
        item_folds = metrics.value["folds"]
        classes = ["All 20 items"] + sorted(item_folds["demand_class"].unique())
        cls = st.selectbox("Demand class", classes, key="mc_item_class", help="Averages are equal-weight over the items in the chosen class.")
        sub = item_folds if cls == classes[0] else item_folds[item_folds["demand_class"] == cls]
        macro = da.macro_item_fold_table(sub)
        n_items = int(sub["item_id"].nunique())
        summary = model_summary(macro)
        imetric = st.selectbox("Metric to plot", list(METRICS), key="mc_item_metric", help="Item-level MAE is in units per item per day.")
        charts.show(charts.comparison_bar_chart(summary, imetric, scope="item", title=f"Item level ({cls}, {n_items} items): mean {imetric} over {n_folds} folds", unit="units/item/day" if imetric in ("MAE", "RMSE", "bias") else "ratio"))
        best = lowest_by_metric(summary)
        st.markdown(" ".join(badge("info", f"{r.metric}: lowest is {r.lowest_model}") for r in best.itertuples()), unsafe_allow_html=True)
        st.caption("Different metrics can name different models. Intermittent items are often forecast best, on MAE, by predicting zero, "
                   "so read MAE and RMSE together and do not treat one winner as universal.")
        show = summary[["model", "family", "folds_scored"] + [f"{m}_{s}" for m in SUMMARY_COLUMNS for s in ("mean", "std")]].rename(columns={"model": "Model", "family": "Family", "folds_scored": "Folds scored"})
        render_table(show, caption=f"Equal-weight mean over {n_items} item(s), then mean and standard deviation across folds", best={"MAE_mean": "min"}, download_name="item_model_summary.csv", key="dl_item_sum")
        charts.show(charts.fold_lines_chart(macro, imetric, scope="item", title=f"Item level ({cls}): {imetric} by fold", unit="units/item/day" if imetric in ("MAE", "RMSE", "bias") else "ratio"))
        render_table(macro[["model", "fold", "MAE", "RMSE", "WAPE", "MASE", "bias", "success"]].rename(columns={"model": "Model", "fold": "Fold"}),
                     caption="Every model and fold (the values behind the chart)", download_name="item_fold_metrics.csv", key="dl_item_fold", max_height=380)
        st.caption("Demand classes here are the Phase 1 full-history labels of the 20 selected items. Notebook 05 also recomputes classes on each fold's own training window for the all-item experiment.")
    missing = b.audit[(b.audit["scope"] == "item") & (b.audit["status"] != "ok")]
    if len(missing):
        empty_state("Some item-level results are missing", "Their forecast files are missing or failed the audit.", sorted({notebook_command(producer_notebook(a) or "") for a in missing["artifact"]}), list(missing["artifact"]))

# ------------------------------------------------------------------------------------------------ final holdout
with tab_final:
    state = da.holdout_state()
    st.markdown("### Development results are not final results")
    st.markdown(
        f"Everything above comes from five walk-forward folds inside the development window. Model choices were made by looking at those folds, so they are "
        f"optimistic for any later period. The last {b.window.n_holdout_days} days ({b.window.holdout_start.date()} to {b.window.holdout_end.date()}) were set aside "
        "for one final evaluation of the frozen candidates, run once, with every candidate reported and not only the best."
    )
    if state["evaluated"]:
        callout("info", "The final holdout has been evaluated", "Results are in artifacts/final (final_holdout_results.csv) with a record of when and how often it ran.")
    else:
        callout("ok", "The final holdout has not been evaluated", "No final-holdout number exists in this project yet, and none is shown. "
                + ("The candidates are frozen; the evaluation is waiting for its explicit confirmation command." if state["frozen"] else "The candidates are not frozen yet."))
    st.markdown("Check the state, freeze the candidates, then run the one-time evaluation (PowerShell, project root):")
    from components.cards import code_block
    code_block("python -m src.evaluation.final_holdout status")
    code_block("python -m src.evaluation.final_holdout freeze")
    code_block("python -m src.evaluation.final_holdout run --confirm EVALUATE-FINAL-HOLDOUT-ONCE")

records = [{"role": s.label, **da.file_record(forecasts_dir(config) / s.filename)} for s in aggregate_sources(config) + item_sources(config)]
provenance_panel(records, "MAE, RMSE, WAPE, MASE and bias are recomputed here from the saved forecasts; MASE uses each fold's training-window lag-7 error as the scale.")
