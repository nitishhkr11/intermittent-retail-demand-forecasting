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
import pandas as pd  # noqa: E402

import data_access as da  # noqa: E402
from components import charts  # noqa: E402
from components.cards import badge, callout, empty_state, loaded_error, page_header, provenance_panel  # noqa: E402
from components.tables import render_table  # noqa: E402
from src.evaluation.forecast_artifacts import expected_fold_windows, forecasts_dir, notebook_command, producer_notebook  # noqa: E402
from src.evaluation.model_comparison import aggregate_sources, item_sources, lowest_by_metric, model_summary  # noqa: E402
from src.utils.paths import resolve_path  # noqa: E402

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
st.caption(f"Walk-forward folds lie inside the development window ({b.window.dev_start.date()} to {b.window.dev_end.date()}). The final {b.window.n_holdout_days} days "
           f"({b.window.holdout_start.date()} to {b.window.holdout_end.date()}) are reported separately in the final tab.")

tab_agg, tab_item, tab_expanded, tab_final = st.tabs(["Aggregate series", "Item level (20 items)", "Expanded development study", "Development vs final holdout"])

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
with tab_expanded:
    st.markdown("### Expanded seasonal study")
    st.caption("Separate, predeclared quarterly folds from 2012–2015. These results use saved development-period forecasts and are not pooled with the original five-fold comparison.")
    expanded_root = resolve_path(Path("artifacts/expanded_seasonal"))
    seasonal_summary_path = expanded_root / "model_summary.csv"
    if seasonal_summary_path.exists():
        seasonal = pd.read_csv(seasonal_summary_path)
        render_table(seasonal, caption="Aggregate-series models across 16 disjoint 28-day quarterly windows", download_name="expanded_seasonal_summary.csv", key="dl_expanded_seasonal")
        st.caption("Fold-local ARIMA, SARIMA and Holt-Winters settings were selected using only each fold's inner chronological validation segment. Means are descriptive; no significance test is reported.")
    else:
        empty_state("Expanded seasonal results are not available", "Run the saved seasonal development experiment.", ["python -m src.evaluation.expanded_seasonal run"], ["artifacts/expanded_seasonal/model_summary.csv"])

    neural_root = resolve_path(Path("artifacts/expanded_neural"))
    neural_summary_path = neural_root / "model_summary.csv"
    if neural_summary_path.exists():
        st.markdown("### LSTM/GRU seasonal comparison")
        st.caption("Four aggregate-series variants across all 16 quarterly folds with five seeds per fold. Calendar features improved both model families on these development windows; this is descriptive evidence for one series.")
        neural_summary = pd.read_csv(neural_summary_path)
        render_table(neural_summary, caption="Ensemble-fold errors and separate seed-level MAE; lower values are not a significance claim", download_name="expanded_neural_summary.csv", key="dl_expanded_neural")
        neural_folds_path = neural_root / "fold_metrics.csv"
        if neural_folds_path.exists():
            with st.expander("LSTM/GRU metrics by seasonal fold"):
                render_table(pd.read_csv(neural_folds_path), caption="All 64 variant-fold statuses and metrics", download_name="expanded_neural_fold_metrics.csv", key="dl_expanded_neural_folds", max_height=420)
        st.caption("Training used five fixed seeds per cell with fold-local scaling and validation-tail early stopping. No final-holdout values were used.")
    else:
        empty_state("Expanded neural results are not available", "Run the checkpointed 16-fold seasonal study.", ["python -m src.evaluation.expanded_neural run"], ["artifacts/expanded_neural/model_summary.csv"])

    prophet_root = resolve_path(Path("artifacts/expanded_prophet_block_conformal"))
    prophet_summary_path = prophet_root / "model_summary.csv"
    if prophet_summary_path.exists():
        st.markdown("### Prophet seasonal interval study")
        st.caption("Three aggregate-series Prophet variants across the same 16 seasonal folds. Coverage is measured at the scored day level; nominal 80% intervals under-covered in this development study.")
        prophet_summary = pd.read_csv(prophet_summary_path)
        render_table(prophet_summary, caption="Point-forecast errors, nominal interval coverage and mean interval width", download_name="expanded_prophet_summary.csv", key="dl_expanded_prophet")
        calibration_path = prophet_root / "interval_calibration_summary.csv"
        if calibration_path.exists():
            calibration = pd.read_csv(calibration_path)
            render_table(calibration, caption="Exploratory block calibration: folds 1–8 calibrate, folds 9–16 validate", download_name="prophet_block_calibration.csv", key="dl_prophet_calibration")
            st.caption("Block conformal intervals are deliberately wide and are an exploratory development result, not an operating calibration guarantee.")
        fold_metrics_path = prophet_root / "fold_metrics.csv"
        if fold_metrics_path.exists():
            with st.expander("Prophet metrics by seasonal fold"):
                render_table(pd.read_csv(fold_metrics_path), caption="Every Prophet variant and quarterly fold", download_name="expanded_prophet_fold_metrics.csv", key="dl_prophet_folds", max_height=420)

    st.markdown("### Expanded item-level study")
    st.caption("All 823 CA_1 / FOODS_3 items, 16 identical seasonal folds, and fold-local demand classes. Metrics are equal-weight across successful item-fold cells; they are not comparable to aggregate-series errors.")
    item_summary_path = resolve_path(Path("artifacts/expanded_intermittent/model_summary.csv"))
    class_summary_path = resolve_path(Path("artifacts/expanded_intermittent/class_summary.csv"))
    metrics_path = resolve_path(Path("artifacts/expanded_intermittent/fold_metrics.csv"))
    if item_summary_path.exists() and metrics_path.exists():
        expanded_items = pd.read_csv(item_summary_path)
        render_table(expanded_items, caption="All-item model summary (successful item-folds only; failure counts shown)", download_name="expanded_item_model_summary.csv", key="dl_expanded_items")
        expanded_classes = pd.read_csv(class_summary_path) if class_summary_path.exists() else pd.DataFrame()
        if not expanded_classes.empty:
            render_table(expanded_classes, caption="Fold-local demand-class summary; special strata such as NoDemand remain separate", download_name="expanded_item_class_summary.csv", key="dl_expanded_classes", max_height=420)
        with st.expander("Per-item, per-fold metrics and fit failures"):
            detailed = pd.read_csv(metrics_path)
            failed = detailed[~detailed["success"].astype(bool)]
            st.write(f"{len(failed):,} failed model-item-fold fits out of {len(detailed):,}; failures are not imputed or replaced.")
            render_table(detailed, caption="Complete expanded experiment result table", download_name="expanded_item_fold_metrics.csv", key="dl_expanded_item_folds", max_height=500)
    else:
        empty_state("Expanded item-level results are not available", "Run the all-item development experiment.", ["python -m src.evaluation.expanded_intermittent run"], ["artifacts/expanded_intermittent/model_summary.csv"])

    xgb_root = resolve_path(Path("artifacts/expanded_xgboost"))
    xgb_summary_path = xgb_root / "pooled_model_summary.csv"
    xgb_sample_path = xgb_root / "matched_sample_model_summary.csv"
    if xgb_summary_path.exists() and xgb_sample_path.exists():
        st.markdown("### XGBoost feature ablations")
        st.caption("Six configurations across the same 16 quarterly folds. Pooled models use all 823 products; per-item and matched-sample comparisons use the fixed first-origin stratified sample of 183 products. Errors are equal-weight item-fold means, not aggregate-series scores.")
        render_table(pd.read_csv(xgb_summary_path), caption="Pooled variants · 823 products · 16 folds", download_name="expanded_xgboost_pooled_summary.csv", key="dl_xgb_pooled")
        render_table(pd.read_csv(xgb_sample_path), caption="Same 183 products for the pooled versus per-item comparison", download_name="expanded_xgboost_matched_sample_summary.csv", key="dl_xgb_matched")
        paired_path = xgb_root / "paired_fold_comparisons.csv"
        if paired_path.exists():
            render_table(pd.read_csv(paired_path).drop(columns="fold_reductions"), caption="Paired fold MAE changes; positive reduction means the candidate was lower in that fold", download_name="expanded_xgboost_paired_folds.csv", key="dl_xgb_paired")
        importance_path = xgb_root / "feature_importance_summary.csv"
        if importance_path.exists():
            importance = pd.read_csv(importance_path)
            st.caption("XGBoost gain importance is a model diagnostic, not causal importance.")
            render_table(importance.sort_values(["model", "gain_mean"], ascending=[True, False]).groupby("model", as_index=False).head(10), caption="Top 10 features by mean gain within each model", download_name="expanded_xgboost_feature_importance.csv", key="dl_xgb_importance", max_height=360)
        folds_path = xgb_root / "pooled_fold_metrics.csv"
        if folds_path.exists():
            with st.expander("Pooled XGBoost metrics by fold"):
                render_table(pd.read_csv(folds_path), caption="All feature variants on each common seasonal fold", download_name="expanded_xgboost_fold_metrics.csv", key="dl_xgb_folds", max_height=400)
        st.caption("Features use historical lags/rolling values and known-in-advance calendar/event flags. Differences are descriptive; no statistical significance or causal feature effect is claimed.")
    else:
        empty_state("Expanded XGBoost ablations are not available", "Run the checkpointed full-panel feature study.", ["python -m src.evaluation.expanded_xgboost run"], ["artifacts/expanded_xgboost/pooled_model_summary.csv"])

    inventory_summary_path = resolve_path(Path("artifacts/expanded_inventory/service_level_summary.csv"))
    if inventory_summary_path.exists():
        st.markdown("### Inventory service-level calibration")
        inventory_summary = pd.read_csv(inventory_summary_path)
        render_table(inventory_summary, caption="7-day protection interval; calibration uses earlier folds only", download_name="expanded_inventory_service_level.csv", key="dl_expanded_inventory")
        st.caption("Overlapping windows are dependent; intervals use an item-fold cluster bootstrap. Targets remain illustrative inputs, not achieved guarantees.")
    alternative_inventory = resolve_path(Path("artifacts/expanded_inventory_calibration_v2"))
    alternative_summary = alternative_inventory / "calibration_method_summary.csv"
    if alternative_summary.exists():
        st.markdown("### Alternative inventory calibration methods")
        st.caption("All methods use the same later evaluation folds and 7-day windows. Calibration uses only strictly earlier, completed out-of-sample folds. The target is 95% cycle service; no method is a guarantee.")
        render_table(pd.read_csv(alternative_summary), caption="Target and observed service by model and calibration rule", download_name="alternative_inventory_calibration.csv", key="dl_alt_inventory")
        class_summary = alternative_inventory / "calibration_model_class_summary.csv"
        if class_summary.exists():
            with st.expander("Inventory calibration by fold-local demand class"):
                render_table(pd.read_csv(class_summary), caption="Class-specific achieved service and cluster-bootstrap interval; special strata stay separate", download_name="inventory_calibration_by_class.csv", key="dl_alt_inventory_class", max_height=440)
        st.caption("These are cycle-service estimates, not fill rate or supplier performance. Lead/review times are illustrative; the shortfall is retained.")

# ------------------------------------------------------------------------------------------------ final holdout
with tab_final:
    state = da.holdout_state()
    st.markdown("### Development results are not final results")
    st.markdown(
        f"Everything above comes from walk-forward folds inside the development window. Model choices were made by looking at those folds, so they are "
        f"optimistic for any later period. The last {b.window.n_holdout_days} days ({b.window.holdout_start.date()} to {b.window.holdout_end.date()}) were set aside "
        "for one final evaluation of the frozen candidates. The final results below are shown separately and were not used for model selection."
    )
    if state["evaluated"]:
        callout("info", "The final holdout was evaluated once", "Scores below cover the reserved 28 days. Candidate settings and result hashes are in the final evaluation record.")
        final_dir = resolve_path(Path(config["phase3"]["final_dir"]))
        final_results_path = final_dir / config["final_holdout"]["results_filename"]
        if final_results_path.exists():
            render_table(pd.read_csv(final_results_path), caption="One-time final holdout results. Aggregate and item-level scores are separate scopes.",
                         download_name="final_holdout_results.csv", key="dl_final_holdout_results", max_height=500)
        final_class_path = final_dir / config["final_holdout"].get("class_results_filename", "final_holdout_class_results.csv")
        if final_class_path.exists():
            with st.expander("Final holdout results by development-only demand class"):
                render_table(pd.read_csv(final_class_path), caption="Class assignments come from the development window; they did not use final-period sales.",
                             download_name="final_holdout_class_results.csv", key="dl_final_holdout_class", max_height=440)
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
