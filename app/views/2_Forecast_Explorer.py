"""Page 2 - Forecast Explorer: saved backtest forecasts next to what actually happened."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
APP_DIR = Path(__file__).resolve().parents[1]
for _p in (str(ROOT), str(APP_DIR)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import pandas as pd  # noqa: E402
import streamlit as st  # noqa: E402

import data_access as da  # noqa: E402
from components import charts  # noqa: E402
from components.cards import callout, empty_state, loaded_error, page_header, provenance_panel, render_kpis  # noqa: E402
from components.tables import render_table  # noqa: E402
from src.evaluation.forecast_artifacts import (  # noqa: E402
    expected_fold_windows, forecasts_dir, notebook_command, producer_notebook, truncate_horizon,
)
from src.evaluation.metrics import compute_metrics  # noqa: E402
from src.evaluation.model_comparison import aggregate_sources, item_sources  # noqa: E402

page_header("Forecast Explorer", "Pick a series, a saved model, a forecast origin and a horizon. Each forecast is shown beside the sales that actually followed.")

config = da.get_config()
bundle = da.load_bundle()
if not bundle.ok:
    loaded_error(bundle, "Saved forecasts are not available")
    st.stop()
b = bundle.value
AGG_LABEL = "Aggregate series (all items together)"
horizons = [h for h in config["dashboard"]["horizons"] if h <= int(config["backtest"]["horizon"])]

c1, c2 = st.columns(2)
with c1:
    series_options = [AGG_LABEL] + (sorted(b.item_series) if b.item is not None else [])
    series_choice = st.selectbox("Series", series_options, key="fx_series",
                                 help="The aggregate series, or one of the 20 selected items (item-level forecasts are separate files).")
is_aggregate = series_choice == AGG_LABEL
table = b.aggregate if is_aggregate else b.item
history = b.dev_series if is_aggregate else b.item_series[series_choice]
if table is None:
    files = [s.filename for s in (aggregate_sources(config) if is_aggregate else item_sources(config))]
    empty_state("No saved forecasts for this series level", "None of the expected forecast files is available.",
                sorted({notebook_command(producer_notebook(f) or "") for f in files}), files)
    st.stop()
scope = "aggregate" if is_aggregate else "item"
if not is_aggregate:
    table = table[table["item_id"] == series_choice]
models = da.ordered_models(table, scope)
with c2:
    default_model = next((m for m in models if m.startswith("Prophet(additive)") and "no-" not in m), models[0])
    model = st.selectbox("Model", models, index=models.index(default_model) if default_model in models else 0, key="fx_model",
                         help="Only models with saved forecasts appear here. Models that have not been run are listed below.")
c3, c4 = st.columns(2)
windows = expected_fold_windows(b.dev_series.index, int(config["backtest"]["horizon"]), int(config["backtest"]["n_folds"]))
fold_labels = {int(r.fold): f"Fold {int(r.fold)}: cutoff {r.origin_date.date()}, forecast {r.forecast_start.date()} to {r.forecast_end.date()}" for r in windows.itertuples()}
with c3:
    fold = st.selectbox("Forecast origin (training cutoff)", sorted(fold_labels), index=len(fold_labels) - 1, format_func=fold_labels.get, key="fx_fold",
                        help="The model saw sales up to the cutoff only; the window after it is what it forecast.")
with c4:
    horizon = st.selectbox("Forecast horizon (days)", horizons, index=horizons.index(config["dashboard"]["default_horizon"]) if config["dashboard"]["default_horizon"] in horizons else 0,
                           key="fx_horizon", help="Saved forecasts cover 28 days; shorter horizons show their first days.")

part = table[(table["model"] == model) & (table["fold"] == fold)].copy()
part["date"] = pd.to_datetime(part["date"])
part = part.sort_values("date")
if part.empty:
    callout("warn", "No saved forecast for this selection", f"{model} has no rows for fold {fold}.")
    st.stop()
if "success" in part.columns and not part["success"].astype(bool).all():
    err = part["error"].dropna().iloc[0] if "error" in part.columns and part["error"].notna().any() else "the fit failed"
    callout("error", "This model failed on this fold", f"No forecast was saved (reason: {err}). Nothing is substituted for it.")
    st.stop()

group_cols = ["model"] + (["item_id"] if not is_aggregate else [])
shown = truncate_horizon(part.assign(date=part["date"]), horizon, group_cols)
cutoff = pd.Timestamp(windows.loc[windows["fold"] == fold, "origin_date"].iloc[0])
unit = "units" if not is_aggregate else "units, all items"
m = compute_metrics(shown["actual"].to_numpy(), shown["predicted"].to_numpy())
bias = float((shown["predicted"] - shown["actual"]).mean())

render_kpis([
    {"label": "Model", "value": model, "note": f"{'aggregate series' if is_aggregate else series_choice}"},
    {"label": "Training cutoff", "value": str(cutoff.date()), "note": "last day of sales the model saw"},
    {"label": "Forecast window starts", "value": str(shown['date'].min().date()), "note": f"through {shown['date'].max().date()} ({len(shown)} days, horizon {horizon})"},
    {"label": f"MAE, these {len(shown)} days", "value": f"{m.mae:,.2f}", "note": f"RMSE {m.rmse:,.2f}; bias {bias:+,.2f} (forecast − actual)"},
])
charts.show(charts.forecast_chart(history, shown, model=model, cutoff=cutoff, unit=unit))
callout("info", "These are backtest forecasts for days that have already happened",
        f"The app never forecasts beyond {b.window.dev_end.date()}. Each forecast was produced from data up to its training cutoff, then compared "
        "with the sales that followed. The reserved final holdout is not shown.")

st.markdown("## Forecast table")
out = shown.copy()
out["horizon_step"] = range(1, len(out) + 1)
out["error (forecast − actual)"] = out["predicted"] - out["actual"]
cols = ["date", "horizon_step", "actual", "predicted", "error (forecast − actual)"] + [c for c in ("lower", "upper") if c in out.columns and out[c].notna().any()]
out = out[cols].rename(columns={"predicted": "forecast", "lower": "interval lower (80%)", "upper": "interval upper (80%)"})
out["date"] = out["date"].dt.strftime("%Y-%m-%d")
render_table(out, caption=f"{model}, fold {fold}, first {len(shown)} forecast days", formats={"actual": "{:,.1f}", "forecast": "{:,.2f}"},
             download_name=f"forecast_{scope}_{fold}.csv", key="fx_download")

absent = [s for s in (aggregate_sources(config) if is_aggregate else item_sources(config)) if s.filename in set(b.audit.loc[b.audit["status"] != "ok", "artifact"])]
if absent:
    empty_state("Some models are not available to choose", "Their forecast files have not been saved, so they are not offered and nothing is estimated for them.",
                sorted({notebook_command(producer_notebook(s.filename) or "") for s in absent}), [s.filename for s in absent])

source_file = b.sources.get(model, "")
records = [{"role": f"Forecasts and actuals of {model}", **da.file_record(forecasts_dir(config) / source_file)}] if source_file else []
records.append({"role": "Development series used for the history line", **da.file_record(da.aggregate_series_path(config))})
provenance_panel(records, f"Model {model}; training cutoff {cutoff.date()}; forecast window {shown['date'].min().date()} to {shown['date'].max().date()}; "
                          "walk-forward backtest, expanding training window.")
