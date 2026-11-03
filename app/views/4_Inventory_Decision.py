"""Page 4 - Inventory Decisions: an illustrative reorder point built from saved forecasts and saved forecast errors."""

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
from components.cards import badge, callout, empty_state, esc, loaded_error, page_header, provenance_panel, render_kpis  # noqa: E402
from components.tables import render_table  # noqa: E402
from src.evaluation.forecast_artifacts import expected_fold_windows, forecasts_dir, notebook_command, producer_notebook  # noqa: E402
from src.evaluation.model_comparison import item_sources  # noqa: E402
from src.inventory import reorder as inv  # noqa: E402
from src.utils.paths import resolve_path  # noqa: E402

page_header("Inventory Decisions", "Turn a saved forecast and its saved forecast errors into an illustrative reorder point for one product.")
callout("warn", "Illustrative, not an operating policy",
        inv.ASSUMPTIONS_TEXT + " The calculation replays a past forecast origin; it does not forecast the future.")

config = da.get_config()
section = config["inventory"]
bundle = da.load_bundle()
if not bundle.ok:
    loaded_error(bundle, "Saved forecasts are not available")
    st.stop()
b = bundle.value
if b.item is None:
    files = [s.filename for s in item_sources(config)]
    empty_state("No item-level forecasts to build a reorder point from", "An inventory estimate needs a saved item-level forecast and its saved errors.",
                sorted({notebook_command(producer_notebook(f) or "") for f in files}), files)
    st.stop()

models = da.ordered_models(b.item, "item")
windows = expected_fold_windows(b.dev_series.index, int(config["backtest"]["horizon"]), int(config["backtest"]["n_folds"]))
fold_labels = {int(r.fold): f"Fold {int(r.fold)}: {r.forecast_start.date()} to {r.forecast_end.date()}" for r in windows.itertuples() if int(r.fold) >= 2}

st.markdown("## Inputs")
a1, a2, a3 = st.columns(3)
with a1:
    item = st.selectbox("Product", sorted(b.item_series), key="inv_item", help="One of the 20 selected items.")
    model = st.selectbox("Forecast model", models, index=models.index("TSB(alpha=0.1,beta=0.1)") if "TSB(alpha=0.1,beta=0.1)" in models else 0, key="inv_model",
                         help="The saved item-level forecast the expected demand comes from.")
with a2:
    lead = st.number_input("Lead time (days)", min_value=1, max_value=inv.MAX_PROTECTION_DAYS - 1, value=int(section["default_lead_time_days"]), step=1, key="inv_lead",
                           help="Days between placing an order and receiving it. Illustrative default from config.yaml.")
    review = st.number_input("Review period (days, 0 = continuous review)", min_value=0, max_value=inv.MAX_PROTECTION_DAYS - 1, value=int(section["default_review_period_days"]), step=1, key="inv_review",
                             help="Days between stock checks. The protection interval is lead time + review period.")
with a3:
    service = st.slider("Target cycle service level", min_value=float(section["min_service_level"]), max_value=float(section["max_service_level"]),
                        value=float(section["default_service_level"]), step=0.005, format="%.3f", key="inv_service",
                        help="Target probability of not running out during one replenishment cycle. Not the fill rate.")
    fold = st.selectbox("Forecast origin to replay", sorted(fold_labels), index=len(fold_labels) - 1, format_func=fold_labels.get, key="inv_fold",
                        help="The safety stock is calibrated on the saved forecast errors of EARLIER folds only.")
know_stock = st.checkbox("I know the stock position (optional)", key="inv_know_stock", help="Adds an inventory position and a suggested order quantity.")
on_hand = on_order = None
if know_stock:
    s1, s2 = st.columns(2)
    with s1:
        on_hand = st.number_input("Inventory on hand (units)", min_value=0.0, value=0.0, step=1.0, key="inv_on_hand")
    with s2:
        on_order = st.number_input("Already on order (units)", min_value=0.0, value=0.0, step=1.0, key="inv_on_order")

try:
    assumptions = inv.InventoryAssumptions(lead_time_days=int(lead), review_period_days=int(review), service_level=float(service),
                                           on_hand=None if on_hand is None else float(on_hand), on_order=0.0 if on_order is None else float(on_order))
except ValueError as exc:
    callout("error", "These inputs cannot be used", str(exc))
    st.stop()

try:
    result, context = inv.reorder_point_at_origin(b.item, model=model, item_id=item, fold=int(fold), assumptions=assumptions,
                                                  min_error_windows_warning=int(section["min_error_windows_warning"]))
except ValueError as exc:
    callout("error", "No real estimate can be produced for this selection", f"{exc} Nothing is estimated in its place.")
    st.stop()

errors = inv.cumulative_error_samples(b.item, assumptions.protection_days, model=model, item_id=item, folds=context["calibration_folds"])
low, high = inv.bootstrap_safety_stock_interval(errors, assumptions.service_level, n_resamples=int(section["bootstrap_resamples"]),
                                                seed=int(section["bootstrap_seed"]), confidence=0.90, block_size=1)

st.markdown("## Result")
unit_note = f"units of {item}"
render_kpis([
    {"label": "Expected demand over the protection interval", "value": f"{result.expected_demand:,.1f}", "note": f"{unit_note}, {result.protection_days} days, from the saved {model} forecast"},
    {"label": "Safety stock", "value": f"{result.safety_stock:,.1f}", "note": f"90% bootstrap interval {low:,.1f} to {high:,.1f}"},
    {"label": "Reorder point", "value": f"{result.reorder_point:,.1f}", "note": "expected demand + safety stock"},
] + ([{"label": "Suggested order", "value": f"{result.suggested_order_quantity:,.1f}", "note": f"inventory position {result.inventory_position:,.1f} vs target {result.reorder_point:,.1f}"}]
     if result.suggested_order_quantity is not None else []))
if result.suggested_order_quantity is not None:
    status = badge("warn", "Position below the reorder point: order") if result.suggested_order_quantity > 0 else badge("ok", "Position at or above the reorder point")
    st.markdown(status, unsafe_allow_html=True)
for w in result.warnings:
    callout("warn", "Caution", w)

realised = None
shown = b.item[(b.item["model"] == model) & (b.item["item_id"] == item) & (b.item["fold"] == fold)].sort_values("date")
if not shown.empty:
    realised = shown["actual"].to_numpy(dtype=float)
charts.show(charts.inventory_chart(context["forecast_values"], realised, protection_days=assumptions.protection_days, expected=result.expected_demand,
                                       reorder_point=result.reorder_point, safety_stock=result.safety_stock, start=context["forecast_start"]))
st.caption(f"The dotted line is what was actually sold ({context['realised_demand_over_protection']:,.1f} units over the protection interval). It was not known when the order would have been placed and is shown only for comparison.")

st.markdown("## What this means, in plain English")
st.markdown(f'<div class="rd-callout info" style="white-space:pre-line">{esc(inv.explain_result(result, assumptions, item_label=item))}</div>', unsafe_allow_html=True)

st.markdown("## Assumptions and method")
render_table(pd.DataFrame([
    {"Assumption": "Product / model", "Value": f"{item} / {model}"},
    {"Assumption": "Lead time + review period", "Value": f"{assumptions.lead_time_days} + {assumptions.review_period_days} = {assumptions.protection_days} days"},
    {"Assumption": "Target cycle service level", "Value": f"{assumptions.service_level:.1%} (error quantile used: {result.error_quantile_level:.1%})"},
    {"Assumption": "Forecast used", "Value": f"saved forecast for {context['forecast_start']} to {context['forecast_end']}"},
    {"Assumption": "Safety stock method", "Value": f"empirical quantile of {result.n_error_samples} cumulative forecast errors (actual − forecast) over overlapping {assumptions.protection_days}-day windows in folds {context['calibration_folds']}"},
    {"Assumption": "Uncertainty of the safety stock", "Value": f"{int(section['bootstrap_resamples'])} bootstrap resamples of those errors (seed {int(section['bootstrap_seed'])}), 90% interval"},
    {"Assumption": "Not modelled", "Value": "costs, lot sizes, minimum order quantities, lost-sales censoring, supplier variability, promotions after the origin"},
]), caption="Assumptions behind this result", max_height=None)
st.caption("Safety stock assumes the saved forecast errors are representative of the next cycle's errors. Sales are censored by stock-outs in the data, so true demand can be higher than recorded.")

provenance_panel([
    {"role": f"Forecasts and errors of {model}", **da.file_record(forecasts_dir(config) / b.sources.get(model, ""))},
    {"role": "Optional saved demonstration (notebook 10)", **da.file_record(resolve_path(config["phase3"]["inventory_dir"]) / config["phase3"]["inventory_demo_filename"])},
], f"Computed live from the saved {model} forecast table by src/inventory/reorder.py; calibration folds {context['calibration_folds']}.")
