"""Page 1 - Overview: what the data is, how much of it there is, and which models have saved results."""

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
from src.evaluation.forecast_artifacts import notebook_command, producer_notebook  # noqa: E402
from src.utils.paths import resolve_path  # noqa: E402

page_header("Overview", "What the data covers, how large it is, and which forecasting models have saved development results.")

config = da.get_config()
overview = da.load_overview()
if not overview.ok:
    loaded_error(overview, "The development data is not available")
    st.stop()
v = overview.value
w = v["window"]

render_kpis([
    {"label": "Store / department", "value": f"{config['scope']['store_id']} · {config['scope']['dept_id']}", "note": "Walmart M5, one store, one food department"},
    {"label": "Products", "value": f"{v['n_items']:,}", "note": "distinct items in the department"},
    {"label": "Development window", "value": f"{v['n_days']:,} days", "note": f"{w.dev_start.date()} to {w.dev_end.date()}"},
    {"label": "Item-day observations", "value": f"{v['dev_item_days']:,}",
     "note": (f"in the development window; the full panel has {v['panel_item_days']:,}, of which {v['panel_item_days'] - v['dev_item_days']:,} "
              f"are in the reserved final {w.n_holdout_days} days" if v["panel_item_days"] else "in the development window")},
    {"label": "Zero-sales item-days", "value": f"{v['zero_share']:.1%}", "note": "share of item-days with no units sold (development window)"},
    {"label": "Units sold, all items", "value": f"{v['total_units']:,.0f}", "note": "development window total"},
])

st.markdown("## Sales history")
charts.show(charts.history_trend_chart(v["series"]))
st.caption("Daily units summed over all items (grey line) and the trailing 28-day average (blue). The series ends on the last "
           f"development day, {w.dev_end.date()}; the final {w.n_holdout_days} days are not shown anywhere in this app.")

st.markdown("## Demand classes")
left, right = st.columns([5, 4])
with left:
    charts.show(charts.class_bar_chart(v["dev_classes"], title="Items per demand class (development window)"))
with right:
    table = v["dev_classes"].rename(columns={"demand_class": "Demand class", "items": "Development window"})
    if v["phase1_classes"] is not None:
        table = table.merge(v["phase1_classes"].rename(columns={"demand_class": "Demand class", "items": "Phase 1 labels"}), on="Demand class", how="outer")
    render_table(table, caption="Number of items per class", max_height=None)
st.caption("Classes follow the Syntetos-Boylan-Croston rule on average demand interval (ADI 1.32) and the squared coefficient of variation "
           "of demand size (CV² 0.49). Smooth: ADI < 1.32 and CV² < 0.49. Erratic: ADI < 1.32, CV² ≥ 0.49. Intermittent: ADI ≥ 1.32, CV² < 0.49. "
           "Lumpy: ADI ≥ 1.32 and CV² ≥ 0.49. The \"Development window\" column counts items by their class in the 1,913 development days; the \"Phase 1 labels\" column was computed on the full "
           "history, so a few items differ by one class.")

st.markdown("## Model status")
bundle = da.load_bundle()
if not bundle.ok:
    loaded_error(bundle, "Saved forecasts could not be read")
else:
    b = bundle.value
    rows = []
    for _, r in b.audit.iterrows():
        ok = r["status"] == "ok"
        rows.append({"Forecast file": r["artifact"], "Level": "aggregate series" if r["scope"] == "aggregate" else "item level (20 items)",
                     "Status": ("available" if ok else str(r["status"])), "Models": r["models"] or "none", "Rows": int(r["rows"])})
    status_table = pd.DataFrame(rows)
    ok_count = int((b.audit["status"] == "ok").sum())
    st.markdown(badge("ok" if ok_count == len(b.audit) else "warn", f"{ok_count} of {len(b.audit)} forecast files available"), unsafe_allow_html=True)
    render_table(status_table, caption="Saved forecast files and the models they contain", max_height=None)
    missing = b.audit[b.audit["status"] != "ok"]
    if len(missing):
        empty_state("Some results have not been produced yet", "Pages that need them show what is missing instead of estimating it.",
                    sorted({notebook_command(producer_notebook(a) or "") for a in missing["artifact"]}), list(missing["artifact"]))
    for note in b.notes:
        st.markdown(f'<p class="rd-hint">{esc(note)}</p>', unsafe_allow_html=True)

st.markdown("## What the metrics mean")
st.markdown(
    "- **MAE** is the average absolute miss in units per day. **RMSE** squares the misses first, so large misses count more.\n"
    "- **WAPE** is total absolute error divided by total sales; **MASE** divides MAE by the error of repeating last week on the training data (below 1 beats that rule).\n"
    "- **MAPE** divides each miss by that day's sales, so it cannot be computed on zero-sales days; this app excludes them and says how many.\n"
    "- Aggregate-series errors (all items together, thousands of units) and item-level errors (one item, a few units) are on different scales and are never ranked against each other."
)
fh = da.holdout_state()
callout("info", "Development results only",
        "Every number here comes from walk-forward backtests inside the development window. The reserved final 28 days have "
        + ("been evaluated once; see artifacts/final." if fh["evaluated"] else "not been evaluated, so no final-holdout result exists yet."))

records = [{"role": "Aggregate series", "file": da.repo_path(da.aggregate_series_path(config)), **{k: da.file_record(da.aggregate_series_path(config))[k] for k in ("exists", "modified_utc", "sha256")}}]
panel = resolve_path(Path(config["data"]["interim_dir"]) / config["output"]["panel_filename"])
records.append({"role": "Item-day panel (counts, zero share, classes)", **da.file_record(panel)})
provenance_panel(records, "Counts and class labels are computed from the development window only (the date filter is applied before aggregation).")
