"""Page 5 - Model Reliability: only the diagnostics that saved runs actually produced, plus known limitations."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
APP_DIR = Path(__file__).resolve().parents[1]
for _p in (str(ROOT), str(APP_DIR)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import streamlit as st  # noqa: E402

import data_access as da  # noqa: E402
from components import charts  # noqa: E402
from components.cards import callout, empty_state, loaded_error, page_header, provenance_panel  # noqa: E402
from components.tables import render_table  # noqa: E402
from src.evaluation.forecast_artifacts import metrics_dir  # noqa: E402
from src.evaluation.model_comparison import model_summary  # noqa: E402
from src.utils.paths import PROJECT_ROOT  # noqa: E402

page_header("Model Reliability", "What the saved runs say about convergence, residuals and stability. A diagnostic appears only if a run produced it.")

config = da.get_config()
bundle = da.load_bundle()
if not bundle.ok:
    loaded_error(bundle, "Saved forecasts are not available")
    st.stop()
b = bundle.value
records = []

# ------------------------------------------------------------------------------------------------ data validation
st.markdown("## Data validation")
validation = da.data_validation_summary()
if not validation.ok:
    loaded_error(validation, "The development series could not be checked")
else:
    checks = validation.value["checks"].copy()
    checks["result"] = np.where(checks["passed"], "✔ passed", "✖ FAILED")
    render_table(checks[["check", "result", "detail"]].rename(columns={"check": "Check", "result": "Result", "detail": "What was verified"}),
                 caption=f"Series checks on the {validation.value['window'].n_development_days:,}-day development series", max_height=None)
    records.append({"role": "Data validation", **da.file_record(da.aggregate_series_path(config))})

# ------------------------------------------------------------------------------------------------ fit and convergence
st.markdown("## Fit and convergence")
rows = []
res = da.read_optional_csv("residual_diagnostics_fold_summary.csv")
if res.ok:
    records.append({"role": "Classical-model convergence", **da.file_record(metrics_dir(config) / "residual_diagnostics_fold_summary.csv")})
    for model, g in res.value.groupby("model", sort=False):
        rows.append({"Model": model, "Fits": len(g), "Converged": int(g["converged"].astype(bool).sum()), "Source": "Phase 2 residual diagnostics",
                     "Note": "; ".join(sorted({str(n) for n in g["notes"].dropna() if str(n).strip()}))[:140] or "-"})
pro = da.read_optional_csv("prophet_fit_diagnostics.csv")
if pro.ok:
    records.append({"role": "Prophet fit diagnostics", **da.file_record(metrics_dir(config) / "prophet_fit_diagnostics.csv")})
    for model, g in pro.value.groupby("model", sort=False):
        rows.append({"Model": model, "Fits": len(g), "Converged": int(g["optimizer_converged"].astype(bool).sum()), "Source": "Prophet fit diagnostics",
                     "Note": f"{int((~g['success'].astype(bool)).sum())} failed fit(s)"})
nn = da.read_optional_csv("lstm_gru_fit_diagnostics.csv")
if nn.ok:
    records.append({"role": "LSTM/GRU training diagnostics", **da.file_record(metrics_dir(config) / "lstm_gru_fit_diagnostics.csv")})
    d = nn.value
    for model, g in d.groupby("model", sort=False):
        capped = int(g["hit_max_epochs"].astype(bool).sum()) if "hit_max_epochs" in g.columns else None
        rows.append({"Model": model, "Fits": len(g), "Converged": len(g) - capped if capped is not None else np.nan, "Source": "Neural training diagnostics",
                     "Note": (f"{capped} network(s) stopped at the epoch cap" if capped is not None else "-")})
xg = da.read_optional_csv("xgboost_fold_diagnostics.csv")
if xg.ok:
    records.append({"role": "XGBoost fold diagnostics", **da.file_record(metrics_dir(config) / "xgboost_fold_diagnostics.csv")})
    for model, g in xg.value.groupby("model", sort=False):
        rows.append({"Model": model, "Fits": len(g), "Converged": int(g["success"].astype(bool).sum()), "Source": "XGBoost fold diagnostics",
                     "Note": f"{int(g['n_warnings'].sum())} warning(s); {int(g['n_predictions_clipped_at_zero'].sum())} prediction(s) clipped at zero"})
if rows:
    render_table(pd.DataFrame(rows), caption="Fits that ran to a usable result (Converged = optimiser converged, or training stopped before the epoch cap, or the fit succeeded)", max_height=None)
missing = [n for n, r in (("Phase 2 classical models", res), ("Prophet", pro), ("LSTM / GRU", nn), ("XGBoost", xg)) if not r.ok]
if missing:
    cmds = sorted({r.command for r in (res, pro, nn, xg) if not r.ok and r.command})
    empty_state("Diagnostics for some models are not available", "No diagnostics are shown for: " + ", ".join(missing) + ". A model without a diagnostic file is not treated as reliable.", cmds)
sar = da.read_optional_csv("sarima_fold3_convergence_sensitivity.csv")
if sar.ok:
    render_table(sar.value, caption="SARIMA fold 3: does a different optimiser or iteration cap change convergence? (saved sensitivity run)", max_height=None)

# ------------------------------------------------------------------------------------------------ residuals
st.markdown("## Residual diagnostics")
lb = da.read_optional_csv("residual_diagnostics_ljung_box.csv")
if lb.ok:
    records.append({"role": "Ljung-Box tests", **da.file_record(metrics_dir(config) / "residual_diagnostics_ljung_box.csv")})
    g = lb.value.groupby(["model", "lag"]).agg(folds=("fold", "nunique"), rejects=("reject_h0", lambda s: int(s.astype(bool).sum())), median_p=("lb_pvalue", "median")).reset_index()
    g = g.rename(columns={"model": "Model", "lag": "Lag", "folds": "Folds tested", "rejects": "Folds with autocorrelated errors (p < 0.05)", "median_p": "Median p-value"})
    render_table(g, caption="Ljung-Box test on in-sample residuals of the classical models (saved by notebook 04)", formats={"Median p-value": lambda v: "< 0.001" if v < 0.001 else f"{v:.3f}"}, max_height=320)
    st.caption("A small p-value means the residuals still contain structure the model did not capture. Rejections are evidence of misfit, not just noise. "
               "n/a means the lag is at or below the number of fitted parameters, so the test has no degrees of freedom left and no p-value exists.")
else:
    loaded_error(lb, "Ljung-Box results are not available")

agg = b.aggregate
if agg is not None:
    st.markdown("### Autocorrelation of forecast errors (from the saved forecasts)")
    m1, m2 = st.columns(2)
    with m1:
        rmodel = st.selectbox("Model", da.ordered_models(agg, "aggregate"), key="rel_model")
    with m2:
        rfold = st.selectbox("Fold", sorted(int(f) for f in agg["fold"].unique()), index=4, key="rel_fold")
    info = da.residual_acf(agg, rmodel, int(rfold))
    if info is None:
        callout("warn", "No error autocorrelation can be computed", f"{rmodel} has no usable saved forecast for fold {rfold}, or its errors are constant.")
    else:
        charts.show(charts.acf_chart(info, title=f"{rmodel}, fold {rfold}: autocorrelation of out-of-sample errors (n = {info['n']})"))
        st.caption(f"Only {info['n']} errors per fold, so the band is wide and the test has little power. Bars outside it are worth a look; bars inside it do not prove independence.")
    pdir = PROJECT_ROOT / "reports" / "phase2"
    figures = sorted(pdir.glob("residual_diagnostics_*.png")) if pdir.exists() else []
    if figures:
        with st.expander("Phase 2 residual figures (static images, drawn on a light background)"):
            for f in figures:
                st.image(str(f), caption=f.name)

# ------------------------------------------------------------------------------------------------ stability
st.markdown("## Variation across validation folds")
metrics = da.load_metrics("aggregate")
if metrics.ok:
    s = model_summary(metrics.value["folds"])
    s["range / mean"] = (s["MAE_max"] - s["MAE_min"]) / s["MAE_mean"]
    show = s[["model", "folds_scored", "folds_failed", "MAE_mean", "MAE_std", "MAE_min", "MAE_max", "range / mean"]].rename(columns={
        "model": "Model", "folds_scored": "Folds scored", "folds_failed": "Folds failed", "MAE_mean": "MAE mean", "MAE_std": "MAE std", "MAE_min": "MAE min", "MAE_max": "MAE max"})
    render_table(show, caption="Aggregate series: how much MAE moves from fold to fold (units per day)", best={"range / mean": "min"}, max_height=None)
    st.caption("A model whose error swings widely between folds is less predictable than its mean suggests. 'range / mean' is the spread of fold MAEs relative to their average.")
pro_cov = b.aggregate[b.aggregate["model"].astype(str).str.startswith("Prophet")] if agg is not None else None
if pro_cov is not None and len(pro_cov) and {"lower", "upper"} <= set(pro_cov.columns) and pro_cov["lower"].notna().any():
    cov = pro_cov.dropna(subset=["lower", "upper"]).assign(inside=lambda d: (d["actual"] >= d["lower"]) & (d["actual"] <= d["upper"]))
    cov_table = cov.groupby("model").agg(days=("inside", "size"), coverage=("inside", "mean")).reset_index().rename(columns={"model": "Model", "days": "Days", "coverage": "Share of actuals inside the 80% interval"})
    render_table(cov_table, caption="Prophet interval calibration (nominal 80%)", formats={"Share of actuals inside the 80% interval": "{:.1%}"}, max_height=None)
    st.caption("Only Prophet saves prediction intervals. Coverage well below 80% means the intervals are too narrow to trust as stated.")

# ------------------------------------------------------------------------------------------------ leakage and limitations
st.markdown("## Leakage guards and known limitations")
latest = max(pd.to_datetime(t["date"]).max() for t in (b.aggregate, b.item) if t is not None)
guard_rows = [
    {"Guard": "No saved forecast date is after the last development day", "Result": ("✔ holds" if latest <= b.window.dev_end else "✖ VIOLATED"), "Evidence": f"latest date in any saved table: {latest.date()}; development ends {b.window.dev_end.date()}"},
    {"Guard": "Forecast tables pass the walk-forward audit", "Result": ("✔ holds" if (b.audit["status"] != "audit failed").all() else "✖ VIOLATED"),
     "Evidence": f"{int((b.audit['status'] == 'ok').sum())} of {len(b.audit)} files audited and accepted; missing files are listed on the Overview page"},
    {"Guard": "Final 28 days not evaluated", "Result": ("✔ holds" if not da.holdout_state()["evaluated"] else "ℹ evaluated once"), "Evidence": "artifacts/final holds the record if it was run"},
]
render_table(pd.DataFrame(guard_rows), caption="Checks computed from the saved artifacts when this page loads", max_height=None)
st.markdown(
    "- **Few folds.** Five walk-forward folds are a small sample; differences of a few percent between models are within fold-to-fold variation.\n"
    "- **One store, one department.** Results do not transfer to other stores or categories without re-running the experiments.\n"
    "- **Demand classes.** Phase 1 labels use the full history, so they include the reserved days; the all-item experiment recomputes classes on each fold's training window.\n"
    "- **Calendar inputs.** LSTM/GRU/Prophet/XGBoost may use event and SNAP flags for forecast days because those are published in advance; they never see future sales.\n"
    "- **Censored demand.** Recorded sales are not demand when stock runs out, which biases forecasts and safety stock downward.\n"
    "- **Selection effect.** Choosing the best of many models on the same folds is optimistic; only the one-time final evaluation is a clean estimate."
)

# ------------------------------------------------------------------------------------------------ provenance
st.markdown("## Run provenance")
meta = da.load_run_metadata()
prow = []
for component, r in meta.items():
    if not r["exists"]:
        prow.append({"Component": component, "Run recorded": "not found", "Written (UTC)": "-", "Python": "-", "Settings match config.yaml": "-"})
        continue
    state = {True: "✔ yes", False: "▲ no: config changed since this run", None: "unknown"}[r.get("config_matches")]
    prow.append({"Component": component, "Run recorded": "yes", "Written (UTC)": r.get("generated", "-"), "Python": r.get("python", "-"), "Settings match config.yaml": state})
render_table(pd.DataFrame(prow), caption="Saved run metadata (hashes of the settings, library versions; no machine or user details)", max_height=None)
provenance_panel(records, "Diagnostics shown above were read from these saved files.")
