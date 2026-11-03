"""Page 6 - About and Methodology: question, data, methods, protocol, metrics, limits and how to reproduce."""

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
from components.cards import callout, code_block, page_header  # noqa: E402
from components.tables import render_table  # noqa: E402

config = da.get_config()
bt = config["backtest"]
page_header("About & Methodology", "The research question, the data, the models, how they were tested and what the results can and cannot support.")

st.markdown("## Research question")
st.markdown("How well can daily demand for intermittent retail items be forecast, and does a more complex model (LSTM/GRU, Prophet, a pooled XGBoost model) "
            "beat simple baselines (Seasonal Naive, ARIMA, Holt-Winters, Croston, TSB) under an honest, leakage-free walk-forward evaluation? "
            "A second question is how a forecast becomes a stocking decision: a reorder point needs the forecast **and** its typical error.")

st.markdown("## Data")
st.markdown(f"Walmart **M5** daily unit sales for store **{config['scope']['store_id']}**, department **{config['scope']['dept_id']}**: 823 items, 1,941 days "
            "in the raw file (2011-01-29 to 2016-05-22; the last 28 of them are reserved and never used here), 1,597,443 item-day rows. Calendar events and SNAP days come from the M5 calendar. The raw M5 files are not part of the repository; "
            "the pipeline builds the processed tables from them.")

st.markdown("## Demand classes")
render_table(pd.DataFrame([
    {"Class": "Smooth", "ADI": "< 1.32", "CV²": "< 0.49", "Meaning": "regular demand, stable sizes"},
    {"Class": "Erratic", "ADI": "< 1.32", "CV²": "≥ 0.49", "Meaning": "regular timing, volatile sizes"},
    {"Class": "Intermittent", "ADI": "≥ 1.32", "CV²": "< 0.49", "Meaning": "sporadic demand, stable sizes"},
    {"Class": "Lumpy", "ADI": "≥ 1.32", "CV²": "≥ 0.49", "Meaning": "sporadic demand, volatile sizes"},
]), caption="Syntetos-Boylan-Croston classification. ADI = days ÷ days with demand; CV² = squared coefficient of variation of non-zero demand sizes.", max_height=None)

st.markdown("## Forecasting methods")
render_table(pd.DataFrame([
    {"Level": "Aggregate series", "Model": "Seasonal Naive", "Idea": "repeat the value from 7 days earlier"},
    {"Level": "Aggregate series", "Model": "ARIMA / SARIMA", "Idea": "autoregressive moving-average models (SARIMA adds a weekly seasonal part)"},
    {"Level": "Aggregate series", "Model": "Holt-Winters", "Idea": "exponential smoothing with trend and weekly seasonality"},
    {"Level": "Aggregate series", "Model": "LSTM / GRU", "Idea": "recurrent networks forecasting 28 days directly from the last 56 days, with calendar inputs; mean of 5 seeds"},
    {"Level": "Aggregate series", "Model": "Prophet", "Idea": "additive trend + weekly/yearly seasonality + event regressors, MAP estimation"},
    {"Level": "Item level", "Model": "Croston / TSB", "Idea": "smooth demand size and demand interval (TSB: probability) for intermittent demand"},
    {"Level": "Item level", "Model": "XGBoost (pooled)", "Idea": "one gradient-boosted model for all items from lags, rolling means and calendar flags"},
]), caption="Models with saved results; availability is shown on the Overview page", max_height=None)

st.markdown("## Evaluation protocol")
st.markdown(
    f"- **Walk-forward (rolling origin):** {bt['n_folds']} folds, each forecasting {bt['horizon']} days after an expanding training window. No random splitting.\n"
    f"- **Development window:** the first 1,913 days. **Final holdout:** the last {bt['holdout_days']} days (2016-04-25 to 2016-05-22), reserved for one final evaluation of frozen candidates.\n"
    "- **Leakage controls:** models are refitted per fold on that fold's training prefix only; scalers, validation tails and demand classes for model selection are computed inside the training window; calendar facts known in advance are the only future inputs.\n"
    "- **Aggregate and item-level results are separate experiments** and are never ranked together."
)

st.markdown("## Metric definitions")
render_table(pd.DataFrame([
    {"Metric": "MAE", "Definition": "mean of |actual − forecast|", "Reading": "units per day; robust to large misses"},
    {"Metric": "RMSE", "Definition": "√ mean of (actual − forecast)²", "Reading": "penalises large misses more"},
    {"Metric": "WAPE", "Definition": "Σ|actual − forecast| ÷ Σ actual", "Reading": "scale-free; undefined if all actuals are 0"},
    {"Metric": "MASE", "Definition": "MAE ÷ mean |yₜ − yₜ₋₇| on the training window", "Reading": "below 1 beats repeating last week; undefined if the scale is 0"},
    {"Metric": "MAPE", "Definition": "mean of |error| ÷ actual × 100, actual > 0 only", "Reading": "zero-sales days are excluded and counted"},
    {"Metric": "bias", "Definition": "mean of (forecast − actual)", "Reading": "positive = over-forecasting"},
]), caption="Metrics used on the comparison pages", max_height=None)

st.markdown("## Inventory logic")
st.markdown("Protection interval = lead time + review period. Safety stock = the chosen quantile of saved cumulative forecast errors over that interval (floored at 0). "
            "Reorder point = expected demand over the interval + safety stock. The target is a **cycle service level**, not a fill rate, and no cost model is used, so no savings are claimed.")

st.markdown("## Limitations")
st.markdown("One store and department; five folds; recorded sales understate demand when stock runs out; Phase 1 demand labels use the full history; "
            "heavy models depend on optional libraries; Prophet needs a working Stan toolchain; all reported numbers are development results.")

st.markdown("## How to reproduce")
st.markdown("From the project root in PowerShell, with the project's virtual environment active:")
for c in ("python -m pip install -r requirements.txt",
          "python -m pip install -r requirements-phase3.txt",
          "python -m pytest",
          "python -m jupyter nbconvert --to notebook --execute --inplace notebooks/05_intermittent_models.ipynb",
          "python -m jupyter nbconvert --to notebook --execute --inplace notebooks/07_lstm_gru.ipynb",
          "python -m jupyter nbconvert --to notebook --execute --inplace notebooks/08_prophet.ipynb",
          "python -m jupyter nbconvert --to notebook --execute --inplace notebooks/09_model_comparison.ipynb",
          "python -m jupyter nbconvert --to notebook --execute --inplace notebooks/10_inventory_decision_support.ipynb",
          "streamlit run app/app.py"):
    code_block(c)

st.markdown("## Where things are stored")
render_table(pd.DataFrame([
    {"What": "Raw M5 files (not in the repository)", "Where": "data/raw/"},
    {"What": "Panel and processed tables", "Where": "data/interim/, data/processed/"},
    {"What": "Saved forecasts", "Where": config["phase3"]["forecasts_dir"] + "/"},
    {"What": "Saved metrics, diagnostics, run metadata", "Where": config["phase3"]["metrics_dir"] + "/"},
    {"What": "Inventory demonstration outputs", "Where": config["phase3"]["inventory_dir"] + "/"},
    {"What": "Final-holdout records (empty until the one-time run)", "Where": config["phase3"]["final_dir"] + "/"},
    {"What": "Figures and reports", "Where": "reports/"},
]), caption="Artifact folders (repository-relative)", max_height=None)
callout("info", "Final holdout", "The last 28 days are evaluated once, by `python -m src.evaluation.final_holdout`, after the candidates are frozen. "
        "Until then no final result exists, and none is shown anywhere in this app.")
