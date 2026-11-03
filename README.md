# RiverDemand: Forecasting Intermittent Retail Demand

A comparative classical, machine-learning and deep-learning forecasting framework for inventory decision
support, built on the Walmart M5 Forecasting - Accuracy dataset (store `CA_1`, department `FOODS_3`,
823 items, 1,941 days, 2011-01-29 to 2016-05-22).

**Status: Phases 1-3 are implemented and every development result below was produced by running the code in this
repository. The final holdout (the last 28 days, 2016-04-25 to 2016-05-22) has NOT been evaluated.** It is reserved for one
final evaluation of frozen candidates (see "Final holdout").

## What is in the repository

| Part | Where | State |
|---|---|---|
| Data layer, demand classification (ADI / CV²), EDA | `src/data/`, `src/analysis/`, `notebooks/01`-`03` | Phase 1, complete |
| Walk-forward backtesting, metrics, residual diagnostics | `src/evaluation/`, `notebooks/04` | Phase 2, complete |
| Seasonal Naive, ARIMA, SARIMA, Holt-Winters (aggregate series) | `src/models/`, `notebooks/04` | Phase 2, complete |
| Croston, TSB (item level) | `src/models/croston.py`, `tsb.py`, `notebooks/04`, `05` | Phase 2, complete |
| XGBoost: pooled and per-item, leak-safe features, recursive 28-day forecast | `src/models/xgboost_model.py`, `notebooks/06` | Phase 3, complete |
| Intermittent-demand experiment on 20 items and on all 823 items, fold-local demand classes | `src/evaluation/intermittent_experiment.py`, `notebooks/05` | Phase 3, complete |
| LSTM / GRU (PyTorch, CPU, seed ensemble, resumable) | `src/models/lstm_gru.py`, `notebooks/07` | Phase 3, complete |
| Prophet (additive, multiplicative, no-regressors) | `src/models/prophet_model.py`, `notebooks/08` | Phase 3, complete |
| Unified comparison from saved artifacts (aggregate and item level kept apart) | `src/evaluation/model_comparison.py`, `notebooks/09` | Phase 3, complete |
| Inventory layer: reorder point, safety stock, service-level backtest | `src/inventory/`, `notebooks/10` | Phase 3, complete |
| Streamlit dashboard, six pages, light and dark theme | `app/` | Phase 3, complete |
| Controlled final-holdout workflow (freeze, then run once) | `src/evaluation/final_holdout.py` | Phase 3, complete; **not run** |

## Protocol (the rules every result obeys)

- **No random splitting.** Five expanding-window walk-forward folds of 28 days inside the 1,913-day development window
  (2011-01-29 to 2016-04-24). Fold 1 forecasts 2015-12-07 to 2016-01-03; fold 5 forecasts 2016-03-28 to 2016-04-24.
- **No leakage.** Scalers, features, class labels and hyper-parameters are fitted on a fold's training prefix only; tests
  perturb the validation tail and the forecast window and check that nothing the model sees changes.
- **Final holdout untouched.** 2016-04-25 to 2016-05-22 is never loaded by a notebook, a model, or the dashboard.
  Only `src/evaluation/final_holdout.py` can read it, and only after candidates are frozen and an explicit confirmation
  phrase is typed.
- **Aggregate and item-level errors are never ranked together.** The aggregate series is in thousands of units per day;
  one item is a few units per day. They live in separate tables, tabs and conclusions.
- **Nothing is invented.** A model without a saved forecast file is absent from every table and chart, with a message
  naming the file and the command that creates it. A failed fit is reported, never replaced by a naive forecast.

## Development results (five folds, no significance test)

All numbers are recomputed from the saved forecast files by `notebooks/09_model_comparison.ipynb`. "Lowest" describes
these five overlapping-history folds only: the spread between folds is as large as many of the gaps between models,
and no significance test was run.

**Aggregate series (units per day, mean of five folds; std = standard deviation across folds)**

| Model | MAE | MAE std | RMSE | WAPE | MASE |
|---|---:|---:|---:|---:|---:|
| LSTM (calendar inputs) | 181.6 | 42.9 | 234.7 | 0.094 | 0.690 |
| Holt-Winters (additive trend) | 193.3 | 62.2 | 266.4 | 0.100 | 0.734 |
| GRU (calendar inputs) | 199.3 | 61.0 | 251.8 | 0.103 | 0.757 |
| GRU (univariate) | 203.1 | 73.8 | 278.8 | 0.106 | 0.771 |
| LSTM (univariate) | 209.7 | 81.1 | 286.2 | 0.109 | 0.796 |
| Prophet (additive) | 222.2 | 44.7 | 272.7 | 0.113 | 0.844 |
| Prophet (additive, no regressors) | 225.9 | 53.7 | 294.6 | 0.116 | 0.858 |
| SARIMA(6,1,1)x(0,0,2,7) | 228.0 | 68.8 | 302.2 | 0.118 | 0.865 |
| Prophet (multiplicative) | 228.0 | 46.1 | 279.0 | 0.116 | 0.866 |
| ARIMA(6,1,1) | 245.6 | 44.7 | 333.4 | 0.124 | 0.933 |
| Seasonal Naive | 256.6 | 52.4 | 329.1 | 0.131 | 0.974 |

**Item level (20 selected items, macro average: every item counts equally; units per item per day)**

| Model | MAE | RMSE | MASE |
|---|---:|---:|---:|
| TSB (alpha 0.1, beta 0.1) | 1.954 | 2.500 | 1.410 |
| XGBoost (pooled, no event features) | 1.965 | 2.461 | 1.408 |
| XGBoost (pooled, Poisson loss) | 1.976 | 2.461 | 1.402 |
| XGBoost (pooled) | 1.980 | 2.467 | 1.396 |
| XGBoost (per item) | 2.058 | 2.586 | 1.574 |
| Seasonal Naive | 2.342 | 3.083 | 1.607 |
| Croston (alpha 0.1) | 2.380 | 2.876 | 1.553 |

Differences among TSB and the three pooled XGBoost variants are far smaller than the fold-to-fold variation; the
per-class tables in notebook 09 show that the ranking changes with the demand class and with the error measure.

**Inventory (notebook 10, illustrative assumptions, no cost model, no savings claimed).** A reorder point built from
saved forecasts and saved forecast errors reached lower cycle service on later folds than its target in all 16
(model, target) combinations (for example 87.7% to 89.8% achieved at a 95% target). The notebook reports the
interval and lists plausible reasons without isolating them.

## Setup (Windows, PowerShell)

```powershell
cd "C:\path\to\intermittent-retail-demand-forecasting"
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements-phase3.txt      # requirements.txt + torch + prophet + streamlit + plotly
```

`requirements.txt` alone is enough for Phases 1-2 and XGBoost. The heavy libraries are optional and each one only
affects the part that uses it: `torch` (notebook 07), `prophet` (notebook 08), `streamlit` and `plotly` (the dashboard).
If `torch` or `prophet` is missing, the model raises an ImportError that names the package and the exact install
command; nothing else is substituted and labelled LSTM, GRU or Prophet. The tests that need an optional library skip
themselves, with that command, when it is absent.

Place the raw M5 files `sales_train_evaluation.csv`, `calendar.csv` and `sell_prices.csv` in `data/raw/`. They are not in
Git and never should be.

**Prophet on Windows:** Prophet starts a bundled Stan executable. If the project (and its `.venv`) sits in a path with
spaces, such as `D:\My Folder\project`, Stan can fail to start. The saved Prophet results do not need it. To re-run notebook 08,
use a virtual environment outside the spaced path (for example `python -m venv C:\venvs\riverdemand`), see
`MERGE_AND_RUN_WINDOWS.md`.

## Run order

```powershell
# Phase 1 data (needs the raw files)
python -m src.data.preprocessor
python -m src.analysis.demand_classification

# Phase 2 and 3 development artifacts (each notebook writes its own files; run in this order)
python -m jupyter nbconvert --to notebook --execute --inplace notebooks/04_classical_models.ipynb
python -m jupyter nbconvert --to notebook --execute --inplace notebooks/05_intermittent_models.ipynb
python -m jupyter nbconvert --to notebook --execute --inplace notebooks/06_xgboost.ipynb
python -m jupyter nbconvert --to notebook --execute --inplace notebooks/07_lstm_gru.ipynb      # about 20 minutes on one CPU thread; resumable
python -m jupyter nbconvert --to notebook --execute --inplace notebooks/08_prophet.ipynb
python -m jupyter nbconvert --to notebook --execute --inplace notebooks/09_model_comparison.ipynb
python -m jupyter nbconvert --to notebook --execute --inplace notebooks/10_inventory_decision_support.ipynb

# Dashboard
streamlit run app/app.py

# Tests
python -m pytest -m "not slow"        # minutes, no notebook re-execution
python -m pytest                       # also re-executes notebooks (slow tests)
```

Notebook 07 saves a checkpoint after every (variant, fold) in `artifacts/models/lstm_gru_checkpoints/` (git-ignored). If a
run is interrupted, re-running it continues from the checkpoints; a checkpoint is used only when the settings, the data
and the library versions are unchanged. The executed notebook states how many folds came from checkpoints.

## Dashboard

`streamlit run app/app.py` opens six pages: **Overview**, **Forecast Explorer**, **Model Comparison**, **Inventory
Decisions**, **Model Reliability** and **About & Methodology**. The sidebar has a working **Dark mode** switch (it changes
the sidebar, cards, inputs, tables, tooltips and charts, and survives page changes), a data-scope summary and the
final-holdout status.

- The dashboard only **reads saved artifacts**. It trains nothing and never forecasts beyond 2016-04-24.
- If an artifact is missing, the page says which file, why it matters and the exact PowerShell command that creates it.
- Aggregate and item-level results appear in separate tabs and are never ranked against each other.
- Each page has a collapsed provenance area (file, modification time, content hash).
- Open directly in dark mode with `http://localhost:8501/?theme=dark`.

The design was derived from the written specification; no reference screenshots were provided.

## Final holdout

```powershell
python -m src.evaluation.final_holdout status     # read-only: what is frozen, what is evaluated, what is missing
python -m src.evaluation.final_holdout freeze     # writes artifacts/final/frozen_candidates.json (development evidence only)
python -m src.evaluation.final_holdout run --confirm EVALUATE-FINAL-HOLDOUT-ONCE
```

`freeze` picks each scope's headline model by lowest mean development-fold MAE and stores a hash of the settings.
`run` refuses to start without the exact phrase, without a freeze, if the settings changed since the freeze, if the
evaluation already happened (unless `--override-existing`, which archives the old files), or if a model name drifted.
Forecasts are made before the actuals are first read. A candidate that fails is reported as failed, never substituted.
Aggregate and item-level results are reported separately. **Nothing has been frozen or evaluated in this repository.**

## Repository structure

```
app/                  Streamlit dashboard: app.py (entry), views/ (six pages), components/, theme.py, data_access.py
artifacts/            generated results (git-ignored): forecasts/, metrics/, inventory/, models/, final/
config.yaml           every setting; Phase 3 sections: phase3, neural, prophet, xgboost, final_holdout, dashboard
data/                 raw/ (never committed), interim/, processed/
notebooks/            01 data -> 10 inventory decision support
reports/              phase1/, phase2/, phase3/ (figures and write-ups)
src/
  data/               loading, scoping, validation, splitting, aggregate-series loader
  analysis/           demand classification, EDA, model diagnostics
  models/             baseline, arima, sarima, holt_winters, croston, tsb, xgboost_model, lstm_gru, prophet_model
  evaluation/         backtesting, metrics, residuals, forecast artifacts, model comparison,
                      intermittent experiment, final holdout
  inventory/          reorder point, safety stock, service-level backtest
  utils/              config, paths, provenance helpers
tests/                mirrors src/ and app/
.streamlit/           dashboard theme defaults
```

## Limitations and known caveats

- One store, one department, 20 selected items plus the aggregate series: the results do not generalise to other stores,
  categories or seasons.
- Five walk-forward folds with overlapping history are a small sample. Fold-to-fold variation is reported next to every
  mean; no statistical significance test was run, so no model is declared significantly better.
- The neural networks are small, fixed-size starting points (one layer, 32 units, no hyper-parameter search). The
  `calendar` variants use event and SNAP information that the Phase 2 classical models do not use, so differences between
  families mix model class and information set; the `univariate` variants are the like-for-like comparison.
- Neural and Prophet results depend on library versions and the platform; a re-run on another machine can differ in the
  last digits. The run metadata records the versions that produced the saved numbers (Python 3.13, torch 2.15.0,
  prophet 1.5.0, pandas 3.0.5, Linux).
- Demand classes for model selection are fold-local (computed from each fold's training prefix); the Phase 1 labels were
  computed on the full history, so 110 of 4,115 (item, fold) pairs change class.
- Sales are censored by stock-outs, so true demand can be higher than recorded. The inventory layer uses illustrative
  lead times and service levels, no cost model and claims no savings.
- `src/pipeline.py` is a documented placeholder; the real entry points are the commands in "Run order".
- `notebooks/06_xgboost.ipynb` still carries its original Step 1 sentence that Phase 3 was not complete; it was left
  byte-for-byte unchanged because it is a completed deliverable and a test asserts it.

## License

MIT, see [LICENSE](LICENSE).
