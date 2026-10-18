# Forecasting Intermittent Retail Demand

A comparative classical–machine learning framework for inventory decision
support, built on the Walmart M5 Forecasting–Accuracy dataset.

**Status: Phase 1, Stage 3 (EDA) — no forecasting results yet.**

## Problem

Retail demand for individual SKUs is often intermittent — many zero-sales
days punctuated by irregular spikes — which breaks the assumptions behind
many standard forecasting methods. This project:

1. Classifies products by demand pattern (Smooth / Intermittent / Erratic /
   Lumpy) using ADI and CV².
2. Compares classical, intermittent-demand-specific, machine-learning, and
   deep-learning forecasters against each pattern, using walk-forward
   backtesting (no random train/test splitting).
3. Asks whether the best-performing model depends on the demand pattern.
4. Converts the resulting forecasts into inventory reorder recommendations.

Scope is deliberately narrow: one store (`CA_1`), one department
(`FOODS_3`), ~20-25 representative products plus one dense aggregate
series — not the full M5 catalog.

## Current stage: Stage 4

Implemented so far (all part of the faculty's Phase 1 / synopsis prep):
- **Stage 1 — data layer** (`src/data/`): loads the raw M5 files, scopes
  them to `CA_1` / `FOODS_3`, converts wide to long format, joins
  calendar and price information, validates the result.
- **Stage 2 — demand classification** (`src/analysis/demand_classification.py`,
  `src/analysis/eda.py::build_aggregate_series`): computes ADI and CV²
  for the *entire* CA_1/FOODS_3 catalog, classifies every item into
  Smooth / Intermittent / Erratic / Lumpy (plus explicit `NoDemand` /
  `InsufficientData` labels for series the quadrant rule can't score),
  selects up to 5 representative items per category without fabricating
  any, and builds the full-catalog aggregate daily series.
- **Stage 3 — EDA** (`src/analysis/eda.py`): dataset overview, missing-value
  analysis, rolling statistics, day-of-week pattern, demand-category
  distribution, ADI/CV² quadrants, and representative-product plots.
  Figures save to `reports/phase1/`.
- **Stage 4 — synopsis preparation**: `reports/phase1/synopsis_notes.md`,
  `references.md`, and `figure_manifest.md`, plus the synopsis slide
  deck. No forecasting model of any kind — Phase 2/3 modeling is
  presented as a planned methodology, not a result.
- `notebooks/01_data_understanding.ipynb`, `notebooks/02_eda.ipynb`,
  `notebooks/03_demand_classification.ipynb`.

Not yet implemented: any forecasting model, backtesting, the inventory
layer, and the Streamlit dashboard. `notebooks/04`-`09` remain
placeholders for later phases.

## Setup

```bash
pip install -r requirements.txt
```

Download the **M5 Forecasting – Accuracy** dataset from Kaggle and place
`sales_train_evaluation.csv`, `calendar.csv`, and `sell_prices.csv` into
`data/raw/`. Raw data is not committed to this repo (see `.gitignore`).

## Running the pipeline

```bash
python -m src.data.preprocessor              # Stage 1: raw -> data/interim/panel_ca1_foods3.parquet
python -m src.analysis.demand_classification # Stage 2: classification + selection + aggregate series
```

Then open `notebooks/01_data_understanding.ipynb`,
`notebooks/02_eda.ipynb`, and `notebooks/03_demand_classification.ipynb`
to explore the results. Stage 3's figures also save to `reports/phase1/`.

## Tests

```bash
pytest tests/
```

## Repository structure

```
src/
├── pipeline.py   # full orchestration across all layers (Phase 2/3)
├── data/         # loading, scoping, cleaning, validation, splitting (Phase 1-2)
├── analysis/     # demand classification, EDA (Phase 1); model diagnostics (Phase 2)
├── models/       # baseline/arima/sarima/holt_winters/croston/tsb (Phase 2);
│                 # xgboost/prophet/lstm_gru (Phase 3) -- common interface throughout
├── evaluation/   # backtesting, metrics, residuals, comparison (Phase 2)
├── inventory/    # reorder point / safety stock (Phase 3)
└── utils/        # config, paths, logging, plotting
app/              # Streamlit dashboard: pages/ + components/ (Phase 3)
notebooks/        # 01 data understanding -> 09 model comparison
artifacts/        # models, forecasts, metrics, residuals, inventory (generated, gitignored)
reports/          # phase1/ phase2/ phase3/ final/ write-ups
tests/            # mirrors src/
```

Only `src/data/`, `src/analysis/demand_classification.py`,
`src/analysis/eda.py`, `src/utils/`, `tests/data/`, `tests/analysis/`,
and `notebooks/01`-`03` have real logic right now — everything else is
a scaffolded placeholder for its phase (see each file's docstring).

## Roadmap

Phase numbers below follow the faculty's Synopsis/Mid-Sem/Final
rubric, not the build-stage numbering used in commit history.

- **Phase 1** *(current)* — data understanding, preprocessing, demand
  classification, EDA, synopsis preparation. No forecasting model or
  forecast result of any kind.
- **Phase 2** — Seasonal Naive, ARIMA, SARIMA, Holt-Winters, Croston,
  TSB, walk-forward backtesting, RMSE/MAE/MAPE, model comparison.
- **Phase 3** — XGBoost, LSTM/GRU, Prophet (where appropriate),
  comparative analysis, inventory decision support, the Streamlit
  dashboard, final report/codebase.

## License

MIT — see [LICENSE](LICENSE).
