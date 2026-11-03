# Phase 3 Completion Report

Project: RiverDemand, forecasting intermittent retail demand on Walmart M5 (store `CA_1`, department `FOODS_3`).
Scope of this report: what was added on top of the Phase 3 Step 1 (XGBoost) snapshot, what the saved development
artifacts show, what was tested, and what is still open.

Every number below was read from files produced by running the code in this repository (`artifacts/metrics/`,
`artifacts/inventory/`, executed notebooks). No number was estimated, rounded up or taken from memory. **The final
holdout (2016-04-25 to 2016-05-22) has not been evaluated.**

## 1. What was delivered

| Item | File(s) | Result |
|---|---|---|
| LSTM / GRU on the aggregate series | `src/models/lstm_gru.py`, `notebooks/07_lstm_gru.ipynb` | Executed. 4 variants x 5 folds x 5 seeds = 100 networks, 0 failed fits |
| Prophet on the aggregate series | `src/models/prophet_model.py`, `notebooks/08_prophet.ipynb` | Executed. 3 variants x 5 folds = 15 fits, 0 failed, 0 non-converged |
| Intermittent-demand experiment (20 items and all 823 items, fold-local classes) | `src/evaluation/intermittent_experiment.py`, `notebooks/05_intermittent_models.ipynb` | Executed |
| Unified comparison from saved artifacts | `src/evaluation/model_comparison.py`, `notebooks/09_model_comparison.ipynb` | Executed: 11 aggregate models and 7 item-level models, 0 unavailable artifacts |
| Inventory layer and decision-support notebook | `src/inventory/reorder.py`, `notebooks/10_inventory_decision_support.ipynb` | Executed |
| Streamlit dashboard, six pages, light and dark theme | `app/` | Runs; tested in AppTest and in a real browser |
| Controlled final-holdout workflow | `src/evaluation/final_holdout.py`, `artifacts/final/README.md` | Implemented and tested; **not run on the real data** |
| Documentation | `README.md`, `MERGE_AND_RUN_WINDOWS.md`, `IMPLEMENTATION_MANIFEST.md`, this report, `reports/phase3/PHASE3_TEST_RECORD.md` | Written |

Notebook 06 (XGBoost, Step 1) and every Phase 1 and Phase 2 file are unchanged byte for byte; see
`IMPLEMENTATION_MANIFEST.md`.

## 2. Safeguards that apply to every result

- Five expanding-window walk-forward folds of 28 days inside the 1,913-day development window (2011-01-29 to
  2016-04-24); no random splitting. Fold 1 forecasts 2015-12-07 to 2016-01-03, fold 5 forecasts 2016-03-28 to 2016-04-24.
- Neural networks: the scaler is fitted on the inner training part only; the last 56 training days are used only for
  early stopping. The notebook shows a perturbation test (adding 1,000,000 to every validation-tail value changes no
  training sample and not the scaler) and a determinism check (re-fitting LSTM(calendar), fold 5, seed 0 reproduces the
  saved member forecast bit for bit). A test confirms every scaler was fitted before the validation tail.
- Prophet: the same perturbation and determinism checks; regressors (events, SNAP) are calendar information known in
  advance; a no-regressors variant shows what they add.
- The comparison notebook recomputes every metric from the saved forecast tables, validates each file first (same days,
  same actuals, no duplicate model names, no dates after 2016-04-24) and says which file is missing instead of
  substituting anything. Aggregate (thousands of units per day) and item-level (units per item per day) results are in
  separate tables and are never ranked together.
- MAPE is zero-protected (zero-sales days are excluded and counted). MASE divides MAE by the lag-7 seasonal-naive MAE on
  the fold's own training window. WAPE is the sum of absolute errors over the sum of actual sales.
- The dashboard only reads saved artifacts. Tests confirm that no page plots or prints a date inside the holdout window
  as data, that app code cannot call the holdout reader, and that no result files for the holdout exist.

## 3. Development results

Five folds, mean over folds, no significance test. The fold-to-fold standard deviation of MAE (43 to 81 units/day) is
comparable to many of the gaps between models, so "lowest" below describes these five folds only.

### 3.1 Aggregate series (units per day)

| Model | MAE | MAE std | RMSE | WAPE | MASE | Bias (forecast - actual) |
|---|---:|---:|---:|---:|---:|---:|
| LSTM (calendar) | 181.6 | 42.9 | 234.7 | 0.094 | 0.690 | +12.1 |
| Holt-Winters (additive trend) | 193.3 | 62.2 | 266.4 | 0.100 | 0.734 | -58.0 |
| GRU (calendar) | 199.3 | 61.0 | 251.8 | 0.103 | 0.757 | -10.1 |
| GRU (univariate) | 203.1 | 73.8 | 278.8 | 0.106 | 0.771 | -49.5 |
| LSTM (univariate) | 209.7 | 81.1 | 286.2 | 0.109 | 0.796 | -55.7 |
| Prophet (additive) | 222.2 | 44.7 | 272.7 | 0.113 | 0.844 | -127.3 |
| Prophet (additive, no regressors) | 225.9 | 53.7 | 294.6 | 0.116 | 0.858 | -115.5 |
| SARIMA(6,1,1)x(0,0,2,7) | 228.0 | 68.8 | 302.2 | 0.118 | 0.865 | -86.5 |
| Prophet (multiplicative) | 228.0 | 46.1 | 279.0 | 0.116 | 0.866 | -135.3 |
| ARIMA(6,1,1) | 245.6 | 44.7 | 333.4 | 0.124 | 0.933 | -45.6 |
| Seasonal Naive | 256.6 | 52.4 | 329.1 | 0.131 | 0.974 | -79.2 |

What the saved results support:

- MAE, RMSE, WAPE and MASE all pick LSTM (calendar) as the lowest-error model on these folds. Against the best
  Phase 2 model, Holt-Winters, its mean MAE is 11.7 units/day lower (6.1%); it had the lower MAE in 3 of 5 folds.
  The gap is smaller than the fold-to-fold standard deviation of either model.
- Every neural variant and every Prophet variant has a lower mean MAE than Seasonal Naive. LSTM (calendar) and GRU
  (calendar) beat it in 5 of 5 folds; the univariate variants in 4 of 5.
- Calendar information helped the LSTM (181.6 with, 209.7 without; lower in 3 of 5 folds) and barely moved the GRU
  (199.3 with, 203.1 without; lower in 1 of 5 folds). Because the Phase 2 classical models use no calendar
  information, the like-for-like comparison with them is the univariate variants, not the calendar variants.
- Training noise is visible: within a fold the standard deviation of MAE across the five seeds of one network averages 10 to 18
  units/day depending on the variant. The reported figure is the seed-ensemble mean; the average single-seed MAE of LSTM (calendar) is 184.7.
- Prophet's 80% prediction intervals contained the actual value on 75.0% (additive), 81.4% (no regressors) and
  72.9% (multiplicative) of forecast days, against a nominal 80%. Prophet's mean bias is strongly negative (it
  under-forecasts by 115 to 135 units/day on average).
- Four of the 25 LSTM (calendar) networks and one of the 25 GRU (calendar) networks reached the 80-epoch limit; the
  other 95 of 100 stopped early. Total training time was about 21 minutes on one CPU thread.

### 3.2 Item level (20 selected items, macro average, units per item per day)

| Model | MAE | RMSE | WAPE | MASE |
|---|---:|---:|---:|---:|
| TSB (alpha 0.1, beta 0.1) | 1.954 | 2.500 | 1.153 | 1.410 |
| XGBoost (pooled, no event features) | 1.965 | 2.461 | 1.262 | 1.408 |
| XGBoost (pooled, Poisson loss) | 1.976 | 2.461 | 1.301 | 1.402 |
| XGBoost (pooled) | 1.980 | 2.467 | 1.273 | 1.396 |
| XGBoost (per item) | 2.058 | 2.586 | 1.552 | 1.574 |
| Seasonal Naive | 2.342 | 3.083 | 1.254 | 1.607 |
| Croston (alpha 0.1) | 2.380 | 2.876 | 1.341 | 1.553 |

By demand class (macro-mean MAE, five items per class; lowest first):

| Class | Lowest MAE | Next | Highest |
|---|---|---|---|
| Smooth | XGBoost (pooled) 2.144 | XGBoost (pooled, Poisson) 2.165 | Seasonal Naive 2.940 |
| Intermittent | TSB 0.946 | XGBoost (pooled) 0.977 | XGBoost (per item) 1.111 |
| Erratic | XGBoost (per item) 3.373 | TSB 3.388 | Croston 4.390 |
| Lumpy | TSB 1.185 | XGBoost (pooled, no events) 1.269 | Croston 1.594 |

What the saved results support:

- The ranking depends on the demand class and on the error measure. For the Intermittent class MAE, RMSE, WAPE and MASE
  do not agree on one model; WAPE averaged over items exceeds 1 for intermittent items because their total sales are
  small.
- On the 20 items, an always-zero forecast has a lower MAE than Seasonal Naive on 7 items, than Croston on 9 and than
  TSB on 4, which is a reminder that MAE rewards low forecasts on intermittent demand.
- On all 823 items (notebook 05, classes computed from each fold's training prefix) TSB has the lowest mean MAE in all
  four classes, matching the 20-item result in 4 of 4 classes. 110 of 4,115 (item, fold) pairs change class when the
  label comes from the training prefix instead of the full history.
- Differences among TSB and the three pooled XGBoost variants are much smaller than the variation between folds.

### 3.3 Inventory (notebook 10)

Reorder point = expected demand over the protection interval (from a saved forecast) + safety stock (the empirical
quantile of saved cumulative forecast errors at the chosen cycle service level, with a bootstrap interval). Lead time,
review period and service level are user inputs; no cost, holding or shortage model exists, so **no savings are
claimed**.

Backtest of the rule on later folds (60 item-fold cells, 1,320 overlapping windows per model and target):

| Model | Target 95%: achieved | 90% interval |
|---|---:|---|
| Seasonal Naive | 87.7% | 82.4% to 92.3% |
| Croston | 88.9% | 84.8% to 92.6% |
| TSB | 89.2% | 84.6% to 93.3% |
| XGBoost (pooled) | 89.8% | 85.9% to 93.5% |

In all 16 (model, target) combinations the achieved cycle service fell short of the target by more than 2 percentage
points; in none did it exceed the target by more than 2 points. The rule reaches its target in-sample by construction,
so the shortfall is a generalisation gap. The notebook lists plausible reasons (a short noisy error history, level
changes between folds, autocorrelated and skewed errors) and states that it did not isolate them. The service level
used is a cycle service level, not a fill rate.

## 4. Dashboard

`streamlit run app/app.py`. Pages: Overview, Forecast Explorer, Model Comparison, Inventory Decisions, Model Reliability,
About & Methodology.

- Light and dark theme with a working toggle in the sidebar. Theme tokens drive the sidebar, cards, inputs and select
  lists, sliders, tables, tooltips, tabs, expanders, code blocks and Plotly charts. The choice is kept in the session
  across page changes; `?theme=dark` sets the starting theme.
- Accessibility: every text and background pair used in either theme meets the WCAG AA contrast ratio (4.5:1) and
  every chart colour meets 3:1 against the chart surface (checked by tests); status is shown by an icon and a word,
  never by colour alone; chart lines also differ by dash and marker; the best row of a table is marked with text.
- Empty states: a missing artifact produces a message naming the file and the exact PowerShell command that creates
  it. A test runs the whole app against a repository copy with no data and no results and confirms no exception, no
  chart and no table of numbers is drawn.
- Every control present when a page opens (series, model, origin, horizon, metric, lead time, review period, service
  level, the stock-position checkbox) was operated in tests: every option of a select list (first, middle, last), both
  ends of a number input or slider, both states of a checkbox.
- The Forecast Explorer plots exactly the saved forecast and actual values (a test compares the chart data with the
  CSV). It never forecasts beyond 2016-04-24.
- The design was derived from the written specification; no reference screenshots were provided, so the layout is not
  a copy of any mock-up.

## 5. Final holdout: what is frozen and what remains

State of this repository (`python -m src.evaluation.final_holdout status`): **candidates frozen: no; holdout
evaluated: no.** `artifacts/final/` holds only its README.

All development artifacts needed to freeze are present and notebook 09 has been run, so the only things that remain are
your decisions and two commands:

1. Decide that no model, feature or setting will change any more. The candidates are declared in `config.yaml`
   (`final_holdout.candidates`): Seasonal Naive, ARIMA, SARIMA, Holt-Winters, LSTM (calendar), GRU (calendar) and
   Prophet (additive) for the aggregate series; Seasonal Naive, Croston, TSB and XGBoost (pooled) for the item level.
2. `python -m src.evaluation.final_holdout freeze` stores the headline model of each scope (lowest mean
   development-fold MAE) and a hash of the settings. On the present artifacts that rule would pick LSTM (calendar) for
   the aggregate scope and TSB for the item-level scope.
3. `python -m src.evaluation.final_holdout run --confirm EVALUATE-FINAL-HOLDOUT-ONCE` evaluates every frozen candidate
   once. No holdout result exists in this repository and none is quoted in any document.

The freeze was not run here because it is the point after which models may no longer change, and the deadline for
that decision is yours (the faculty final window is 16 to 27 November 2026).

## 6. Tests

See `reports/phase3/PHASE3_TEST_RECORD.md` for the exact counts of the final run, how each group was run, and what
was skipped or deselected and why.

## 7. Limitations

- One store, one department, 20 items and one aggregate series; five overlapping-history folds; no significance test.
- Small fixed neural networks without hyper-parameter search; the calendar variants and the Phase 2 classical models
  use different information.
- Neural and Prophet results can differ in the last digits on another machine or library version. The run
  metadata records Python 3.13, torch 2.15.0, prophet 1.5.0, pandas 3.0.5 on Linux; the project's declared minimum
  is pandas 2.0.
- Sales are censored by stock-outs, so recorded demand understates true demand when an item was out of stock.
- The inventory results rest on illustrative lead times and service levels, use no cost model and cover one store.
- Prophet's bundled Stan executable may fail to start from a project path that contains spaces; notebook 08 reports
  that as a `ProphetBackendError`, with the workaround in `MERGE_AND_RUN_WINDOWS.md`.
- `src/pipeline.py` is still the documented Phase 1 placeholder.
- `notebooks/06_xgboost.ipynb` still states that Phase 3 was not complete (its test asserts that sentence); it was
  deliberately left byte for byte unchanged.
- Notebook 04 already contained sandbox-style paths in some Phase 2 outputs; it is a completed Phase 2 deliverable
  and was not altered.
- The SARIMA fit of fold 4 is sensitive to the library versions it runs under. Re-running the Phase 2 notebook-04
  execution test here re-estimated it and gave a fold 4 MAE of 169.54 instead of the saved 164.73 (folds 1, 2, 3 and 5
  were identical). Every number in this report uses the saved Phase 2 values, which were restored after that test and
  were not changed. That test overwrites the Phase 2 SARIMA artifacts as a side effect; if it does so on your machine
  with different numbers, re-run notebook 09 so the comparison matches (the comparison test says so).
