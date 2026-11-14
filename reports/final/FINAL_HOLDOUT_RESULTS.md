# Final holdout results — single authorized evaluation

**Status:** evaluated once; do not rerun or tune against this window.

- Frozen at: 2026-11-14 20:29:40 UTC.
- Evaluation record timestamp: 2026-11-14 20:31:56 UTC.
- Evaluation count: **1**; override not used.
- Window: **2016-04-25 through 2016-05-22 (28 days)**, after development cutoff 2016-04-24.
- Frozen headline candidates: aggregate LSTM(calendar); item-level class-selected strategy.
- The record hashes the frozen candidates, development selection, configuration, evidence artifacts, forecasts and result tables. All saved forecast rows are successful; date range and 28-day coverage were independently checked.

## Primary results

| Scope | Candidate | Series | Days | MAE | RMSE | MAPE | MAPE exclusions | WAPE | MASE | Bias |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Aggregate | LSTM(calendar) | 1 | 28 | 128.0869 | 183.1430 | 5.0693% | 0 | 0.05434 | 0.49197 | -63.9673 |
| Item level (equal-weight macro) | Class-selected strategy | 823 | 23,044 | 1.683711 | 2.188407 | 62.4531% | 8,432 | 1.007349 | 1.375490 | -0.215173 |

Aggregate and per-item macro metrics use different scales and must not be compared. Item-level MAPE excludes zero actuals; WAPE is total absolute error divided by total actual demand and can exceed 1.0. The item-level strategy was frozen from development evidence; its 1.6837 MAE is close to, but not a claim of superiority over, the pooled calendar XGBoost result (1.6822). This is one future 28-day period, not evidence of broad generalization.

## Frozen item-level candidate comparison

| Model | MAE | RMSE | MAPE | MAPE exclusions | WAPE | MASE | Bias |
|---|---:|---:|---:|---:|---:|---:|---:|
| XGBoost pooled Poisson, event/calendar | 1.704776 | 2.161313 | 56.3960% | 8,432 | 1.091751 | 1.422434 | 0.007896 |
| XGBoost pooled calendar | 1.682242 | 2.185347 | 64.3091% | 8,432 | 1.013566 | 1.365145 | -0.204308 |
| TSB(alpha=0.1,beta=0.1) | 1.752062 | 2.222832 | 58.9589% | 8,432 | 1.072465 | 1.449061 | -0.118899 |
| Seasonal Naive fallback | 2.025039 | 2.701185 | 83.3844% | 8,432 | 1.204937 | 1.671259 | -0.200963 |
| Class-selected strategy | 1.683711 | 2.188407 | 62.4531% | 8,432 | 1.007349 | 1.375490 | -0.215173 |

The frozen class-selection rule produced 20 class/model rows in `final_holdout_class_results.csv`. NoDemand items use the predeclared deterministic SeasonalNaive fallback. Per-class scores, candidate source and class counts are in that CSV. Forecast rows are in `final_holdout_forecasts.csv` and should remain restricted project evidence, not included in the public dashboard bundle.

## Integrity and provenance

The record is `artifacts/final/FINAL_EVALUATION_RECORD.json`; the exact one-time count is recorded at the top level and under `details`. SHA-256 values in that record were verified against `final_holdout_results.csv`, `final_holdout_class_results.csv`, and `final_holdout_forecasts.csv`. All 115,248 prediction rows cover the same 28 dates and report successful forecasts.

Earlier development loaders transiently read reserved sales before slicing to the development window. Those values did not enter fitting, selection, or scoring; current loaders apply the cutoff in the data read. This remains a provenance limitation and is described in `HOLDOUT_DATA_BOUNDARY_AUDIT.md`.
