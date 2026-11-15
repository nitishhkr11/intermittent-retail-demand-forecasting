# RiverDemand final research results

## Current status — 15 November 2026

This is a development-results report with the final holdout reported separately. The protected 28-day period was evaluated once after candidate freeze; record count is 1 and the record is immutable. Do not rerun or tune against it. The later alternative inventory calibration results below supersede earlier statements that those alternatives had not been evaluated. Current acceptance and hashes are in `reports/final/FINAL_CONTENT_CONSISTENCY_AUDIT.md`.


## Scope and holdout status

This report records development-only experiments completed on 14–15 November 2026. The original five-fold results and raw user work remain in place. The final 28 days (2016-04-25 through 2016-05-22) were evaluated once after candidate freeze; final-test metrics are reported separately in `FINAL_HOLDOUT_RESULTS.md` and were not used for selection. Earlier development loaders transiently read reserved sales before slicing; those values were not used for fitting, tuning, candidate selection or scoring. See `HOLDOUT_DATA_BOUNDARY_AUDIT.md` for the correction and limitation.

## Expanded seasonal aggregate experiment

Sixteen disjoint quarterly 28-day windows cover 2012–2015, with all four quarters represented over four years. Minimum training history is 365 days; the first forecast window begins 2012-01-29. Each fold has shared forecast dates across models. ARIMA, SARIMA and Holt-Winters fold-local settings are selected on an inner chronological validation segment that ends before its 28-day scored window. Historical fixed settings were scored on the same windows.

| Model/configuration | Mean MAE | Fold SD | Successful folds |
|---|---:|---:|---:|
| Seasonal Naive | 352.917 | 130.383 | 16/16 |
| ARIMA legacy fixed | 312.979 | 71.009 | 16/16 |
| ARIMA fold-local | 319.253 | 79.367 | 16/16 |
| SARIMA legacy fixed | 254.074 | 48.590 | 16/16 |
| SARIMA fold-local | 254.074 | 48.590 | 16/16 |
| Holt-Winters legacy fixed | 260.392 | 54.187 | 16/16 |
| Holt-Winters fold-local | 260.721 | 55.345 | 16/16 |

No significance claim is made. SARIMA's matching values reflect the selected fold-local order matching its legacy configuration across these folds. Some statsmodels convergence warnings occurred; the run metadata records fit success, but a full per-fit optimizer convergence-status artifact was not added. Historical outputs remain unmodified.

## Expanded intermittent-demand experiment

The full CA_1 / FOODS_3 panel contains 823 products. Seasonal Naive, Croston, SBA and TSB were evaluated over 16 common seasonal folds (52,672 model-item-fold rows; 1,474,816 daily prediction rows). Demand classes were recomputed from each fold's training prefix. The class table keeps NoDemand and InsufficientData separate from the four SBC classes.

| Model | MAE | RMSE | WAPE | MASE | Bias | Failed item-folds |
|---|---:|---:|---:|---:|---:|---:|
| Seasonal Naive | 1.988677 | 2.654698 | 1.399597 | 3.067161 | -0.183201 | 0 |
| Croston alpha 0.1 | 2.290293 | 2.823724 | 1.434288 | 2.970610 | 0.168509 | 2,236 |
| SBA alpha 0.1 | 2.255086 | 2.803063 | 1.395444 | 2.949635 | 0.000413 | 2,236 |
| TSB alpha/beta 0.1 | 1.712453 | 2.180209 | 1.287942 | 2.696463 | -0.067035 | 0 |

MAE/RMSE/WAPE/MASE and bias are equal-weight item-fold summaries for successful forecasts. Croston/SBA failures arose on all-zero training prefixes and were not replaced. WAPE and MASE remain undefined where their denominators are zero. The forecast zero-share and all-zero-forecast counts are saved in the fold metrics and model summary artifacts. Scores are descriptive and not inferential.

SBA is implemented as `Croston forecast × (1 − alpha/2)` and has unit tests for the correction factor, alpha validation and all-zero behavior. A separate canonical five-fold SBA artifact was created for the existing selected 20-item comparison; existing historical files were not rewritten.

Descriptive MAE by fold-local SBC class (means across successful class/fold summaries):

| Class | Mean items per fold | Seasonal Naive | Croston | SBA | TSB |
|---|---:|---:|---:|---:|---:|
| Intermittent | 370.06 | 1.804 | 1.860 | 1.832 | 1.573 |
| Smooth | 94.69 | 3.625 | 3.294 | 3.252 | 3.292 |
| Erratic | 36.75 | 6.035 | 5.262 | 5.162 | 5.193 |
| Lumpy | 181.44 | 2.288 | 2.168 | 2.130 | 1.925 |

These are four baseline methods only; they are not a complete cross-family class-specific selection protocol. `NoDemand` and `InsufficientData` remain separate strata and are not folded into these four labels.

## Inventory recalibration

The source formula was verified: protection interval is lead plus review days; expected demand is forecast sum; errors are actual minus predicted cumulative demand; safety stock is the nonnegative higher empirical quantile; reorder point is expected demand plus safety stock. The 7-day backtest uses prior folds only, at least two earlier calibration folds, separate item/model errors, and cluster-bootstrap uncertainty by item-fold cell to preserve overlap dependence.

| Model | Target | Achieved | 90% cluster interval | Evaluation windows |
|---|---:|---:|---:|---:|
| Croston | 95% | 93.8532% | 93.5825–94.1146% | 211,640 |
| SBA | 95% | 93.8565% | 93.5853–94.1211% | 211,640 |
| Seasonal Naive | 95% | 91.4744% | 91.1647–91.7720% | 253,484 |
| TSB | 95% | 91.1706% | 90.8716–91.4488% | 253,484 |

None reaches the 95% target. Failed or uncalibrated cells remain labeled in the output. This measures cycle service, not fill rate; it does not estimate operating savings or establish a real supplier policy. This original Phase 3 table predates the later alternative calibration study; see the subsequent “Alternative inventory calibration” section for current four-rule results and shared-window comparisons.

## Prophet seasonal interval study

All three configured Prophet variants were scored on the same 16 quarterly 28-day windows (48/48 successful fits). Mean MAE / fold SD / RMSE / nominal 80% day-level coverage / mean interval width were:

| Variant | MAE (SD) | RMSE | Nominal 80% coverage | Mean width |
|---|---:|---:|---:|---:|
| Additive, calendar regressors | 243.63 (97.24) | 301.38 | 65.63% | 554.61 |
| Additive, no regressors | 283.05 (128.39) | 343.79 | 67.63% | 691.03 |
| Multiplicative, calendar regressors | 221.68 (93.77) | 277.89 | 66.07% | 524.93 |

The nominal 80% intervals under-covered on this broader seasonal sample. Fold-level coverage is saved in `artifacts/expanded_prophet_block_conformal/fold_metrics.csv`; it ranges from 14.3% to 92.9% depending on variant and origin. No nominal interval is described as calibrated.

An exploratory block split-conformal adjustment used the first eight chronological folds for calibration and the later eight for validation, treating each 28-day fold as one block to retain within-window dependence. The learned symmetric offsets were 749.20, 930.29 and 724.83 aggregate units for additive, additive/no-regressor and multiplicative variants, respectively. Validation point coverage averaged 99.55–100%; simultaneous whole-window coverage was 87.5–100% across eight validation folds. Mean interval width expanded to 2,007–2,576 units. This is a conservative development-only sensitivity result from eight calibration and eight validation windows, not evidence of nominal 80% operating calibration.

## Complete neural seasonal study

The checkpointed study covered four aggregate variants across the same 16 disjoint quarterly folds, using five seeds per cell (320 networks total). All 64 variant-fold cells completed with finite 28-day forecasts; no fit failed or was skipped. Forty-seven matching checkpoints were reused and the 17 missing cells were resumed. Fold-level and seed-level results, per-seed training diagnostics, histories, and forecasts are saved under `artifacts/expanded_neural/`.

| Variant | Mean ensemble MAE (fold SD) | Mean ensemble RMSE (fold SD) | Mean bias | Mean seed MAE | Mean fit time / cell |
|---|---:|---:|---:|---:|---:|
| LSTM, calendar | 185.09 (60.74) | 235.45 (76.58) | -32.79 | 192.25 | 20.3 s |
| GRU, calendar | 194.77 (55.81) | 246.54 (68.66) | -33.10 | 200.40 | 53.9 s |
| GRU, univariate | 267.77 (79.21) | 338.35 (113.50) | -91.21 | 271.28 | 67.9 s |
| LSTM, univariate | 283.55 (82.48) | 358.44 (113.06) | -94.04 | 287.46 | 349.2 s |

On identical fold dates, calendar inputs lowered LSTM MAE on all 16 folds (mean paired reduction 98.46 units/day, fold SD 62.54) and lowered GRU MAE on 15 of 16 folds (mean paired reduction 73.00, SD 63.40). These are descriptive comparisons for one aggregate series, not a general model superiority claim. One historical LSTM univariate fold took 5,133.6 seconds; it was recovered from its already completed checkpoint and was not retrained. Across all cells, 303 of 320 seed fits stopped early; no fit reached the maximum epoch count. Training-only scaling and validation-tail separation remain enforced by the model implementation and tests.

The additive `artifacts/expanded_neural/completion_verification.json` records all 64 cell statuses, checkpoint and output hashes, runtime versions, and an independent metric audit. Its original holdout-access flag was too broad: the pre-correction aggregate loader transiently read the reserved sales column before slicing. No holdout value entered neural training or scoring; details are in `HOLDOUT_DATA_BOUNDARY_AUDIT.md`.

## Full-panel XGBoost ablations

Five pooled configurations were evaluated on all 823 products over the same 16 quarterly 28-day folds (80/80 pooled model-fold fits). The per-item full-feature model was evaluated on the fixed first-origin stratified sample of 183 products (2,928/2,928 fits). The sample was limited to 33 Lumpy products at the first origin; the other three classes supplied 50 each. Per-item results are reported separately from whole-panel pooled results.

| Pooled configuration | Mean item-fold MAE | RMSE | WAPE | MASE | Bias | Folds / products |
|---|---:|---:|---:|---:|---:|---:|
| Lags only | 1.702 | 2.178 | 1.301 | 2.652 | -0.062 | 16 / 823 |
| Lags + rolling | 1.708 | 2.162 | 1.359 | 2.762 | +0.043 | 16 / 823 |
| Lags + calendar | 1.673 | 2.142 | 1.322 | 2.674 | +0.028 | 16 / 823 |
| Lags + calendar + event/SNAP | 1.680 | 2.122 | 1.387 | 2.747 | +0.169 | 16 / 823 |
| Same features, Poisson objective | **1.664** | **2.098** | 1.385 | 2.676 | +0.184 | 16 / 823 |

On paired fold macro MAE, calendar-only features lowered error versus lags-only on 10/16 folds (mean reduction 0.028 units/item/day); rolling statistics alone lowered error on 5/16 folds and were slightly worse on average. Adding event/SNAP features to calendar lowered MAE on 6/16 folds and was slightly worse on average, although mean RMSE fell from 2.142 to 2.122. The Poisson objective lowered MAE versus square-error on the full feature set on 13/16 folds (mean reduction 0.016). These are small descriptive differences with substantial fold variation; no significance or causal feature-importance claim is made.

On the common 183-item sample, pooled Poisson XGBoost had MAE 2.500 and per-item full-feature XGBoost 2.783. This is a different panel from the all-823 pooled comparison; do not generalize the sampled result to every product. Independent audit recalculated MAE, RMSE and bias from 1,925,504 saved forecast rows across 68,768 item-fold cells: zero mismatches, zero failed cells, and no forecast date beyond 2015-10-28. Full artifacts and figures are under `artifacts/expanded_xgboost/` and `reports/phase3/expanded_xgboost_*.png`.

## Alternative inventory calibration

Four causal methods were compared on the same 42,284 eligible model/item/fold cells and 930,248 overlapping seven-day evaluation windows; each origin used earlier completed folds only. Confidence intervals use item-fold cluster resampling, keeping overlapping windows together.

| Calibration rule | Target | Observed cycle service | 90% cluster interval | Mean safety stock | Mean reorder point | Mean calibration samples (range) |
|---|---:|---:|---:|---:|---:|---:|
| All earlier folds, empirical higher quantile | 95% | 92.475% | 92.321–92.629% | 19.129 | 39.616 | 178.2 (22–330) |
| Recency weighted (2-fold half-life) | 95% | 91.654% | 91.497–91.807% | 17.682 | 38.170 | 178.2 (22–330) |
| Most recent two folds | 95% | 86.724% | 86.511–86.922% | 13.855 | 34.343 | 43.7 (22–44) |
| Same-class normalized errors | 95% | 92.594% | 92.443–92.737% | 17.497 | 37.984 | 46,293 (22–120,054) |

Each method covers 823 products over 14 later evaluation folds (folds 3–16). The safety-stock and reorder-point means are equal-weight means over successful item/model/fold cells, in units for a seven-day protection interval. Overlapping windows are dependent and are not an independent sample size. Each rule has 3,210 failed test cells, 6,584 cells with fewer than two earlier folds, and 594 additional cells without usable prior calibration history or scale. Those cells are excluded, never imputed.

Same-class normalization has the highest point estimate, only 0.12 percentage points above all-history; their cluster-bootstrap intervals overlap. That does not establish a dependable improvement. All-history empirical calibration remains the more defensible dashboard default because it is simpler, calibrates each item/model from its own earlier errors, and performs nearly as well. None of the alternatives reaches 95%.

Fold-level and demand-class results are saved in `artifacts/expanded_inventory_calibration_v2/calibration_fold_summary.csv` and `calibration_class_operating_summary.csv`; model/class cluster-bootstrap intervals are in `calibration_model_class_summary.csv`. Across rules, pooled service varied from 83.29% to 96.51% by evaluation fold. Demand-class rates were 85.57–92.21% for Lumpy, 85.91–93.18% for Intermittent, 85.89–94.37% for Erratic and 91.65–92.90% for Smooth. NoDemand was 96.71% for each rule; the InsufficientData stratum was 51.70% based on only four products and should be read as highly uncertain. Fill rate is not inferred.

## Development candidate selection

The 16-fold aggregate candidate and all four required fold-local demand-class candidates have been selected using primary MAE, secondary RMSE/WAPE/MASE/bias, complete fold coverage and reliability criteria. The exact development-selected candidates, settings, product coverage and hashed evidence are in `artifacts/final/DEVELOPMENT_SELECTION.json` and `reports/final/MODEL_SELECTION_AND_FREEZE_REPORT.md`. The protected freeze is separately recorded in `artifacts/final/FROZEN_CANDIDATES.json`; the one-time evaluation is summarized in `FINAL_HOLDOUT_RESULTS.md`.

Public hosting was not deployed because no authenticated Streamlit/GitHub deployment session was available. See `DEPLOYMENT_VERIFICATION.md`.

## Reproducibility

See `EXPERIMENT_REPRODUCIBILITY.md` for exact commands and artifact hashes. Principal outputs are in `artifacts/expanded_seasonal/`, `artifacts/expanded_intermittent/`, `artifacts/expanded_inventory/`, `artifacts/expanded_prophet_block_conformal/`, `artifacts/expanded_neural/`, and `artifacts/forecasts/sba_item_level_backtest_forecasts.csv`.
