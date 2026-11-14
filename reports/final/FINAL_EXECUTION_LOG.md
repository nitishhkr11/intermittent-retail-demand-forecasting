# RiverDemand Final Execution Log

Execution began: 2026-11-14 (Asia/Kolkata)
Repository: `D:\MY PATH\TIME SERIES PROJECT\intermittent-retail-demand-forecasting`
Branch: `finalization/riverdemand-research-completion`
Starting commit: `6ef0824f7d6c62c1ea2af92923a5553c12c29a9e`

## Baseline and preservation

- Captured the pre-execution tracked worktree diff and status in `reports/final/_execution_checkpoint_20261114/`.
- Starting worktree already contained modified notebooks and untracked documentation, report, presentation, and dashboard-review artifacts. These are existing user work and must be preserved.
- Available resources at start: 16 logical CPU cores, approximately 4.2 GiB free RAM, and 27.8 GiB free on D:.
- Raw files found: M5 calendar, `sales_train_evaluation.csv`, and `sell_prices.csv`.
- Read-only holdout status command reported development 2011-01-29 through 2016-04-24 (1,913 days), holdout 2016-04-25 through 2016-05-22 (28 days), candidates not frozen, and holdout not evaluated.
- `src/pipeline.py` is currently a placeholder. No `.github` workflow exists. Existing saved artifacts include baseline experiments, but no expanded seasonal fold manifest or candidate freeze was found.

## Execution entries

### Expanded seasonal aggregate experiment

- Prepared 16 disjoint quarterly 28-day development windows across 2012–2015; the first window begins 2012-01-29 to satisfy the 365-day minimum training history.
- Fold manifest: `artifacts/expanded_seasonal/fold_manifest.json`; SHA-256 `24575c7c1c1131a5c7fd630c735965f063b52eafc82a53dc103c3de0b0719b11`.
- Completed 112 model-fold metric rows and 3,136 daily forecast rows, with zero failed folds. Results include fixed legacy settings and fold-local settings selected on an inner chronological validation slice.
- Mean MAE (aggregate units/day): Seasonal Naive 352.917; ARIMA legacy 312.979; SARIMA legacy and fold-local 254.074; Holt-Winters legacy 260.392; ARIMA fold-local 319.253; Holt-Winters fold-local 260.721. These are descriptive; no significance test was run.
- Saved forecast, fold metrics, summary, legacy-vs-local comparison and run metadata under `artifacts/expanded_seasonal/`.

### Expanded all-item intermittent experiment

- Evaluated all 823 CA_1 / FOODS_3 series on the same 16 seasonal folds: 52,672 item-fold-model metric records and 1,474,816 daily forecast records.
- Fold manifest SHA-256 `d89952d80a88897d711e8cb190901ef5957eb8518797c55addc7444d2b2b9c67`; shared seasonal manifest SHA-256 `24575c7c1c1131a5c7fd630c735965f063b52eafc82a53dc103c3de0b0719b11`.
- Mean equal-weight item-fold MAE / RMSE / WAPE / MASE / bias: Seasonal Naive 1.988677 / 2.654698 / 1.399597 / 3.067161 / -0.183201; Croston 2.290293 / 2.823724 / 1.434288 / 2.970610 / 0.168509; SBA 2.255086 / 2.803063 / 1.395444 / 2.949635 / 0.000413; TSB 1.712453 / 2.180209 / 1.287942 / 2.696463 / -0.067035.
- There are 4,472 failed Croston/SBA item-fold fits (2,236 each), arising from all-zero training prefixes; no forecast was imputed. Seasonal Naive and TSB had no failed fits. Undefined WAPE/MASE remain missing where their denominator is zero.
- Fold-local class summaries retain `NoDemand` and `InsufficientData` strata. The saved run metadata states `holdout_accessed: false`.

### Added SBA and inventory calibration

- Added an SBA forecast implementation and configured default alpha 0.1. Added unit tests and updated the gated holdout registry/factory; this does not freeze or execute the final holdout.
- The 95% 7-day cycle-service calibration used earlier seasonal folds only, separately per item and model.

Completed separately per item and model from the expanded quarterly forecasts. The 7-day protection interval was calibrated using at least two earlier folds; each later-fold test was never used to fit its own safety stock. Overlapping test windows were retained and uncertainty used a cluster bootstrap over item-fold cells.

| Model | 95% target | Achieved cycle service | 90% cluster-bootstrap interval | Test windows | Valid item-fold cells |
|---|---:|---:|---:|---:|---:|
| Croston (alpha 0.1) | 95.0% | 93.8532% | 93.5825–94.1146% | 211,640 | 9,620 |
| SBA (alpha 0.1) | 95.0% | 93.8565% | 93.5853–94.1211% | 211,640 | 9,620 |
| Seasonal Naive | 95.0% | 91.4744% | 91.1647–91.7720% | 253,484 | 11,522 |
| TSB (alpha/beta 0.1) | 95.0% | 91.1706% | 90.8716–91.4488% | 253,484 | 11,522 |

This does not meet the configured 95% target. 6,584 cells had fewer than two earlier calibration folds, 3,210 were unevaluable because test forecasts had failed/missing values, and 594 had no usable calibration samples. These statuses are retained in the cell output and are excluded from achieved-rate denominators rather than imputed. The selected all-item seasonal folds do not produce an independent full data history across every calendar day, and the 7-day result should be read as an empirical development backtest, not a service guarantee.

### Pipeline, dashboard, and CI changes

- Replaced the placeholder `src/pipeline.py` with a development-only staged workflow (`data`, `classify`, `seasonal`, `intermittent`, `inventory`, `development`) and an opt-in `--resume` that leaves existing summaries unchanged. The holdout command is not in the stage registry.
- Added `.github/workflows/ci.yml` with dashboard tests and the requested non-slow suite.
- Added dashboard-only pinned dependencies in `app/requirements.txt` and a raw-data-free dashboard bundle builder. The archive was rebuilt after refreshing comparison outputs; its current size and hash are recorded in the final QA entry below.
- Added an expanded development-study tab to the model comparison dashboard. It reads saved CSV outputs, identifies their separate scopes, shows model failures, and displays inventory calibration when present. It does not train or use holdout values.
- Added `src.evaluation.sba_backtest` and a separate five-fold SBA artifact for the canonical 20-item benchmark. The file is registered as a new source; previous forecast artifacts are not overwritten. Saved artifact SHA-256: `b9b2944b35100a3d496e14396910251dc721937389a3892622ac081522e63f23`.
- Initial dashboard run before browser dependencies were installed: 135 passed, 1 skipped. The final browser-enabled run and requested non-slow suite are recorded below.
- First non-slow run (12m 04s) exposed three consistency failures after SBA was registered: one test still asserted only three item artifact sources, and two tests found that comparison hashes/audit outputs had not yet been regenerated. The source assertion and notebook expectations were updated to include SBA; notebook 09 was re-executed to refresh comparison tables, audit and input hashes.
- Focused regression run `.venv\Scripts\python.exe -m pytest tests/evaluation/test_model_comparison.py tests/notebooks/test_comparison_notebook.py tests/test_pipeline.py tests/evaluation/test_final_holdout.py tests/models/test_sba.py -q`: 80 passed, 0 failed (7.62 seconds).
- The exact non-slow suite passed after those fixes: 936 passed, 1 skipped, 4 deselected, 1 warning in 729.99 seconds (ZMQ Proactor warning).
- Pipeline smoke command `.venv\Scripts\python.exe -m src.pipeline validate` verified all three configured raw files and reproduced the recorded SHA-256 values without loading raw sales values. `.venv\Scripts\python.exe -m src.pipeline development --resume` exited successfully and skipped all five development stages because their declared outputs were already present; existing outputs were left unchanged.

### Final dashboard UI and package QA

- Corrected the inventory browser check to wait for the saved-error bootstrap result before checking the lower explanatory panel. Fixed the narrow-screen screenshot capture to close Streamlit's initially open navigation drawer before capturing tablet/mobile content.
- Final browser-enabled `.venv\Scripts\python.exe -m pytest tests/app -q`: 198 passed in 416.71 seconds. Follow-up recapture of all 24 tablet/mobile page-theme combinations: 24 passed, 39 deselected. Focused model-comparison, pipeline, final-holdout, SBA, and theme regression tests: 127 passed in 9.25 seconds.
- Real Chromium computed styles at 1440×900 in light mode: app/main/sidebar/header/main container backgrounds all `rgb(255, 255, 255)`. Main block container width 1,168 px; computed `max-width: 1280px`. The responsive browser assertions passed at 1440, 1366, 768 and 390 px in both themes, with no horizontal page overflow.
- Captures: 48 final screenshots in `reports/final/dashboard_browser_qa/after/` (six pages × two themes × four viewports). Earlier before/after captures remain in `reports/final/dashboard_ui_review/before/` and `reports/final/dashboard_ui_review/after/`.
- The initial rebuilt bundle listed at this point was later superseded after an additional data-boundary audit. The correction and current safe archive are recorded in the addendum below.
- Final holdout status remains `candidates frozen: no`, `holdout evaluated: no`. No final-period values were used.
- Final exact non-slow command `.venv\Scripts\python.exe -m pytest -q -m "not slow"`: 936 passed, 67 deselected, 1 ZMQ Proactor warning in 750.63 seconds.

### Access and current holdout state

- Re-ran `python -m src.evaluation.final_holdout status`: development and holdout dates remain as above; all currently configured candidate development artifacts are present, candidates are not frozen and the holdout is not evaluated. Broader directive prerequisites (especially seasonal neural/Prophet and XGBoost ablations and class-specific selection) remain incomplete, so no freeze or holdout evaluation was attempted.
- `git ls-remote origin HEAD` succeeded; GitHub CLI `gh` is not installed. No deployment credentials or authenticated hosting session have been verified. No push, deployment, external message, or holdout run has occurred.

### Additional seasonal research and deployment-data boundary correction

- Expanded Prophet study: 3 variants × 16 shared quarterly folds = 48 successful fits, 0 failures. Nominal 80% day coverage was 65.63% additive, 67.63% additive/no-regressors and 66.07% multiplicative. The first eight folds calibrated 28-day block-max conformal offsets and the later eight validated them; validation average point coverage was 99.55–100%, simultaneous whole-fold coverage 87.5–100%, with mean interval widths 2,007–2,576 aggregate units. This is an exploratory wide-interval sensitivity result, not an operating guarantee. Outputs are in `artifacts/expanded_prophet_block_conformal/`.
- Expanded LSTM/GRU run declared 4 variants ×16 folds ×5 seeds (320 networks), checkpointed 47/64 variant-fold cells, then stopped after one fit required 5,133.6 seconds. A non-selection pilot summarizes the first 11 fully common folds (55 seeds/variant) in `artifacts/expanded_neural_partial/`; 17 cells remain incomplete. The full seasonal neural requirement is therefore only partially complete.
- The additional packaging audit found that the earlier archive contained full-history sales parquet files, including reserved-period rows, despite the dashboard not evaluating those rows. That archive was moved to `reports/final/_superseded/dashboard_data_bundle_INCLUDES_RESERVED_PERIOD_DATA_DO_NOT_DEPLOY.zip` and marked not for deployment. This is a correction to the earlier statement that no holdout targets were in the archive; the archive included them, although no holdout scoring or model selection used them.
- Added `src/deployment/development_data.py`, which reads sales with a Parquet predicate ending at 2016-04-24, creates development-only dashboard parquet files, and records `holdout_dates_included: false`. Dashboard data access now prefers those sanitized files, and the overview no longer displays counts derived from the reserved period.
- Current safe archive: `reports/final/dashboard_data_bundle_development_only.zip`, 4,137,819 bytes, SHA-256 `ae25fbe70f9e2ebeb433172a1afbddd06912cccf2cf128d90cac0b6d17aa6d30`. It has 75 manifest-listed files; all embedded hashes verify. It contains no full-history sales inputs, and its three sales data files end on 2016-04-24. Deployment remains unverified due absent account authorization.
- Added the Prophet interval table to the expanded dashboard study. Focused browser verification for the new section passed (1 passed); development-data boundary tests and Prophet calibration tests passed (7 passed). The full exact test suites are rerun after this addendum's code changes below.
- Final post-correction dashboard suite `.venv\Scripts\python.exe -m pytest tests/app -q`: 199 passed in 401.39 seconds; 48 screenshots refreshed.
- Final post-correction required suite `.venv\Scripts\python.exe -m pytest -q -m "not slow"`: 943 passed, 68 deselected, 1 ZMQ Proactor warning in 600.05 seconds. The seven new development-data/Prophet tests are included.


### Holdout prerequisite lock and current state — 14 November 2026

- Corrected the status path that previously suggested the aggregate/item registry alone was sufficient to freeze. The workflow now refuses freeze while the broader directive prerequisites are incomplete and prints the missing studies without offering freeze/run commands.
- Added regression coverage for the live lock. `.venv\Scripts\python.exe -m pytest tests/evaluation/test_final_holdout.py -q`: 36 passed in 4.30 seconds.
- Rechecked `.venv\Scripts\python.exe -m src.evaluation.final_holdout status`: candidates frozen `no`, holdout evaluated `no`; it lists XGBoost ablations, full neural seasonal coverage, class-specific candidate selection and alternative inventory evaluation as blockers. No holdout result or manifest was created.
- Existing presentation and technical report files have not yet been synchronized with all latest study results. This remains open work; no claim of final content synchronization is made.
- Final post-gate required suite `.venv\Scripts\python.exe -m pytest -q -m "not slow"`: 944 passed, 68 deselected, 1 ZMQ Proactor warning in 587.91 seconds. This includes the new prerequisite-lock regression test. `git diff --check` reported no whitespace errors.

### Resumed complete LSTM/GRU seasonal experiment — 14 November 2026

- Read the existing 16-fold manifest, current experiment fingerprint and checkpoint envelope before running. All 47 completed cells had matching fingerprints and were reusable. Missing cells were LSTM calendar 13–16, LSTM univariate 13–16, GRU calendar 13–16, and GRU univariate 12–16.
- Resumed with `.venv\Scripts\python.exe -m src.evaluation.expanded_neural run`. The runner reused 47 saved cells and completed 17 cells. Final status: 64/64 successful, 0 failed, 0 skipped; 320 seed results and 1,792 daily forecasts.
- Summary mean ensemble MAE: LSTM calendar 185.092 (fold SD 60.742); GRU calendar 194.767 (55.810); GRU univariate 267.768 (79.206); LSTM univariate 283.554 (82.479). Calendar reduced LSTM MAE in 16/16 paired folds and GRU in 15/16.
- Independently recomputed MAE/RMSE/bias from each 28-row forecast group; all 64 records matched `fold_metrics.csv` to 1e-9. All 320 seed forecast groups are finite. Diagnostics show 303/320 seed fits stopped early and 0 hit the maximum epoch count.
- `artifacts/expanded_neural/completion_verification.json` records all status counts, environment, data/calendar/fold/code hashes, output/checkpoint hashes and `holdout_accessed: false`. Its current SHA-256 is `a39b73b0ca94bb9c5ad086e3e50ebeeb346dc7208c279beeca98a9c99716c62d`.
- The holdout gate was updated to remove this completed prerequisite. XGBoost ablations, class-specific selection, and alternative inventory calibration/class sensitivity still block candidate freezing.

### Holdout loader boundary correction — 14 November 2026

- Source audit found that earlier `load_aggregate_development` and `load_all_item_development_sales` implementations read sales across the full stored history before slicing to development dates. The completed experiments only received/scored the development slices, but reserved-period sales were transiently read into memory; older `holdout_accessed: false` metadata did not distinguish this.
- Updated both loaders to determine the split from date-only reads and apply `date <= 2016-04-24` to the Parquet sales read. The calendar loader continues to read calendar columns only with the cutoff.
- Added sentinel tests proving that the Parquet sales read is filtered before the holdout values can be returned. Focused suite: 115 passed across XGBoost feature, aggregate loader and item loader tests.
- Created `reports/final/HOLDOUT_DATA_BOUNDARY_AUDIT.md`. This correction is explicit: the holdout has not been evaluated, but prior code transiently loaded its sales values. No final forecasts, scores, candidate decisions or tuning used them. The final evaluation remains blocked and no candidate was frozen.


### Final single-evaluation and delivery pass — 15 November 2026

- Completed the prerequisite gate, refreshed the development selection, froze four unique candidate families with an aggregate LSTM(calendar) headline and class-selected item strategy, then ran the protected final-holdout command exactly once.
- The 28-day holdout (2016-04-25–2016-05-22) completed successfully. `FINAL_EVALUATION_RECORD.json` records evaluation_count=1, no override, frozen/config/evidence hashes, and matching hashes for the three output CSVs. All 115,248 saved prediction rows succeeded.
- Aggregate headline metrics: LSTM(calendar), MAE 128.0869, RMSE 183.1430, MAPE 5.0693%, WAPE 0.05434, MASE 0.49197, bias -63.9673. Item class-selected strategy: macro MAE 1.683711, RMSE 2.188407, MAPE 62.4531% (8,432 zero actuals excluded), WAPE 1.007349, MASE 1.375490, bias -0.215173. Aggregate and macro metrics are separate scales.
- `pytest tests/app -q`: 214 passed. `pytest -q -m "not slow"`: 958 passed, 83 deselected, 0 failed, one Windows ZMQ Proactor warning (600.13 seconds). Final-holdout/deployment focused verification: 37 passed.
- Rebuilt `reports/final/dashboard_data_bundle_final_evaluation.zip` from the final results. It includes only compact summary metrics/class rows and the record, not row-level holdout forecasts.
- Completed synchronized README, methodology/results/audit/reproducibility/deployment/dashboard/viva records, the 33-slide PPTX/PDF and technical report addendum. Slide 28 geometry was retained and rechecked.
- Public deployment and remote CI remain unverified due unavailable authenticated deployment/remote workflow access. No deployment URL or remote workflow success is claimed.
- Limitation retained: before the cutoff fix, some development loaders transiently read reserved sales values before slicing; those values did not enter training, selection, or scoring.


Final bundle manifest check (15 November 2026): `dashboard_data_bundle_final_evaluation.zip`, 7,431,020 bytes, SHA-256 `1ba069f8c14470e4cc55ff18268011cc1d8d299b30a15fe0e91502781987cdb8`; 99 manifest-listed files verified. Raw inputs and row-level final-holdout forecasts are absent. The archive includes candidate/freeze metadata and compact result summaries.
