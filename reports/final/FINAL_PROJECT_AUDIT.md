# RiverDemand Finalization Audit

**Audit date:** 2026-11-13  
**Repository branch:** `finalization/riverdemand-research-completion`  
**Base commit:** `9b141a8` — Phase 3 forecasting and dashboard implementation  
**Audit scope:** source, local M5 data, saved development artifacts, tests, validation design, inventory method, dashboard, and deployment evidence.

## Executive status

RiverDemand has a substantial Phase 1–3 implementation and saved development-fold results. The local M5 files are available and the raw-to-panel pipeline was rerun for this audit. The original holdout remains unscored and no candidates are frozen. This audit does not certify the project as final-ready: the brief's expanded seasonal experiments, corrected fold-local classical selection, inventory recalibration, and downstream documentation changes remain outstanding.

## Repository and data evidence

- The branch at `9b141a8` contains the Phase 3 implementation and matches the requested Phase 3 checkpoint. A new finalization branch was created before project edits.
- Two pre-existing user edits to notebook kernel metadata in `notebooks/01_data_understanding.ipynb` and `notebooks/02_eda.ipynb` were preserved. No commit was created.
- Raw M5 CSVs are present locally under `data/raw/` and are Git-ignored. They were not added to Git.
- `src.data.preprocessor.build_panel()` was rerun in memory. It loaded the three raw CSVs, scoped CA_1 / FOODS_3, reshaped and joined the data, and passed all six panel checks.
- The rebuilt panel contained 1,597,443 rows, 823 items, and 1,941 dates from 2011-01-29 through 2016-05-22. There were no duplicate item/date keys, missing dates, discontinuities, negative sales, or join row-count changes.
- 317,261 item-days (19.86%) had no selling-price record; every one also had zero recorded sales. The dataset does not establish whether these observations represent genuine zero demand, product unavailability, or stockout-censored demand.
- The aggregate series passed all five daily-series checks and matched the item-level sum on each date. Development is 2011-01-29 through 2016-04-24 (1,913 days); the reserved 28-day holdout is 2016-04-25 through 2016-05-22.

## Component evidence matrix

| Component | Source code | Experiments run | Saved results | Tests | Dashboard | Reproducibility |
|---|---|---|---|---|---|---|
| Data loading, joins, validation, aggregation | Yes | Raw-to-panel audit rerun; Phase 1 outputs exist | Panel, aggregate, class tables, figures | Data and aggregate tests exist; full suite pending at audit time | Overview and Reliability | Local raw files available; not tracked in Git |
| ADI / CV² classification | Yes | Full-history and development labels exist; fold-local class artifact exists | Class counts and label-difference artifacts | Tests exist | Overview and Methodology | Window provenance is documented; full-history labels are descriptive only |
| Seasonal Naive, ARIMA, SARIMA, Holt-Winters | Yes | Five 28-day folds each in saved Phase 2 artifacts | Fold forecasts, metrics, residual diagnostics, convergence sensitivity | Tests exist | Comparison and Reliability | Reproducible settings; ARIMA/SARIMA orders and Holt-Winters trend were selected using data overlapping later fold tests |
| Croston, TSB, Seasonal Naive | Yes | Selected 20 items and all 823 items, five folds | Item, class, fold, zero-forecast artifacts | Tests exist | Comparison and Reliability | Fixed alpha values; no SBA implementation or tuning evidence found in saved results |
| XGBoost | Yes | Pooled and per-item variants on 20 sampled items, five folds | Per-item, class, fold, feature-importance artifacts | Tests exist | Comparison and Reliability | Seed/config recorded; not full-panel coverage |
| LSTM / GRU | Yes | Four variants × five folds × five seeds (100 networks); 0 failed fits in saved metadata | Forecasts, seed scores, training histories, diagnostics | Tests exist | Comparison and Reliability | Checkpoints and environment metadata exist; calendar variants have additional future-known inputs |
| Prophet | Yes | Three variants × five folds (15 fits); 0 failed fits in saved metadata | Forecasts, metrics, fit diagnostics, interval coverage | Tests exist | Comparison and Reliability | Fixed configuration and run metadata exist; intervals are not yet calibrated |
| Model comparison | Yes | 11 aggregate and 7 item-level models on five common folds | Metrics, artifact audit, summary tables | Tests exist | Comparison | Aggregate and item scales separated; folds are concentrated Dec–Apr and overlap in training history |
| Inventory decision support | Yes | Backtests on sampled items and saved folds | Reorder demos, sensitivity and service-level artifacts | Tests exist | Inventory Decisions | Inputs are illustrative; observed service at a 95% target was 87.7–89.8% on later folds |
| Streamlit dashboard | Yes; six read-only pages | Prior page screenshots and browser tests are recorded | Saved screenshots and data artifacts | App tests exist; current full suite pending | This is the dashboard | Reads saved artifacts; no public deployment URL was verified |
| Final holdout workflow | Yes; guarded CLI | Not run | No frozen candidates or holdout outputs | Guard tests exist | Status badge only | `status` confirmed candidates frozen: no; holdout evaluated: no |

## Experiment and methodology findings

### Existing aggregate results

Saved five-fold aggregate mean MAE is 193.35 for Holt-Winters and 181.62 for LSTM with calendar inputs. The LSTM result is 11.73 units/day lower than Holt-Winters, while the folds are few and non-independent. The calendar-enabled model also receives future calendar/event inputs that the Phase 2 classical models do not receive. These results are descriptive development evidence, not proof of general superiority.

### Existing item-level results

Saved item results cover 20 products (five per full-history demand class) for Croston, TSB, Seasonal Naive, and XGBoost variants. The saved comparison table reports per-item XGBoost as the lowest Erratic-class MAE at 3.373 and TSB at 3.388. TSB is lowest among the compared methods for Intermittent and Lumpy in that table; pooled XGBoost is lowest for Smooth. Croston and TSB baselines also have an all-823-item experiment. These results do not support a universal product-level claim.

### Selection and validation limitations

- Existing development validation uses five expanding folds with 28-day horizons concentrated between December 2015 and April 2016.
- ARIMA/SARIMA orders and the Holt-Winters trend were selected using the full development series, which overlaps the later fold test windows. This limits the interpretation of those historical scores.
- Neural preprocessing, XGBoost features, and demand classes have fold-local handling in Phase 3 artifacts, but this does not correct the Phase 2 model-selection exception.
- No expanded multi-season validation, paired uncertainty analysis, or calibrated Prophet intervals are present in the saved evidence audited here.

### Inventory method

The code computes expected demand over a protection interval, adds an error-quantile safety stock, and reports a reorder point. Saved backtests show a shortfall from the configured 95% service target. The existing record does not establish a calibrated operational policy, and no holding, shortage, or ordering cost data are available for cost-based claims.

## Tests, application, and deployment

- Automated tests exist across data, analysis, models, evaluation, inventory, notebooks, and dashboard code. A fresh Windows run of `python -m pytest -m "not slow"` completed with 915 passed, 2 skipped, 4 deselected, 0 failed, and one ZMQ Proactor event-loop warning. This was a local run, not a GitHub Actions result.
- The dashboard source has six pages and is designed to read saved CSV/JSON artifacts without training or forecasting beyond the development period.
- No live public deployment URL or CI success was verified during this audit.
- No compact deployment data bundle or dashboard-specific pinned dependency file was verified here.

## Required corrections and status

| Correction from the finalization brief | Status |
|---|---|
| Recheck raw data, row counts, price joins, dates, non-negative sales, and aggregate consistency | **VERIFIED COMPLETE** for this local audit run |
| Preserve the Phase 2 model-selection caveat | **VERIFIED COMPLETE** in the source evidence and updated presentation |
| Replace historical classical scores with fold-local parameter selection | **NOT COMPLETE** |
| Add seasonal forecast origins across multiple years | **NOT COMPLETE** |
| Expand intermittent methods with SBA and larger class-balanced item coverage | **NOT COMPLETE** |
| Run full-panel pooled XGBoost and requested ablations | **NOT COMPLETE** |
| Reassess neural/calendar and Prophet interval behavior across seasons | **NOT COMPLETE** |
| Calibrate and re-evaluate inventory service levels on untouched development windows | **NOT COMPLETE** |
| Freeze candidates and evaluate the final holdout once | **NOT STARTED**; prerequisites and candidate freeze remain outstanding |
| Verify deployment bundle, public URL, CI, and six-page user experience | **NOT VERIFIED** |
| Update all final documentation and generate the technical report package | **PARTIALLY COMPLETE**; this audit and the presentation were updated |

## Final-readiness determination

**PARTIALLY COMPLETE.** The data path and baseline saved evidence are verifiable, and the presentation's known wording and class-winner issues have been corrected. The expanded experiments, inventory calibration, current test result, and final holdout prerequisites still need completion before the project can be labeled academically final-ready.

## Supplemental execution results — 14 November 2026

The earlier audit above is retained as a record of its scope and date. Since that audit, this execution added and ran 16 disjoint seasonal windows (2012–2015), fold-local classical selection, a complete 823-item Seasonal Naive/Croston/SBA/TSB experiment, a separate canonical five-fold SBA artifact, and an earlier-fold-only 7-day inventory calibration. Exact metrics and file hashes are documented in `FINAL_RESEARCH_RESULTS.md` and `EXPERIMENT_REPRODUCIBILITY.md`.

The inventory results still fall short of the 95% target (91.17%–93.86%). The requested XGBoost ablations, seasonal LSTM/GRU comparisons, Prophet interval coverage study, and class-specific selection protocol remain incomplete. The final holdout therefore remains intentionally unevaluated. No public hosting deployment or GitHub Actions run was verified. A raw-data-free 4.32 MB dashboard data archive and dashboard-only pinned requirements have been prepared; its 71 manifest entries passed SHA-256 verification.

The latest local run of `pytest tests/app -q` passed 135 tests and skipped 1. The required non-slow suite is in progress; this report will not call that run passed until its final status and any failure are recorded.

## Updated evidence — 14 November 2026, final pass

The historical audit above is retained unchanged as a record of its earlier scope. Subsequent development-only evidence includes the 16-fold classical and all-item experiments, SBA, inventory calibration, a complete 16-fold/three-variant Prophet interval study, and a partial 11-common-fold LSTM/GRU pilot. Exact current metrics and artifacts are in `FINAL_RESEARCH_RESULTS.md`.

The complete seasonal Prophet study reports nominal 80% interval coverage from 65.63% to 67.63%, below target. The neural plan completed 47/64 variant-fold cells before a single fit reached 5,133.6 seconds; its common 11-fold summary is partial and must not guide selection. Full-panel XGBoost ablations and class-specific candidate selection are still absent. Therefore there is no frozen candidate manifest and no final-holdout run record; the holdout is not evaluated.

Dashboard QA passed on the final worktree: 199 app tests passed; 943 non-slow tests passed, 68 slow tests were deselected, and one ZMQ Proactor warning was reported. Chromium verified pure-white light-mode surfaces and a 1,168 px content container (max 1,280 px) at 1440 px. The refreshed 48-screenshot matrix covers six pages, two themes and four widths.

The prior dashboard archive included full-history sales parquet files and has been superseded; see `reports/final/_superseded/README.md`. The current deployable bundle is `reports/final/dashboard_data_bundle_development_only.zip`. It contains only the development-period sales extracts through 2016-04-24; its 75 manifest-listed file hashes were verified. No public deployment or GitHub Actions run has been verified. The technical report and final presentation still require content synchronization with this latest evidence.

## Finalization-prerequisite gate — 14 November 2026

The preceding audit sections are dated snapshots. At the latest check, the 16-fold seasonal classical, all-item intermittent, inventory, Prophet, partial neural and dashboard work described in `FINAL_RESEARCH_RESULTS.md` exists. Full-panel XGBoost feature ablations, the complete 16-fold LSTM/GRU comparison, selection evidence for each demand class, and alternative inventory calibration/class sensitivity are still incomplete. Accordingly, `src/evaluation/final_holdout.py` now blocks `freeze` and its read-only `status` output lists the same blockers without presenting holdout commands. The verified state remains: no frozen candidate file and no holdout evaluation record.

The technical report and 33-slide final presentation in `presentation/` have not yet been synchronized to the latest expanded research outputs. The current documentation does not certify them as current deliverables. Dashboard and non-slow suite results are 199 passed, and 943 passed / 68 deselected with one ZMQ Proactor warning, respectively. No public deployment URL or GitHub Actions run has been verified.

The final post-gate non-slow suite completed with 944 passed, 68 deselected and one ZMQ Proactor warning (587.91 seconds). The current 33-slide deck still contains an outdated local test count and does not yet include the expanded study tables; the DOCX/PDF report also predates these addenda. These artifacts remain open and are not certified synchronized.


## Final completion snapshot — 15 November 2026

The dated audit snapshots above record the state at the time each was written. This final snapshot supersedes their pending-work statements. Expanded seasonal, all-item, XGBoost, neural, Prophet, and inventory calibration studies are complete; development selection and the protected candidate freeze are recorded separately. The 28-day holdout was evaluated once, with record count 1 and matching hashes for forecasts, aggregate/item metrics, and class metrics. See `FINAL_HOLDOUT_RESULTS.md`. The earlier transient data-read disclosure remains a provenance limitation: those reserved values did not enter training, selection, or scoring, and current loaders predicate the development cutoff at read time.

Local verification: `pytest tests/app -q` passed 214 tests; `pytest -q -m "not slow"` passed 958 tests, with 83 slow tests deselected and one Windows ZMQ Proactor warning. The 60-screenshot browser matrix covers six pages, both themes, and 1920/1440/1366/768/390 px; computed light-mode app surfaces were white, and the 1440 px main container was 1,168 px with a 1,280 px max width. The current 33-slide presentation and matching PDF were synchronized and checked; Slide 28 retains nine 0.60-inch rows at y=1.40 inches (5.40-inch total, bottom 6.80 inches).

Public provider deployment and a remote GitHub Actions run were not verified. The deployable data package is `dashboard_data_bundle_final_evaluation.zip`; it contains compact evaluation summaries and the record, excludes raw M5 inputs and row-level final forecasts. This is local completion evidence, not a claim of production operation.


Final bundle manifest check (15 November 2026): `dashboard_data_bundle_final_evaluation.zip`, 7,431,020 bytes, SHA-256 `1ba069f8c14470e4cc55ff18268011cc1d8d299b30a15fe0e91502781987cdb8`; 99 manifest-listed files verified. Raw inputs and row-level final-holdout forecasts are absent. The archive includes candidate/freeze metadata and compact result summaries.
