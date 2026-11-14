# RiverDemand experiment reproducibility

Run these commands from the repository root in the configured project environment. The expanded forecasts use only the development prefix; `src.evaluation.final_holdout` remains separately gated.

```powershell
python -m src.pipeline validate
python -m src.evaluation.expanded_seasonal prepare
python -m src.evaluation.expanded_seasonal run
python -m src.evaluation.expanded_intermittent prepare
python -m src.evaluation.expanded_intermittent run
python -m src.evaluation.expanded_prophet run
python -m src.evaluation.expanded_neural run
python -m src.evaluation.expanded_xgboost prepare
python -m src.evaluation.expanded_xgboost run
python -m src.evaluation.audit_expanded_xgboost
python -m src.evaluation.summarize_expanded_xgboost
python -m src.evaluation.sba_backtest
python -m src.evaluation.expanded_inventory --window-days 7 --service-level 0.95
python -m src.evaluation.expanded_inventory_calibration
python -m src.evaluation.summarize_expanded_inventory_calibration
python -m src.pipeline development --resume
python -m pytest tests/app -q
python -m pytest -q -m "not slow"
```

`--resume` only skips a stage when its declared output exists; it does not overwrite it. The expanded experiment runners refuse to overwrite their final output artifacts and verify manifest/config/data/code hashes before running. Checkpoint chunks in `artifacts/checkpoints/expanded_intermittent/` are retained to support recovery.

## Input and manifest hashes

Raw input SHA-256 values verified at startup:

| File | SHA-256 |
|---|---|
| `data/raw/calendar.csv` | `d12b5914ef03e66649adf5dd9e996e6602251c22b7a6af8f1f7e3aa12f8860f5` |
| `data/raw/sales_train_evaluation.csv` | `4b4a47c44c38380d2a9168216fea8c9ff2f31b1ddb772f8a0995952a038b8aa0` |
| `data/raw/sell_prices.csv` | `9da3ad1f8b8ccacdbdc70612191dd375ec24a4ac6625c24b75b3bc60b0bed2ef` |

| Manifest/artifact | SHA-256 |
|---|---|
| `artifacts/expanded_seasonal/fold_manifest.json` | `24575c7c1c1131a5c7fd630c735965f063b52eafc82a53dc103c3de0b0719b11` |
| `artifacts/expanded_intermittent/fold_manifest.json` | `d89952d80a88897d711e8cb190901ef5957eb8518797c55addc7444d2b2b9c67` |
| `artifacts/forecasts/sba_item_level_backtest_forecasts.csv` | `b9b2944b35100a3d496e14396910251dc721937389a3892622ac081522e63f23` |
| `artifacts/expanded_inventory/service_level_summary.csv` | `c71cfd4fc0da5710e176a54b2d717780717a7b2131bf36f86fb3915524d8a42a` |
| `artifacts/expanded_inventory/service_level_cells.csv` | `3e9e8cd87c5e59b28dd529d4492887bac4934af81760d4fafff66484af6aaac9` |

The initial dashboard package was rebuilt after later study outputs were added; the current hash and manifest validation are below.

## Additional seasonal model-family artifacts

- `python -m src.evaluation.expanded_prophet run` evaluates three configured Prophet variants over the shared 16 quarterly folds. Fold checkpoints are in `artifacts/checkpoints/expanded_prophet/`; primary outputs are in `artifacts/expanded_prophet_block_conformal/`.
- `python -m src.evaluation.expanded_neural run` evaluates four neural variants over the same 16 folds, five seeds per cell, and resumes from matching checkpoints. The verified result contains 64/64 successful variant-fold cells, 320 seed fits and 1,792 daily forecasts; the independent metric audit is in `artifacts/expanded_neural/completion_verification.json`.
- `python -m src.evaluation.expanded_xgboost prepare` predeclares the shared 16 folds and first-origin class-stratified 183-product per-item sample. `run` fits five pooled variants across all 823 products and one per-item variant across that fixed sample, checkpointing each pooled model/fold and each item/fold. The independently recalculated row-level audit is `artifacts/expanded_xgboost/independent_audit.json`.
- `python -m src.evaluation.expanded_inventory_calibration` compares all-history empirical, recent-two-fold, recency-weighted and class-normalized calibration over identical later 7-day evaluation cells. Each order-origin uses only earlier completed forecast folds. Model/class summaries use cell-cluster bootstrap intervals.
- Prophet block-conformal calibration reserves folds 1–8 for calibration and folds 9–16 for validation; its reported coverage applies to a later development period, not the final holdout.

The current safe dashboard archive is `reports/final/dashboard_data_bundle_development_only.zip`: 4,161,491 bytes, SHA-256 `7a90dd9b22ef73e33482c4d5fae842f77d5a0135040fa509c7feac30110b6303`. It contains 86 manifest-listed files, including XGBoost and alternative inventory summaries; all embedded hashes were verified.

The deployment-only sales extracts are described by `artifacts/dashboard_data/development_data_manifest.json`: start 2011-01-29, end 2016-04-24, 1,913 development days, 1,574,399 all-item rows and 38,260 selected-item rows. The package build used a development date predicate and excludes the holdout. This package-specific fact does not erase earlier transient reads by older development loaders; see `HOLDOUT_DATA_BOUNDARY_AUDIT.md` for the full disclosure.


## Final test execution record

On 15 November 2026, `python -m pytest tests/app -q` passed 214 browser/UI tests and `python -m pytest -q -m "not slow"` passed 958 tests, deselecting 83 slow tests, with zero failures and one Windows ZMQ Proactor warning. Browser screenshots cover six pages, light/dark modes, and 1920/1440/1366/768/390 px. No GitHub Actions run or public deployment is claimed.


Final bundle manifest check (15 November 2026): `dashboard_data_bundle_final_evaluation.zip`, 7,691,254 bytes, SHA-256 `0f3118283fd2f81a33b6a695210b1d240dd5fdcad15d8fa5151030a3d84a17ae`; 102 manifest-listed files verified. Raw inputs and row-level final-holdout forecasts are absent. The archive includes candidate/freeze metadata and compact result summaries.
