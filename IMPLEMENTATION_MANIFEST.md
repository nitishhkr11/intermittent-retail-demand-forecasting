# Implementation manifest (Phase 3 completion)

This lists every file the Phase 3 completion adds, modifies or supersedes, relative to the Phase 3 Step 1 (XGBoost) snapshot that was delivered earlier. Paths are relative to the project root. Nothing here is a Git instruction: you commit and push yourself.

## What the two ZIP files contain

| ZIP | Kind | Contents |
|---|---|---|
| `riverdemand_completion_patch.zip` | **Reduced package, not a repository** | The 99 added and 18 modified files listed below, with project-root-relative paths and no wrapper folder. |
| `riverdemand_full_source.zip` | **Complete repository snapshot** (inside one folder named `intermittent-retail-demand-forecasting/`) | All 254 files of the updated project: the 137 files identical to the Step 1 snapshot plus the added and modified ones. |

### Files omitted from the complete snapshot (and from the patch)

| Omitted | Why |
|---|---|
| `.git/` | Your Git history stays in your own folder |
| `.venv/` | Your virtual environment |
| `data/raw/*` (the M5 CSV files) | Raw competition data is never delivered or committed |
| `artifacts/models/lstm_gru_checkpoints/` (20 small files) | Resume checkpoints of notebook 07; git-ignored, machine and library specific, safe to delete |
| `__pycache__/`, `.pytest_cache/`, `.ipynb_checkpoints/`, `*.pyc` | Caches |
| Secrets, `.env*`, `*.pem`, `*.key` | None existed; the build also scans every text file for key-like strings |

`data/interim/panel_ca1_foods3.parquet` and `data/processed/*.parquet` (small derived tables, unchanged from your Step 1 snapshot) are inside the complete snapshot but not in the patch (the patch does not touch them).

## Preservation of completed work

- **137 of the 159 files in the Step 1 snapshot are byte-for-byte identical** (SHA-256 compared): all Phase 1 and Phase 2 source code, notebooks 01 to 04 and 06, every earlier test, the Phase 2 and XGBoost artifacts (forecasts, metrics, residual diagnostics), the Phase 1, Phase 2 and Step 1 figures and reports, `.gitignore`, `LICENSE` and `requirements.txt`.
- The 18 files that differ are all listed under 'Modified files'. Every one is a Phase 3 placeholder, `config.yaml` (additions only), `README.md`, or the docstring of `src/models/__init__.py`.
- Notebook 06 (XGBoost) is unchanged. Its original sentence that Phase 3 is not complete is still in it because a test asserts that text.
- All previously completed code, notebooks, test results, metrics and figures are therefore preserved. The previous test files are unchanged and still pass (see `reports/phase3/PHASE3_TEST_RECORD.md`).
- Notebook 04 (completed Phase 2) still contains sandbox-style absolute paths in some saved outputs from the time it was executed. It was not touched, so those remain in the complete snapshot (not in the patch).
- While verifying this delivery, one command was run by mistake in the working copy that regenerated three Phase 1 files (`data/processed/aggregate_ca1_foods3.parquet`, `demand_classification.parquet`, `selected_items.parquet`). They were restored from the Step 1 snapshot and re-hashed: identical to the snapshot. Your own folder was never touched.
- The existing Phase 2 test that re-executes notebook 04 overwrites the saved SARIMA artifacts as a side effect, and in the development sandbox it re-estimated SARIMA fold 4 differently (MAE 169.54 instead of the saved 164.73). Those files (`sarima_backtest_forecasts.csv`, `sarima_backtest_metrics.csv`, `sarima_fold3_convergence_sensitivity.csv`, `residual_diagnostics_ljung_box.csv`, `model_comparison_backtest_metrics.csv`, two `reports/phase2` figures) and `xgboost_fold_diagnostics.csv` (fit-time column only) were restored from your snapshot and verified identical; the delivered files carry the saved Phase 2 values. Details: `reports/phase3/PHASE3_TEST_RECORD.md`, section 3.

## Removal list (separate; nothing is deleted automatically)

These four Phase 1 placeholder pages are superseded by `app/views/`. They are not in either ZIP. Move them out of your project (command in `MERGE_AND_RUN_WINDOWS.md`, section 4); leaving `app/pages/` in place makes Streamlit treat it as its own page list.

- `app/pages/1_Overview.py`
- `app/pages/2_Forecast.py`
- `app/pages/3_Model_Comparison.py`
- `app/pages/4_Inventory_Decision.py`

## Modified files (18; the patch replaces these)

| File | Belongs in | What changed |
|---|---|---|
| `README.md` | `README.md` | Rewritten (the old text described Phase 1, Stage 3) |
| `app/app.py` | `app/app.py` | Phase 1 placeholder replaced by the real entry point |
| `app/components/cards.py` | `app/components/cards.py` | Phase 1 placeholder replaced |
| `app/components/charts.py` | `app/components/charts.py` | Phase 1 placeholder replaced |
| `app/components/sidebar.py` | `app/components/sidebar.py` | Phase 1 placeholder replaced |
| `app/components/tables.py` | `app/components/tables.py` | Phase 1 placeholder replaced |
| `app/data_access.py` | `app/data_access.py` | Phase 1 placeholder replaced |
| `app/theme.py` | `app/theme.py` | Phase 1 placeholder replaced |
| `config.yaml` | `config.yaml` | 172 lines added, 0 removed (Phase 3 sections); every earlier setting is unchanged |
| `notebooks/05_intermittent_models.ipynb` | `notebooks/05_intermittent_models.ipynb` | Placeholder replaced by the executed notebook |
| `notebooks/07_lstm_gru.ipynb` | `notebooks/07_lstm_gru.ipynb` | Placeholder replaced by the executed notebook |
| `notebooks/08_prophet.ipynb` | `notebooks/08_prophet.ipynb` | Placeholder replaced by the executed notebook |
| `notebooks/09_model_comparison.ipynb` | `notebooks/09_model_comparison.ipynb` | Placeholder replaced by the executed notebook |
| `src/inventory/__init__.py` | `src/inventory/__init__.py` | Placeholder docstring replaced |
| `src/inventory/reorder.py` | `src/inventory/reorder.py` | Placeholder replaced by the implementation |
| `src/models/__init__.py` | `src/models/__init__.py` | Docstring only (stale 'not yet implemented' notes); no code |
| `src/models/lstm_gru.py` | `src/models/lstm_gru.py` | Placeholder replaced by the implementation |
| `src/models/prophet_model.py` | `src/models/prophet_model.py` | Placeholder replaced by the implementation |

## Added files (99)

### (project root)

| File | Purpose |
|---|---|
| `IMPLEMENTATION_MANIFEST.md` | Delivery documentation |
| `MERGE_AND_RUN_WINDOWS.md` | Delivery documentation |
| `PHASE3_COMPLETION_REPORT.md` | Delivery documentation |
| `pytest.ini` |  |
| `requirements-phase3.txt` | Optional heavy dependencies (torch, prophet, streamlit, plotly) |

### .streamlit

| File | Purpose |
|---|---|
| `.streamlit/config.toml` | Streamlit defaults (base theme, headless, no file watcher, no usage stats) |

### app

| File | Purpose |
|---|---|
| `app/components/__init__.py` | Dashboard components (cards, tables, charts, sidebar) |
| `app/views/1_Overview.py` | Dashboard page 1: Overview |
| `app/views/2_Forecast_Explorer.py` | Dashboard page 2: Forecast Explorer |
| `app/views/3_Model_Comparison.py` | Dashboard page 3: Model Comparison |
| `app/views/4_Inventory_Decision.py` | Dashboard page 4: Inventory Decisions |
| `app/views/5_Model_Reliability.py` | Dashboard page 5: Model Reliability |
| `app/views/6_About_Methodology.py` | Dashboard page 6: About & Methodology |

### artifacts/final

| File | Purpose |
|---|---|
| `artifacts/final/README.md` | Explains the (empty) final-holdout folder |

### artifacts/forecasts

| File | Purpose |
|---|---|
| `artifacts/forecasts/intermittent_item_level_backtest_long.csv` | Saved backtest forecasts (generated by the notebooks; git-ignored) |
| `artifacts/forecasts/lstm_gru_backtest_forecasts.csv` | Saved backtest forecasts (generated by the notebooks; git-ignored) |
| `artifacts/forecasts/prophet_backtest_forecasts.csv` | Saved backtest forecasts (generated by the notebooks; git-ignored) |

### artifacts/inventory

| File | Purpose |
|---|---|
| `artifacts/inventory/inventory_demo_reorder_points.csv` | Saved inventory results (generated; git-ignored) |
| `artifacts/inventory/inventory_run_metadata.json` | Saved inventory results (generated; git-ignored) |
| `artifacts/inventory/inventory_sensitivity.csv` | Saved inventory results (generated; git-ignored) |
| `artifacts/inventory/inventory_service_level_backtest.csv` | Saved inventory results (generated; git-ignored) |
| `artifacts/inventory/inventory_service_level_backtest_summary.csv` | Saved inventory results (generated; git-ignored) |

### artifacts/metrics

| File | Purpose |
|---|---|
| `artifacts/metrics/comparison_aggregate_fold_metrics.csv` | Saved metrics, diagnostics and run metadata (generated; git-ignored) |
| `artifacts/metrics/comparison_aggregate_summary.csv` | Saved metrics, diagnostics and run metadata (generated; git-ignored) |
| `artifacts/metrics/comparison_artifact_audit.csv` | Saved metrics, diagnostics and run metadata (generated; git-ignored) |
| `artifacts/metrics/comparison_item_level_category_summary.csv` | Saved metrics, diagnostics and run metadata (generated; git-ignored) |
| `artifacts/metrics/comparison_item_level_fold_metrics.csv` | Saved metrics, diagnostics and run metadata (generated; git-ignored) |
| `artifacts/metrics/comparison_item_level_model_summary.csv` | Saved metrics, diagnostics and run metadata (generated; git-ignored) |
| `artifacts/metrics/comparison_run_metadata.json` | Saved metrics, diagnostics and run metadata (generated; git-ignored) |
| `artifacts/metrics/intermittent_all_items_category_summary.csv` | Saved metrics, diagnostics and run metadata (generated; git-ignored) |
| `artifacts/metrics/intermittent_all_items_fold_class_summary.csv` | Saved metrics, diagnostics and run metadata (generated; git-ignored) |
| `artifacts/metrics/intermittent_item_level_category_macro_avg_summary.csv` | Saved metrics, diagnostics and run metadata (generated; git-ignored) |
| `artifacts/metrics/intermittent_item_level_fold_summary.csv` | Saved metrics, diagnostics and run metadata (generated; git-ignored) |
| `artifacts/metrics/intermittent_item_level_per_item_summary.csv` | Saved metrics, diagnostics and run metadata (generated; git-ignored) |
| `artifacts/metrics/intermittent_run_metadata.json` | Saved metrics, diagnostics and run metadata (generated; git-ignored) |
| `artifacts/metrics/intermittent_state_diagnostics.csv` | Saved metrics, diagnostics and run metadata (generated; git-ignored) |
| `artifacts/metrics/intermittent_zero_forecast_reference.csv` | Saved metrics, diagnostics and run metadata (generated; git-ignored) |
| `artifacts/metrics/lstm_gru_backtest_metrics.csv` | Saved metrics, diagnostics and run metadata (generated; git-ignored) |
| `artifacts/metrics/lstm_gru_fit_diagnostics.csv` | Saved metrics, diagnostics and run metadata (generated; git-ignored) |
| `artifacts/metrics/lstm_gru_run_metadata.json` | Saved metrics, diagnostics and run metadata (generated; git-ignored) |
| `artifacts/metrics/lstm_gru_seed_metrics.csv` | Saved metrics, diagnostics and run metadata (generated; git-ignored) |
| `artifacts/metrics/lstm_gru_training_history.csv` | Saved metrics, diagnostics and run metadata (generated; git-ignored) |
| `artifacts/metrics/prophet_backtest_metrics.csv` | Saved metrics, diagnostics and run metadata (generated; git-ignored) |
| `artifacts/metrics/prophet_fit_diagnostics.csv` | Saved metrics, diagnostics and run metadata (generated; git-ignored) |
| `artifacts/metrics/prophet_run_metadata.json` | Saved metrics, diagnostics and run metadata (generated; git-ignored) |

### notebooks

| File | Purpose |
|---|---|
| `notebooks/10_inventory_decision_support.ipynb` | Notebook 10 executed: inventory decision support |

### reports/phase3

| File | Purpose |
|---|---|
| `reports/phase3/PHASE3_TEST_RECORD.md` | Record of the final test runs |
| `reports/phase3/comparison_aggregate_fold_variation.png` | Figures written by the Phase 3 notebooks |
| `reports/phase3/comparison_aggregate_mae_rmse.png` | Figures written by the Phase 3 notebooks |
| `reports/phase3/comparison_item_level_by_category.png` | Figures written by the Phase 3 notebooks |
| `reports/phase3/comparison_item_level_rmse_by_category.png` | Figures written by the Phase 3 notebooks |
| `reports/phase3/intermittent_category_mae_comparison.png` | Figures written by the Phase 3 notebooks |
| `reports/phase3/intermittent_category_rmse_comparison.png` | Figures written by the Phase 3 notebooks |
| `reports/phase3/intermittent_demand_patterns.png` | Figures written by the Phase 3 notebooks |
| `reports/phase3/intermittent_example_forecasts.png` | Figures written by the Phase 3 notebooks |
| `reports/phase3/intermittent_zero_run_lengths.png` | Figures written by the Phase 3 notebooks |
| `reports/phase3/inventory_sensitivity.png` | Figures written by the Phase 3 notebooks |
| `reports/phase3/inventory_service_level_calibration.png` | Figures written by the Phase 3 notebooks |
| `reports/phase3/inventory_worked_example.png` | Figures written by the Phase 3 notebooks |
| `reports/phase3/lstm_gru_forecast_vs_actual.png` | Figures written by the Phase 3 notebooks |
| `reports/phase3/lstm_gru_seed_variability.png` | Figures written by the Phase 3 notebooks |
| `reports/phase3/lstm_gru_training_curves.png` | Figures written by the Phase 3 notebooks |
| `reports/phase3/lstm_gru_vs_classical_mae_rmse.png` | Figures written by the Phase 3 notebooks |
| `reports/phase3/prophet_components.png` | Figures written by the Phase 3 notebooks |
| `reports/phase3/prophet_forecast_vs_actual.png` | Figures written by the Phase 3 notebooks |
| `reports/phase3/prophet_interval_coverage.png` | Figures written by the Phase 3 notebooks |
| `reports/phase3/prophet_vs_classical_mae_rmse.png` | Figures written by the Phase 3 notebooks |

### src

| File | Purpose |
|---|---|
| `src/data/aggregate_series.py` | Loader for the aggregate development series (holdout excluded) |
| `src/evaluation/final_holdout.py` | Controlled final-holdout workflow: status, freeze, run once |
| `src/evaluation/forecast_artifacts.py` | Forecast-table schema, audit checks, metrics from saved forecasts |
| `src/evaluation/intermittent_experiment.py` | Intermittent-demand experiment: fold-local classes, 20 items and all 823 items |
| `src/evaluation/model_comparison.py` | Unified comparison from saved artifacts (aggregate and item level kept apart) |
| `src/utils/provenance.py` | Config hashing, file hashing and run-metadata helpers |

### tests

| File | Purpose |
|---|---|
| `tests/app/__init__.py` | Dashboard tests (theme, components, pages, empty states, browser smoke) |
| `tests/app/conftest.py` | Dashboard tests (theme, components, pages, empty states, browser smoke) |
| `tests/app/helpers.py` | Dashboard tests (theme, components, pages, empty states, browser smoke) |
| `tests/app/test_browser_smoke.py` | Dashboard tests (theme, components, pages, empty states, browser smoke) |
| `tests/app/test_components.py` | Dashboard tests (theme, components, pages, empty states, browser smoke) |
| `tests/app/test_empty_states.py` | Dashboard tests (theme, components, pages, empty states, browser smoke) |
| `tests/app/test_pages.py` | Dashboard tests (theme, components, pages, empty states, browser smoke) |
| `tests/app/test_theme.py` | Dashboard tests (theme, components, pages, empty states, browser smoke) |
| `tests/data/test_aggregate_series.py` | Aggregate-series loader tests |
| `tests/evaluation/test_final_holdout.py` | Final-holdout workflow tests |
| `tests/evaluation/test_forecast_artifacts.py` | Tests of the matching evaluation module |
| `tests/evaluation/test_holdout_artifact_guard.py` | Guard that no result file can contain holdout dates |
| `tests/evaluation/test_intermittent_experiment.py` | Tests of the matching evaluation module |
| `tests/evaluation/test_model_comparison.py` | Tests of the matching evaluation module |
| `tests/inventory/test_reorder.py` | Inventory layer tests |
| `tests/models/test_lstm_gru.py` | LSTM / GRU model and checkpoint tests |
| `tests/models/test_lstm_gru_checkpoints.py` | LSTM / GRU model and checkpoint tests |
| `tests/models/test_prophet_model.py` | Prophet model tests |
| `tests/notebooks/test_comparison_notebook.py` | Notebook structure, hygiene and artifact-audit tests (all new; the Phase 2 notebook tests are unchanged) |
| `tests/notebooks/test_intermittent_notebook.py` | Notebook structure, hygiene and artifact-audit tests (all new; the Phase 2 notebook tests are unchanged) |
| `tests/notebooks/test_inventory_notebook.py` | Notebook structure, hygiene and artifact-audit tests (all new; the Phase 2 notebook tests are unchanged) |
| `tests/notebooks/test_lstm_gru_notebook.py` | Notebook structure, hygiene and artifact-audit tests (all new; the Phase 2 notebook tests are unchanged) |
| `tests/notebooks/test_notebook_hygiene.py` | Notebook structure, hygiene and artifact-audit tests (all new; the Phase 2 notebook tests are unchanged) |
| `tests/notebooks/test_prophet_notebook.py` | Notebook structure, hygiene and artifact-audit tests (all new; the Phase 2 notebook tests are unchanged) |
| `tests/utils/__init__.py` | Provenance helper tests |
| `tests/utils/test_provenance.py` | Provenance helper tests |

## Where each file belongs

Every path in both ZIPs is the path inside your project folder (`app/` next to `src/`, `notebooks/`, `tests/`, `reports/`, `artifacts/`). The patch has no wrapper directory. The complete snapshot has one wrapper folder, `intermittent-retail-demand-forecasting/`, which is the project root. `apply_riverdemand_patch.ps1` (a separate download, not inside either ZIP) copies the files for you.
