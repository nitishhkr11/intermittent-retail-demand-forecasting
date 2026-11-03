# Phase 3 test record

This file records what was actually run for the Phase 3 completion, with the exact counts printed by pytest.
Nothing here is an estimate. The runs were made in the development sandbox (Python 3.13.16, pytest 9.1.1, pandas 3.0.5,
numpy 2.5.3, torch 2.15.0 on the CPU, prophet 1.5.0, statsmodels 0.15.0, xgboost 3.4.1, streamlit 1.66.1,
plotly 7.1.0, Linux, 2 CPU cores). They were **not** run on a Windows machine: the sandbox has no access to yours.
On your machine, run the commands in section 4 and compare.

All runs below were made in throw-away copies of the final repository tree, so the delivered files were not rewritten by
the tests (some Phase 2 notebook tests re-execute a notebook and overwrite its saved artifacts; see section 3).

## 1. Totals

The suite collects **935 tests** (`python -m pytest --collect-only -q`). Every one was run in one of the three runs below.

| Run | Command (from the project root) | Result | Time |
|---|---|---|---|
| Fast suite | `python -m pytest -q -m "not slow"` | **915 passed, 1 skipped**, 19 deselected, 0 failed | 7 min 31 s |
| Slow suite | `python -m pytest -q -m slow --deselect tests/notebooks/test_lstm_gru_notebook.py` | **18 passed**, 0 failed (15 real-browser tests + the end-to-end executions of notebooks 05, 08 and 10) | 2 min 34 s |
| Neural notebook | `RIVERDEMAND_RUN_NEURAL_NOTEBOOK=1 python -m pytest -q tests/notebooks/test_lstm_gru_notebook.py` (checkpoints deleted first, so all 100 networks were trained again) | **17 passed**, 0 skipped, 0 failed | 15 min 0 s |

Counting each test once: 915 + 18 + 1 (the notebook-07 execution test; the other 16 tests of that file are inside the 915)
= 934 passed, 1 skipped, 0 failed. No test was deleted, edited to pass, or marked as expected failure. The one skipped
test is explained in section 3.

## 2. Tests per file (collected)

| Test file | Tests | Added in Phase 3 |
|---|---:|---|
| `tests/analysis/test_demand_classification.py` | 30 | no (earlier phase) |
| `tests/analysis/test_diagnostics.py` | 26 | no (earlier phase) |
| `tests/analysis/test_eda.py` | 18 | no (earlier phase) |
| `tests/app/test_browser_smoke.py` | 15 | yes |
| `tests/app/test_components.py` | 19 | yes |
| `tests/app/test_empty_states.py` | 5 | yes |
| `tests/app/test_pages.py` | 47 | yes |
| `tests/app/test_theme.py` | 59 | yes |
| `tests/data/test_aggregate_series.py` | 11 | yes |
| `tests/data/test_series_validator.py` | 14 | no (earlier phase) |
| `tests/data/test_splitter.py` | 7 | no (earlier phase) |
| `tests/data/test_validator.py` | 17 | no (earlier phase) |
| `tests/evaluation/test_backtesting.py` | 20 | no (earlier phase) |
| `tests/evaluation/test_comparison.py` | 8 | no (earlier phase) |
| `tests/evaluation/test_final_holdout.py` | 35 | yes |
| `tests/evaluation/test_forecast_artifacts.py` | 51 | yes |
| `tests/evaluation/test_holdout_artifact_guard.py` | 2 | yes |
| `tests/evaluation/test_intermittent_experiment.py` | 33 | yes |
| `tests/evaluation/test_item_level.py` | 13 | no (earlier phase) |
| `tests/evaluation/test_item_level_pooled.py` | 25 | no (earlier phase) |
| `tests/evaluation/test_metrics.py` | 10 | no (earlier phase) |
| `tests/evaluation/test_model_comparison.py` | 18 | yes |
| `tests/evaluation/test_residuals.py` | 24 | no (earlier phase) |
| `tests/inventory/test_reorder.py` | 71 | yes |
| `tests/models/test_arima.py` | 9 | no (earlier phase) |
| `tests/models/test_baseline.py` | 7 | no (earlier phase) |
| `tests/models/test_croston.py` | 20 | no (earlier phase) |
| `tests/models/test_holt_winters.py` | 12 | no (earlier phase) |
| `tests/models/test_lstm_gru.py` | 61 | yes |
| `tests/models/test_lstm_gru_checkpoints.py` | 6 | yes |
| `tests/models/test_prophet_model.py` | 43 | yes |
| `tests/models/test_sarima.py` | 8 | no (earlier phase) |
| `tests/models/test_tsb.py` | 23 | no (earlier phase) |
| `tests/models/test_xgboost_model.py` | 67 | no (earlier phase) |
| `tests/notebooks/test_classical_models_notebook.py` | 2 | no (earlier phase) |
| `tests/notebooks/test_comparison_notebook.py` | 17 | yes |
| `tests/notebooks/test_intermittent_notebook.py` | 8 | yes |
| `tests/notebooks/test_inventory_notebook.py` | 8 | yes |
| `tests/notebooks/test_lstm_gru_notebook.py` | 17 | yes |
| `tests/notebooks/test_notebook_hygiene.py` | 25 | yes |
| `tests/notebooks/test_prophet_notebook.py` | 8 | yes |
| `tests/notebooks/test_xgboost_notebook.py` | 3 | no (earlier phase) |
| `tests/utils/test_provenance.py` | 13 | yes |
| **Total** | **935** | 572 added in Phase 3, 363 from earlier phases |

(`test_notebook_hygiene.py` is new in Phase 3: one parametrised check set over notebooks 05, 07, 08, 09 and 10. The test files of earlier phases are unchanged.)

## 3. The one skipped test, and a Phase 2 side effect

`tests/notebooks/test_comparison_notebook.py::test_comparison_mae_equals_mae_recomputed_directly_from_the_forecast_files`
recomputes every fold MAE and RMSE of the unified comparison from the saved forecast files and demands equality to 1e-9.
It first checks the SHA-256 that notebook 09 recorded for each forecast file. If a file changed after notebook 09 ran,
the check cannot be made and the test **skips** with the file name and the command that refreshes the comparison.

In the full fast suite it skipped, for this reason: the existing Phase 2 test
`tests/notebooks/test_classical_models_notebook.py::test_notebook_executes_end_to_end_against_real_data` re-executes
notebook 04, which overwrites the saved SARIMA artifacts. In this sandbox the re-estimated SARIMA fold 4 differs from the
saved one (MAE 169.54 instead of 164.73; folds 1, 2, 3 and 5 are identical), because that SARIMA fit is sensitive to the
library versions. The saved Phase 2 files were restored from your Step 1 snapshot after that discovery and checked
byte for byte; the delivered numbers are the saved Phase 2 ones.

Run on its own, against the delivered files, the test is not skipped and passes: see section 5 (17 passed, 0 skipped).
Before the skip logic was added, the same test failed in a full run with the message "Mismatched elements: 1 / 55"; that
failure is what led to the finding above. The first version of the test was not weakened: the strict comparison is
still performed whenever the inputs are the ones notebook 09 used.

Other side effects of Phase 2 / Step 1 tests, observed by hashing the tree before and after: the notebook-06 test
rewrites `artifacts/metrics/xgboost_fold_diagnostics.csv` (only the fit-time column differs). These tests are
completed earlier work and were not changed.

## 4. Commands for your machine

```powershell
python -m pytest -q -m "not slow"                                   # fast suite
python -m pytest -q -m slow --deselect tests/notebooks/test_lstm_gru_notebook.py    # browser and notebook execution
$env:RIVERDEMAND_RUN_NEURAL_NOTEBOOK = "1"; python -m pytest -q tests/notebooks/test_lstm_gru_notebook.py   # about 15-20 min
```

The browser tests need Playwright's Chromium (`python -m pip install playwright`, then `python -m playwright install chromium`);
without it they skip with that command. Tests that need `torch`, `prophet`, `streamlit` or a saved artifact skip themselves with the
command that provides it, so a minimal environment gives skips, not failures.

Because the notebook-04 and notebook-06 execution tests rewrite saved Phase 2 / Step 1 artifacts, make a Git commit (or a
copy of `artifacts\`) before running the fast suite, as `MERGE_AND_RUN_WINDOWS.md` advises. If SARIMA is re-estimated
differently on your machine, the comparison test will skip and tell you to re-run notebook 09.

## 5. Checks beyond pytest

- **Reproducibility of the shipped artifacts.** The from-scratch notebook-07 run reproduced `lstm_gru_backtest_forecasts.csv`,
  `lstm_gru_backtest_metrics.csv` and `lstm_gru_seed_metrics.csv` bit for bit. Re-executing notebooks 05, 08 and 10
  reproduced the Prophet forecasts and metrics, the intermittent-demand summaries and all four inventory tables
  identically. (Same machine and library versions; other machines can differ in the last digits.)
- **Standalone comparison test on the delivered tree:** `python -m pytest -q tests/notebooks/test_comparison_notebook.py`
  → **17 passed, 0 skipped** on the delivered tree (all nine SHA-256 prefixes recorded by notebook 09 match the delivered forecast files,
  so the strict MAE/RMSE recomputation ran). `python -m src.evaluation.final_holdout status` on the delivered tree prints `candidates frozen : no`, `holdout evaluated : no`.
- **Phase 1 / Phase 2 preservation.** SHA-256 of the 159 files of the Step 1 snapshot compared with the delivered tree:
  137 are byte-identical; the 18 that differ are exactly the intended replacements (the placeholder files, `config.yaml`
  with additions only, `README.md`, and one docstring); the other 4 are the `app/pages/` placeholders on the removal list.
- **Final holdout.** Not frozen, not evaluated. `python -m src.evaluation.final_holdout status` reports
  `candidates frozen : no` and `holdout evaluated : no`; `artifacts/final/` holds only its README.
  A test scans the date columns of every CSV under `artifacts/forecasts`, `metrics`, `inventory` and `residuals` and finds none after 2016-04-24.
- **Browser check.** Each of the six pages was loaded in headless Chromium in light and dark mode with no script error and no
  sideways scroll, and the theme toggle was operated on a live page (the 15 browser tests).
- **Older versions (Python 3.11, pandas 2.0.3).** The fast suite was also run in a separate environment with Python 3.11.17,
  pandas 2.0.3, numpy 1.26.4, statsmodels 0.15.0, xgboost 2.1.4, pmdarima 2.1.1, streamlit 1.49.1 and plotly 5.24.1
  (torch, prophet and Playwright not installed), without the two long Phase 2 / Step 1 notebook-execution tests:
  **871 passed, 1 failed, 35 skipped** (13 need prophet, 16 need torch, 1 is the Playwright browser module), 4 deselected.
  The one failure is `tests/evaluation/test_item_level_pooled.py::test_each_items_forecasts_and_metrics_are_attributed_to_that_item`,
  a Step 1 test that calls `groupby.apply(..., include_groups=False)`, which needs pandas 2.2 or newer. It was left unchanged.
  The check also exposed two things that were fixed or documented: Phase 2's `src/analysis/diagnostics.py` needs
  statsmodels 0.15 (0.14.6 raises `TypeError`; floor now stated in `requirements-phase3.txt`), and one of my own tests used
  `include_groups` (rewritten to `groupby` on a Series; the file passes on 3.13 and 3.11). The LSTM/GRU and Prophet code and
  the real-browser tests were not run on 3.11.
- **Not done:** the PowerShell script and the Windows commands were not executed (no PowerShell in the sandbox), and nothing was
  run on Windows. The script's decision logic (new / same / replace / conflict) was reproduced in Python for the merge
  simulation below.

## 6. Verification of the delivered ZIP files

The runs in section 1 were made before two small edits: one line in `tests/evaluation/test_final_holdout.py` (a `groupby`
rewrite for pandas 2.0 compatibility) and comments in `requirements-phase3.txt`. Everything below was run on the final code.

- **Contents audit.** `riverdemand_completion_patch.zip` has 117 entries (99 added + 18 modified files, forward-slash paths relative to the
  project root, no wrapper folder, no directory entries, none of the four removal-list placeholders). `riverdemand_full_source.zip` has
  254 entries inside one folder named `intermittent-retail-demand-forecasting/`. Neither contains `.git`, `.venv`, `__pycache__`, `.pyc`,
  `.pytest_cache`, raw M5 CSV files (only `data/raw/.gitkeep` from your own snapshot), the resume checkpoints, `.env`, key files or
  secrets. The text files were scanned for key-like strings and for sandbox or Windows user paths: the only hit is the unchanged
  Phase 2 notebook 04, which already contained a sandbox path in a saved output (full ZIP only; see `IMPLEMENTATION_MANIFEST.md`).
  Every entry of both ZIPs has the same SHA-256 as the file in the working tree.
- **Clean extraction of the complete ZIP**, tests run from the extracted folder: fast suite **915 passed, 1 skipped, 19 deselected**
  (6 min 28 s); slow suite (`-m slow`, notebook 07 deselected) **18 passed** (2 min 30 s). The one skip is the one explained in section 3.
  The notebook-07 end-to-end test (17 passed from scratch, section 1) was not repeated on the extraction because no code it uses changed.
- **Merge simulation on a copy of your Step 1 snapshot**, using the new / same / replace / conflict rules and the replace list parsed
  out of `apply_riverdemand_patch.ps1`: dry run = 99 NEW, 18 REPLACE, 0 CONFLICT; after copying, a second run = 117 SAME (so the script is
  safe to run twice); the merged copy is byte-identical to the delivered tree except for the four placeholder pages, which the removal list
  covers. With a hand-edited generated artifact in the copy, that one file is reported as CONFLICT and skipped. Only the rules were
  simulated (in Python), not the PowerShell script itself.

