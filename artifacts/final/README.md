# artifacts/final

This folder holds the **final-holdout** records, and nothing else. The reserved window is the last 28 days
(2016-04-25 to 2016-05-22). No notebook, dashboard page or development script reads those days.

| File | Written by | Meaning |
|---|---|---|
| `frozen_candidates.json` | `python -m src.evaluation.final_holdout freeze` | The candidate models and the headline model per scope, chosen from the DEVELOPMENT folds only, plus a hash of the settings. |
| `final_holdout_forecasts.csv` | `... run --confirm EVALUATE-FINAL-HOLDOUT-ONCE` | The 28-day forecasts of every frozen candidate next to the holdout actuals. |
| `final_holdout_results.csv` | same | MAE, RMSE, zero-protected MAPE, WAPE, bias and MASE per candidate. Aggregate and item-level rows are different scales and are never ranked together. |
| `FINAL_EVALUATION_RECORD.json` | same | When the evaluation ran, with which settings and library versions, and how many times. |

**Status of this repository: the holdout has NOT been evaluated.** No file other than this README is present. Check with
`python -m src.evaluation.final_holdout status`.

Run the evaluation once, after the development results are final. Changing models afterwards because of what the
holdout showed turns it into a validation set and the number into an optimistic one; if a repeat is unavoidable,
`--override-existing` archives the old files and records the repeat.
