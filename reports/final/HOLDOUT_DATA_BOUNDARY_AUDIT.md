## Current record status — 15 November 2026

The period 2016-04-25 through 2016-05-22 has been evaluated exactly once after candidate freeze. See `FINAL_HOLDOUT_RESULTS.md` and `artifacts/final/FINAL_EVALUATION_RECORD.json`. The provenance limitation described below concerns earlier transient reads during development and does not change the saved evaluation count. Do not rerun or tune against the holdout.

# Final holdout data boundary audit

## Current evaluation status

The final period, 2016-04-25 through 2016-05-22, was evaluated exactly once after candidate freeze. The immutable record is `artifacts/final/FINAL_EVALUATION_RECORD.json` (record count 1, no override); results are in `reports/final/FINAL_HOLDOUT_RESULTS.md`. Do not rerun or tune against this period. This status supersedes the earlier pre-evaluation text in the historical audit notes below.

## Loader issue found during source audit — 14 November 2026

Older development loaders read sales columns across the full stored history and then removed the reserved rows in memory:

- `src.data.aggregate_series._read_aggregate_table` loaded the full aggregate `date,sales` table before `load_aggregate_development` split it. That loader also checked the whole table for missing or negative values.
- `src.evaluation.intermittent_experiment.load_all_item_development_sales` loaded `item_id,date,sales` for the full item panel before filtering dates to the development end.

Those code paths were used by earlier development experiments, including the completed seasonal aggregate and all-item runs. Their model inputs and scored forecasts were sliced to development dates, and no holdout metric was computed, but the sales values were transiently present in process memory. Therefore earlier metadata fields that say `holdout_accessed: false` mean “not used for model fitting/scoring” in practice; they do not prove that the bytes were never read. This distinction supersedes claims elsewhere that the holdout target values were never loaded.

## Correction and verification

Both development loaders now first read date metadata only, derive the cutoff, then request sales columns with a Parquet predicate `date <= 2016-04-24`. The event-calendar loader already reads calendar columns only and applies the same cutoff. Regression tests use a `1e12` reserved-period sentinel and assert the sales read received the cutoff predicate and never returned the sentinel.

Focused loader and feature tests passed: 115 tests covering XGBoost features, aggregate development loading, and all-item development loading. The complete neural run and the prior results remain development-only and were not tuned or scored using reserved values. Future development runs now use the predicate-protected loaders.

This correction cannot undo the earlier transient reads. It preserves the distinction between transient data access and model fitting/scoring during those development runs. The later one-time final evaluation happened only after candidate freeze and is recorded separately. The data-boundary history remains a limitation and is disclosed in final materials.
