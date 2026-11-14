# Final holdout data boundary audit

## Current evaluation status

The final period, 2016-04-25 through 2016-05-22, has not been evaluated: no frozen candidate file, final forecast table, result table, or evaluation record exists. No development model or metric was fit/scored on those 28 target values. The protected evaluation workflow remains closed while research prerequisites are incomplete.

## Loader issue found during source audit — 14 November 2026

Older development loaders read sales columns across the full stored history and then removed the reserved rows in memory:

- `src.data.aggregate_series._read_aggregate_table` loaded the full aggregate `date,sales` table before `load_aggregate_development` split it. That loader also checked the whole table for missing or negative values.
- `src.evaluation.intermittent_experiment.load_all_item_development_sales` loaded `item_id,date,sales` for the full item panel before filtering dates to the development end.

Those code paths were used by earlier development experiments, including the completed seasonal aggregate and all-item runs. Their model inputs and scored forecasts were sliced to development dates, and no holdout metric was computed, but the sales values were transiently present in process memory. Therefore earlier metadata fields that say `holdout_accessed: false` mean “not used for model fitting/scoring” in practice; they do not prove that the bytes were never read. This distinction supersedes claims elsewhere that the holdout target values were never loaded.

## Correction and verification

Both development loaders now first read date metadata only, derive the cutoff, then request sales columns with a Parquet predicate `date <= 2016-04-24`. The event-calendar loader already reads calendar columns only and applies the same cutoff. Regression tests use a `1e12` reserved-period sentinel and assert the sales read received the cutoff predicate and never returned the sentinel.

Focused loader and feature tests passed: 115 tests covering XGBoost features, aggregate development loading, and all-item development loading. The complete neural run and the prior results remain development-only and were not tuned or scored using reserved values. Future development runs now use the predicate-protected loaders.

This correction cannot undo the earlier transient reads. It does preserve the distinction between a data-access breach and a holdout evaluation. Whether to treat the period as suitable for a future first scoring event must be disclosed in the final report; no such event has occurred here. The one-time final evaluation remains blocked by the incomplete XGBoost, inventory-calibration and class-selection work.
