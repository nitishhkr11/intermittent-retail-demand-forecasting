# Development candidate selection and freeze

Selection used development-only folds before any final-period scoring. Fold means are descriptive; no significance claim is made.

## Rules

- Primary metric: MAE. Secondary metrics: RMSE, WAPE, MASE and bias; report fold SD and failures.
- Aggregate candidates require all 16 common quarterly folds and zero failed fits.
- Demand-class candidates require all 16 folds, at least five items per fold, and at least 95% successful item-fold cells in each fold.
- Compare equal-weight class/item-fold results only within the item scope. Do not combine with aggregate units.
- Calendar/event features are a richer information set and are identified in the manifest.

## Selected candidates

| Group | Candidate | MAE | RMSE | Fold coverage | Rationale |
|---|---|---:|---:|---:|---|
| Aggregate | LSTM(calendar) | 185.092 | 235.451 | 16/16 | Lowest MAE among complete aggregate candidates |
| Smooth | XGBoost(pooled,lags_calendar_events_poisson,poisson) | 2.858 | 3.566 | 16/16 | Lowest eligible class-specific mean MAE |
| Erratic | XGBoost(pooled,lags_calendar_events_poisson,poisson) | 5.086 | 6.298 | 16/16 | Lowest eligible class-specific mean MAE |
| Intermittent | XGBoost(pooled,lags_calendar) | 1.520 | 1.969 | 16/16 | Lowest eligible class-specific mean MAE |
| Lumpy | TSB(alpha=0.1,beta=0.1) | 1.925 | 2.512 | 16/16 | Lowest eligible class-specific mean MAE |
| NoDemand | SeasonalNaive | — | — | development-only fallback | Deterministic zero-demand fallback; not performance-ranked |

The development selection manifest is `artifacts/final/DEVELOPMENT_SELECTION.json`. It records exact settings, data/config/artifact hashes and selection rationale. The protected `FROZEN_CANDIDATES.json` is written separately by the final-holdout freeze command. Neither step evaluates the holdout.

## Inventory calibration

All-history empirical error quantiles remain the conservative default: recency weighting and a two-fold history were worse; class-normalized calibration was close but did not prove a robust gain. Every evaluated method remained below the 95% target overall. Model and fold-local class tables are in `artifacts/expanded_inventory_calibration_v2/`.

## Limits

The demand classes are fold-local and the number of items per class changes by origin. XGBoost has known calendar/event information that some baselines do not. The per-item XGBoost sample is not selected for whole-class deployment because it covers a fixed 183-item subset only. Inventory service is cycle service, not fill rate. The protected 28-day final holdout was evaluated once after freeze; the score and evaluation count are reported in `FINAL_HOLDOUT_RESULTS.md`. Earlier transient reads are disclosed in `HOLDOUT_DATA_BOUNDARY_AUDIT.md`.
