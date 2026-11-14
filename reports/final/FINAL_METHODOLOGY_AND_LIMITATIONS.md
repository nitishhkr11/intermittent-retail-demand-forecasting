
# Current methodology and limitation update

The final evidence and results are summarized in [`reports/final/FINAL_RESEARCH_RESULTS.md`](reports/final/FINAL_RESEARCH_RESULTS.md), and the one-time final holdout score is in [`reports/final/FINAL_HOLDOUT_RESULTS.md`](reports/final/FINAL_HOLDOUT_RESULTS.md). The protected workflow has been consumed; do not rerun the holdout.


## Final status addendum — 15 November 2026

The protected 2016-04-25–2016-05-22 period was scored exactly once after the candidate freeze. The aggregate headline was LSTM(calendar), MAE 128.0869 and RMSE 183.1430. The class-selected item strategy scored macro MAE 1.683711 and RMSE 2.188407 across 823 items. The two metric scales are not comparable. Item-level MAPE excludes 8,432 zero-actual rows. Detailed results and hashes: `reports/final/FINAL_HOLDOUT_RESULTS.md`.

The complete development evidence and local tests are recorded in `reports/final/FINAL_RESEARCH_RESULTS.md` and `FINAL_PROJECT_AUDIT.md`. Inventory cycle-service calibration remains below 95%; no fill-rate or cost-savings claim is made. A provenance limitation remains: historical development loaders transiently read reserved sales values before slicing, but those values were not used for training, selection, or scoring. Current code predicates the cutoff in the source read. Public hosting and remote CI remain unverified.
