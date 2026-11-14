# RiverDemand dashboard guide

Start the app from the repository root with `streamlit run app/app.py`. The app reads saved development data and forecasts; it does not generate live future sales. The final holdout was evaluated once on 2016-04-25 through 2016-05-22. Its summary is shown in Model Comparison and the exact tables are in `reports/final/FINAL_HOLDOUT_RESULTS.md`; the dashboard package does not contain row-level final forecasts.

## Pages

1. **Overview** — catalog and development-window summary, with the holdout status.
2. **Forecast Explorer** — saved historical backtest forecasts for supported models and products. These are retrospective forecasts, not live predictions.
3. **Model Comparison** — aggregate and item-level comparisons kept on separate scales. The expanded-study tab reports the 16-fold classical and 823-item baseline experiments, fold-local classes, fit failures and inventory calibration.
4. **Inventory Decisions** — an illustrative single-item reorder point using a saved forecast and historical forecast errors. Lead time, review period and target service are user inputs; the expanded earlier-fold cycle-service calibration remains below its 95% target (92.47% for the all-history method; 92.59% for class-normalized). This is cycle service, not fill rate.
5. **Model Reliability** — residual diagnostics, including remaining autocorrelation.
6. **About & Methodology** — data scope, evaluation protocol, limitations and holdout state.

Use the sidebar theme toggle for light/dark mode. Tables provide downloads when shown. Missing artifacts are described in place; the dashboard does not substitute synthetic model results.

## Runtime and deployment data

The pinned dashboard dependencies are in `app/requirements.txt`. Build the raw-data-free saved-data bundle with `python -m src.deployment.dashboard_package`. Extract the verified archive at the repository root before starting the app in a clean deployment environment. The bundle excludes raw M5 sales inputs and includes processed development data and dashboard outputs.
