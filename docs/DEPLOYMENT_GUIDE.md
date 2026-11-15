# RiverDemand dashboard deployment

The dashboard entry point is `app/app.py`. Its separately pinned runtime is in `app/requirements.txt`; it does not install the model-training stack. Streamlit Community Cloud deploys from a GitHub repository and requires repository admin access for the owner account. Current provider instructions: [connect GitHub](https://docs.streamlit.io/deploy/streamlit-community-cloud/get-started/connect-your-github-account) and [deploy an app](https://docs.streamlit.io/deploy/streamlit-community-cloud/deploy-your-app/deploy).

The app reads saved development data and result artifacts, plus compact aggregate/item/class final-holdout metrics and the one-time evaluation record. It does not train models or need raw M5 files. Build the self-contained dashboard-data archive from the project root:

```powershell
python -m src.deployment.dashboard_package
```

The current archive is `reports/final/dashboard_data_bundle_final_evaluation.zip`. Extract it at the repository root in the deployment build so its `artifacts/` paths align with the app. Verify both `dashboard_data_manifest.json` and `artifacts/dashboard_data/development_data_manifest.json` before publishing. Dashboard sales inputs are rebuilt with Parquet date filters and end at 2016-04-24; row-level reserved-period forecasts and full-history sales parquet files are not packaged. Raw M5 inputs, virtual environments, training checkpoints and credentials are excluded. The bundle includes development-only aggregate/item history, saved forecasts and metrics, and expanded comparison summaries without the large all-item daily forecast table.

The builder fails when a required dashboard artifact is missing, filters sales reads at the development cutoff, and refuses to include raw input files. An earlier archive that included full-history data is retained under `reports/final/_superseded/` for audit only and must not be deployed.

Select `app/app.py` as the Community Cloud entry point. The `app/requirements.txt` file pins the dashboard runtime. Do not deploy until the data bundle is present in the build environment and the six-page browser smoke checks pass. A public URL must only be reported after provider build success and anonymous-browser verification.

## Current deployment status

The project has no authenticated Community Cloud session or verified deploy permission. The reviewed Git branch was pushed successfully, and GitHub Actions run 43656567846 passed on commit 5a1ed87; the GitHub CLI is not installed. A genuine public deployment has not been created. Complete GitHub/Streamlit account authorization and repository admin access, then deploy the reviewed branch through the provider UI. Do not put credentials in the repository or chat.


Final package status (15 November 2026): the final-evaluation bundle is rebuilt after the one-time holdout run. It includes summary metrics/record, excludes row-level final forecasts, and does not constitute a live deployment.


Final bundle manifest check (15 November 2026): `dashboard_data_bundle_final_evaluation.zip`, 7,431,020 bytes, SHA-256 `1ba069f8c14470e4cc55ff18268011cc1d8d299b30a15fe0e91502781987cdb8`; 99 manifest-listed files verified. Raw inputs and row-level final-holdout forecasts are absent. The archive includes candidate/freeze metadata and compact result summaries.
