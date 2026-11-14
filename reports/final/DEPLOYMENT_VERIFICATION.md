# RiverDemand deployment verification

## Status

**Not deployed.** No Streamlit Community Cloud login, GitHub OAuth authorization, or deployment permission was available in this execution. `gh auth status` could not run because GitHub CLI is not installed. `git ls-remote origin HEAD` succeeded read-only, but this does not establish push or deployment rights.

Streamlit's current instructions require GitHub connection and repository admin permission to deploy. The remaining external action is to sign in to Streamlit Community Cloud with the owning GitHub account and authorize repository access. After that, deploy `app/app.py` from the reviewed branch with `app/requirements.txt` and the verified data bundle.

## Local deployment preparation

- Dashboard-only pinned requirements: `app/requirements.txt`.
- Raw-data-free bundle builder: `python -m src.deployment.dashboard_package`.
- Expected archive: `reports/final/dashboard_data_bundle_final_evaluation.zip`.
- Raw M5 inputs and row-level final-holdout forecasts are excluded. Compact aggregate/item/class summary metrics and the evaluation record are included.
- No live URL, successful provider build, or anonymous-session check is claimed.

## Current local artifact check (2026-11-14)

- Current safe bundle: 4,137,819 bytes, SHA-256 `ae25fbe70f9e2ebeb433172a1afbddd06912cccf2cf128d90cac0b6d17aa6d30`.
- Verified all 75 manifest-listed file hashes against the archive. The bundle excludes full-history panel/aggregate/item-sales parquet files; its three sales files end on 2016-04-24, and the data manifest records `holdout_dates_included: false`.
- Browser QA passed for all six pages at 1440, 1366, 768 and 390 px in light and dark themes. At 1440 px, computed light mode backgrounds for the app, main region, sidebar, header and content container were all `rgb(255, 255, 255)`; main content container measured 1,168 px with a 1,280 px max width.
- Current local browser evidence is saved under `reports/final/dashboard_browser_qa/after/`.

An earlier archive built before the sales cutoff was enforced included full-history sales parquet files. That archive was moved to `reports/final/_superseded/` and is marked **DO NOT DEPLOY**; the development-only bundle above supersedes it.

## Provider acceptance checks still required

1. Confirm provider build success.
2. Open the resulting URL in an anonymous/private browser session.
3. Verify all six pages, both themes, charts, tables, downloads and no missing-artifact messages.
4. Refresh the app and repeat the check.

Provider reference: [Streamlit Community Cloud deployment](https://docs.streamlit.io/deploy/streamlit-community-cloud/deploy-your-app/deploy).


## Final bundle check — 15 November 2026

The final-evaluation bundle was rebuilt after the one-time holdout run. It packages the hashed development/dashboard inputs plus compact final results, class summary, and evaluation record; it does not include row-level final predictions or raw M5 files. Public deployment and a remote GitHub Actions result remain unverified.


Final bundle manifest check (15 November 2026): `dashboard_data_bundle_final_evaluation.zip`, 8,458,777 bytes, SHA-256 `9c1d3fd5eff3a24eeb42375e25bf1297d79a2380bca680f824fb54d65ab114f7`; 112 manifest-listed files verified. Raw inputs and row-level final-holdout forecasts are absent. The archive includes candidate/freeze metadata and compact result summaries.
