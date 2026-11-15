# RiverDemand Dashboard Browser QA

Date: 2026-11-14  
App entry point: `app/app.py`

## Browser results

- `.venv\Scripts\python.exe -m pytest tests/app -q`: **199 passed** in 401.39 seconds.
- `.venv\Scripts\python.exe -m pytest -q -m "not slow"`: **943 passed, 68 deselected**, 1 ZMQ Proactor warning in 600.05 seconds.
- Tablet/mobile recapture after closing Streamlit's default navigation drawer: **24 passed**, 39 deselected.
- Focused model comparison, pipeline, holdout guard, SBA and theme regression checks: **127 passed** in 9.25 seconds.
- The app rendered all six pages in light and dark modes at 1440×900, 1366×768, 768×1024 and 390×844. No Streamlit exceptions or horizontal page overflow were found.

## Computed styles at 1440×900

Captured from the running Chromium browser with `?theme=light`:

| Surface | Computed background |
|---|---|
| Streamlit app | `rgb(255, 255, 255)` |
| Main region | `rgb(255, 255, 255)` |
| Sidebar | `rgb(255, 255, 255)` |
| Header | `rgb(255, 255, 255)` |
| Main content container | `rgb(255, 255, 255)` |

The main content container measured 1,168 px wide, with a computed `max-width` of 1,280 px. The browser test also confirmed the dark app and sidebar remain dark and the theme toggle persists while navigating.

## Captures

- Final browser captures: `reports/final/dashboard_browser_qa/after/` — 48 PNGs, named `<page>_<theme>_<width>.png`.
- Earlier comparison set: `reports/final/dashboard_ui_review/before/` and `reports/final/dashboard_ui_review/after/` — 36 PNGs per directory.
- At 768 px and 390 px, the capture closes Streamlit's default open navigation drawer to show the responsive content area. The browser viewport checks use the same state and assert no horizontal overflow.

## Final holdout

The sidebar still reports **Not evaluated**. The holdout was not opened, frozen, scored, or included in the dashboard bundle.
