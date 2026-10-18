# Figure Manifest — Phase 1 Synopsis

Maps each figure to where it's used, what it demonstrates, and where
it comes from. All six PNG files now exist, generated from Nitish's
real local M5 run (CA_1/FOODS_3) via `notebooks/02_eda.ipynb` and
`03_demand_classification.ipynb`, and are embedded directly in
`Phase1_Synopsis_FINAL.pptx` at the slides listed below — no
placeholder boxes remain in the deck.

| Figure filename | Slide | What it demonstrates | Source |
|---|---|---|---|
| `missing_values.png` | 7 — Dataset Snapshot & Quality | Which columns have missing values and how much; distinguishes expected gaps (event/price columns) from unexpected ones | `notebooks/02_eda.ipynb` §3, via `src/analysis/eda.py::summarize_missing_values` |
| `aggregate_sales.png` | 8 — EDA & Demand Patterns (backup / appendix) | Full-catalog daily demand over the observed window — overall level and shape | `notebooks/02_eda.ipynb` §4, via `src/analysis/eda.py::build_aggregate_series` |
| `day_of_week_pattern.png` | 8 — EDA & Demand Patterns (backup / appendix) | Weekly seasonality — mean demand by day of week | `notebooks/02_eda.ipynb` §4, via `src/analysis/eda.py::compute_day_of_week_pattern` |
| `demand_category_distribution.png` | 8 — EDA & Demand Patterns (**main, selected**) | How many CA_1/FOODS_3 items fall into each demand class | `notebooks/02_eda.ipynb` §5 / `notebooks/03_demand_classification.ipynb`, via `src/analysis/eda.py::summarize_demand_classes` |
| `adi_cv2_quadrants.png` | 8 — EDA & Demand Patterns (**main, selected**) | The ADI/CV² classification itself — every item plotted against the 1.32 / 0.49 thresholds | `notebooks/02_eda.ipynb` §6 / `notebooks/03_demand_classification.ipynb`, via `src/analysis/eda.py::plot_adi_cv2_quadrants` (shared function, not duplicated) |
| `representative_demand_patterns.png` | 8 — EDA & Demand Patterns (**main, selected**) | Raw daily demand for 1-2 items per category — visually shows *why* the categories differ in shape | `notebooks/02_eda.ipynb` §7, via `src/analysis/eda.py::plot_representative_series` |

## Why only 3 of 6 on the main slide

Slide 8 shows the three figures marked **main, selected** above
(`demand_category_distribution.png`, `adi_cv2_quadrants.png`,
`representative_demand_patterns.png`) — together they tell the core
story: how many items are in each category, where they sit on the
ADI/CV² plane, and what that actually looks like as a time series.
`aggregate_sales.png` and `day_of_week_pattern.png` are general EDA
context rather than demand-pattern findings specifically; they're
available as backup/appendix slides if asked about in the viva, not
on the main slide.

## Statement-type labeling used on Slide 8

Every claim on Slide 8 is tagged as one of:
- **Observed** — read directly off a generated figure/table, once real
  data is in place (e.g., "N items fall in the Lumpy category")
- **Interpretation** — a reasonable reading of an observed result that
  isn't itself directly measured (e.g., "Lumpy items are likely to be
  the hardest to forecast accurately")
- **Hypothesis** — a claim about Phase 2/3 results that hasn't been
  tested yet (e.g., "TSB is expected to outperform Croston on Lumpy
  items")

No accuracy claim of any kind appears, since no model has been run.
