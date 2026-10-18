# Phase 1 Synopsis — Preparation Notes

Author: Nitish Kumar · Course: CSET369 (Time Series Analysis)
Project: Forecasting Intermittent Retail Demand — A Comparative
Classical–Machine Learning Framework for Inventory Decision Support
using Walmart M5 Data

This document backs the 10-slide synopsis presentation
(`reports/phase1/Phase1_Synopsis_FINAL.pptx` — see `figure_manifest.md`
for figure placement). It's organized around the four official rubric
criteria so each can be checked off directly. Every real-data number
below is now the verified value from Nitish's local M5 run (CA_1 /
FOODS_3), matching the finalized slide deck — see "Real values used"
at the end for the full source list.

---

## 1. Problem Domain Identification & Analysis

**Domain:** Retail / Supply Chain Analytics — specifically, demand
forecasting for retail inventory decisions.

**The problem:** Retail SKU-level demand is frequently *intermittent* —
long runs of zero-sales days interrupted by irregular positive demand.
This breaks the assumptions standard forecasting methods rely on
(continuity, roughly-normal errors), and forecasting the wrong way for
a given demand shape leads directly to bad inventory decisions:
overstock (tied-up capital, waste, markdowns) or stockouts (lost
sales, dissatisfied customers). Generic "one model for every SKU"
forecasting is a poor fit precisely because different products have
structurally different demand shapes.

**This is a hypothesis at Phase 1, not a finding.** No model has been
run yet — the claim is the *motivation* for the comparative design in
Phase 2/3, not a result.

**Why it matters (stakeholder framing):**
- Inventory planners need reorder points and safety stock that reflect
  a product's actual demand pattern, not a one-size-fits-all rule.
- Supply-chain managers care about the cost of both overstock and
  stockout, which forecasting quality directly affects.
- Category managers need to know which of their products are
  fundamentally hard to forecast (Lumpy) versus straightforward
  (Smooth), since that changes what inventory policy is even
  appropriate.

## 2. Review of Existing Systems & Project Feasibility

See the comparison table below (also on Slide 5) and
`reports/phase1/references.md` for full citations — every method here
is a verified, published source, not an invented finding.

| Approach | What it does | Strength | Limitation | Gap addressed |
|---|---|---|---|---|
| Classical / exponential smoothing (Holt-Winters) | Smooths trend + seasonality via weighted averages | Simple, interpretable, strong for dense/regular demand | Assumes continuous non-zero demand; degrades under zero-inflation | Baseline for the Smooth category |
| Croston's method (Croston, 1972) | Separately smooths demand size and inter-demand interval, updates only on non-zero periods | First method built specifically for intermittent demand | Structurally biased (shown by Syntetos & Boylan, 2005) | Handles Intermittent-category sparsity classical methods can't |
| SBA (Syntetos, Boylan & Croston, 2005) | Bias-corrected Croston, plus the ADI/CV² taxonomy this project uses | Corrects Croston's bias; gives a principled demand-pattern classification | Still assumes fairly stable demand size; slow to react to obsolescence | Basis for this project's Smooth/Intermittent/Erratic/Lumpy taxonomy |
| TSB (Teunter, Syntetos & Babai, 2011) | Updates demand *probability* every period, not just at demand events | Reacts faster to declining/obsolete demand | More parameters; still an exponential-smoothing-family model | Candidate for Lumpy / obsolescence-prone items (Phase 2) |
| XGBoost (Chen & Guestrin, 2016) | Global gradient-boosted-tree model over lag/calendar/price features across all series | Captures cross-item and calendar/price effects; strong track record in the M5 competition itself | Needs feature engineering; no native prediction interval | Candidate for Erratic/Lumpy items with exploitable calendar/price signal (Phase 3) |
| LSTM (Hochreiter & Schmidhuber, 1997) | Recurrent network learning temporal dependence directly from the sequence | Models long-range temporal structure without hand-crafted features | Data-hungry; one ~5-year aggregate series is a small dataset for a neural net | Aggregate-series comparison point only (Phase 3) |
| Prophet (Taylor & Letham, 2018) | Additive trend + seasonality + holiday decomposition for business time series | Fast, interpretable components, minimal tuning | Assumes dense, regularly-seasonal data — not intermittent series | Aggregate-series benchmark only, explicitly excluded at item level |

**The gap this project addresses as a whole:** these methods are
individually well-studied, but rarely compared head-to-head, on the
same real dataset, stratified explicitly by demand pattern, with an
inventory-decision layer built on top of the forecasts. That
combination — not any single model — is this project's contribution.

**Feasibility:** the M5 dataset is real, public, well-documented, and
already scoped down (CA_1 / FOODS_3 — 823 items, with 20 representative
items selected at 5 per category, plus one aggregate series)
specifically so dataset acquisition isn't the hard part of this
project (see `reports/phase1/references.md` items 4–5 for the dataset
sources). Stages 1–3 (data pipeline, classification, EDA) are built,
tested, and now executed against the real local data — direct evidence
of feasibility within the course timeline.

## 3. Project Design Objectives & Methodology

**Objectives** (see Slide 9):
1. EDA and temporal characterization — *done, Stage 1–3*
2. Demand-pattern classification (ADI/CV²) — *done, Stage 2*
3. Classical/intermittent forecasting — *planned, Phase 2*
4. ML/DL comparison — *planned, Phase 3*
5. Model comparison by demand pattern — *planned, Phase 2–3*
6. Inventory decision support (reorder point / safety stock) — *planned, Phase 3*
7. Streamlit dashboard — *planned, Phase 3*

**Methodology flow** (see Slide 9):

```
M5 Data → Validation → Preprocessing → ADI/CV² Classification → EDA
   → Phase 2 Forecasting (Seasonal Naive, ARIMA, SARIMA, Holt-Winters,
      Croston, TSB, walk-forward backtesting, RMSE/MAE/MAPE)
   → Phase 3 ML/DL (XGBoost, LSTM/GRU, Prophet)
   → Comparative Evaluation (by demand pattern)
   → Inventory Decision Support
   → Streamlit Dashboard
```

Phase 2/3 modeling is **planned, not yet executed** — nothing past EDA
has been run. Validation is chronological only (no random
train/test splitting), enforced by the locked architecture's
evaluation-layer design.

## 4. Communication Clarity & Organization

Handled by the presentation's visual design (monochrome, consistent
with the eventual dashboard — see `figure_manifest.md`) and by keeping
every slide's claims explicitly labeled as one of: **directly observed
result**, **interpretation**, or **future modeling hypothesis** (see
Slide 8 in particular). This document and the slide deck use the same
structure so a reader can move between them without translation.

---

## Slide outline (quick reference — see the deck for full content)

| # | Title | Rubric area |
|---|---|---|
| 1 | Title | — |
| 2 | Introduction to Time Series Analysis | Domain |
| 3 | Problem Statement & Motivation | Domain |
| 4 | Domain, Scope & Stakeholders | Domain / Feasibility |
| 5 | Review of Existing Systems / Related Work | Existing Systems |
| 6 | Dataset Description | Existing Systems / Feasibility |
| 7 | Dataset Snapshot & Quality | Feasibility |
| 8 | EDA & Demand Patterns | Feasibility / Objectives |
| 9 | Objectives & Proposed Methodology | Objectives & Methodology |
| 10 | Conclusion & Next Steps | Objectives & Methodology |

## Real values used (verified from Nitish's local M5 run, CA_1/FOODS_3)

Everything below was filled in from the real local run and now appears
in `Phase1_Synopsis_FINAL.pptx`; nothing in the deck or this document
is an invented or estimated figure:

- Slide 6/7: real panel row count, and the CA_1/FOODS_3 date range —
  2011-01-29 to 2016-05-22, 1,941 continuous days — quoted separately
  from the full M5 dataset's own 2011-01-29 to 2016-06-19 range on
  Slide 6
- Slide 7: 10 real sample rows from the processed panel
  (FOODS_3_001–010, all dated 2011-01-29), plus the real missing-value
  percentages for the event/price columns
- Slide 8: real item counts per demand class — 823 items total: 460
  Intermittent (55.89%), 256 Lumpy (31.11%), 80 Smooth (9.72%), 27
  Erratic (3.28%)
- Slide 8: 20 representative items selected, 5 per category
- Slide 8: all six PNG figures, generated from the real run and
  embedded directly in the deck (see `figure_manifest.md`)
- Slide 1: Date finalized (30 September 2026). Guide's name is still
  genuinely pending — left as a blank editable line in the deck rather
  than invented, and the same gap applies here
