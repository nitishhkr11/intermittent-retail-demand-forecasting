# RiverDemand Final Content Consistency Audit

**Audit date:** 2026-11-15  
**Branch:** `finalization/riverdemand-research-completion`  
**Commit inspected:** `a8b7a6a20215c12ba6fe163532ba0ab39a974057`  
**Scope:** current 19-slide presentation, matching PDF, technical report, speaking script, viva guide, final reports and saved experiment evidence.

## Acceptance matrix

| Requirement | Saved evidence / check | Status |
|---|---|---|
| Expanded classical results | 16 seasonal windows; 112 model-fold rows; no failed folds; Phase 2 selection overlap disclosed | VERIFIED COMPLETE |
| Expanded intermittent results | 823 products × 16 windows; 52,672 item/model/fold rows; 1,474,816 forecasts | VERIFIED COMPLETE |
| SBA implementation and unavailable cells | Croston/SBA zero-prefix failures retained; no unavailable forecasts imputed | VERIFIED COMPLETE |
| XGBoost feature ablations | Five pooled configurations; 80/80 fits; fixed 183-item per-item sample; independent audit found 0 metric mismatches | VERIFIED COMPLETE |
| LSTM/GRU study | 64/64 variant-fold cells; 5 seeds each; 320 fits; 0 failed cells | VERIFIED COMPLETE |
| Expanded Prophet study | 48/48 fits; nominal 80% interval coverage observed at 65.63%–67.63% | VERIFIED COMPLETE |
| Inventory calibration | Four rules, 930,248 common windows each; best observed 92.594% vs 95% target | VERIFIED COMPLETE; TARGET NOT REACHED |
| Frozen candidate selection | `artifacts/final/FROZEN_CANDIDATES.json`; SHA-256 `f4f36f8a78cafcc88df4be10829c609690c6115cbcd1725a4268fd683fc261ea` | VERIFIED COMPLETE |
| One-time final holdout record | `artifacts/final/FINAL_EVALUATION_RECORD.json`; SHA-256 `a680a15ba458a814387c049980c599d792de79c7065d7565c2a9f5f70b9f092d`; record count 1; matching hashes | VERIFIED COMPLETE; DO NOT RERUN |
| Development pipeline / resume | `python -m src.pipeline development --resume`; all 14 hash-verified stages skipped without rewriting or retraining | VERIFIED COMPLETE |
| Six-page dashboard | Six pages, light/dark, five widths; 60 screenshots; white light main surface computed as RGB(255,255,255) | VERIFIED COMPLETE |
| Public deployment | No authenticated provider session or verified public URL | BLOCKED BY EXTERNAL AUTHENTICATION |
| GitHub Actions | Run [43660358970](https://github.com/nitishhkr11/intermittent-retail-demand-forecasting/actions/runs/43660358970), passed on `a8b7a6a` | VERIFIED COMPLETE |
| Local tests | App: 217 passed. Non-slow: 944 passed, 15 skipped, 86 deselected | VERIFIED COMPLETE; counts kept distinct |
| 19-slide presentation and PDF | PPTX has 19 slides; matching PDF has 19 pages; exported from the final saved PPTX and rendered for review | VERIFIED COMPLETE |
| Technical report | DOCX and 3-page PDF generated and rendered for visual review | VERIFIED COMPLETE |
| Speaking script | DOCX/PDF and Markdown; follows 19 slides; core 15–20 minute and short 8–10 minute delivery guidance; QA and transitions for each slide | VERIFIED COMPLETE |
| Dashboard viva guide | DOCX/PDF and Markdown; all six pages, formulas, source map, limits and likely questions | VERIFIED COMPLETE |
| Reproducibility documentation | Existing `docs/EXPERIMENT_REPRODUCIBILITY.md` plus saved artifact manifests and run metadata | VERIFIED COMPLETE |

## Numerical consistency checks

- Aggregate development LSTM(calendar): MAE 185.092, fold SD 60.742, RMSE 235.451. Final holdout: MAE 128.0869, RMSE 183.1430, WAPE 0.05434, MASE 0.49197, bias −63.9673. The different periods are labelled separately.
- Final item class-selected strategy: macro MAE 1.683711, RMSE 2.188407, WAPE 1.007349, MASE 1.375490, bias −0.215173. MAPE 62.4531% excludes 8,432 zero actuals. These metrics are not compared directly with aggregate units/day.
- Inventory service: all-history 92.475%, recent two folds 86.724%, recency-weighted 91.654%, same-class 92.594%, target 95%. Same-class is about 0.12 percentage points above all-history with overlapping intervals; all-history remains the simpler default.
- Prophet nominal daily interval target 80%; observed variant coverage 65.63%–67.63%. Split-conformal sensitivity is identified as exploratory and broad, not an operating guarantee.
- Holdout status is “evaluated once after freeze,” not “not evaluated.” The frozen manifest and evaluation record are immutable. Do not rerun or tune against that period.
- Public deployment is not claimed. The final local dashboard bundle is separate from a hosted public application.

The frozen development-selection snapshot is immutable and was written before final scoring, so its embedded `holdout_status` field says “not evaluated” as-of-freeze. That historical field was not edited. The later authoritative `FINAL_EVALUATION_RECORD.json` records the one-time post-freeze evaluation (count 1); all current final-facing materials use that status.

## Presentation QA

The final deck uses the existing cream, maroon, blue and gold visual identity. All 19 rendered pages were checked. The synthesis slide tables have separate labels and no overlapping reading text. Slide count, page footer numbering, final-result metric labels, slide 17 settings table and PDF page count were checked. The matching PDF reports 19 pages at 16:9 page size.

Slide 17 settings table geometry was reopened from the saved final PPTX and asserted in OOXML-backed dimensions: top 1.40 in (1,280,160 EMU), nine rows exactly 548,640 EMU / 0.60 in each, total 4,937,760 EMU / 5.40 in, bottom 6.80 in (6,217,920 EMU).

## Remaining limitations

Public hosting requires an authenticated Streamlit Community Cloud/GitHub owner session and remains unverified. The study is limited to one store and department; recorded sales may be censored by stockouts. Prophet intervals under-cover, residual autocorrelation remains in 60 testable Phase 2 cases, and no inventory calibration attained the 95% target. No cost or financial-savings claim is supported.
