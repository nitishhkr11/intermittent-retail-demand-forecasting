# References

Verified via web search on 16-Sep-2026 — titles, years, venues, and DOIs
below were checked against publisher/index pages, not recalled from
memory. Where a precise page-level detail wasn't independently
confirmable, it's been left out rather than guessed.

## Intermittent-demand forecasting & classification

1. **Croston, J.D. (1972).** Forecasting and stock control for
   intermittent demands. *Operational Research Quarterly*, 23(3), 289–303.
   — Origin of intermittent-demand forecasting: separately smooths demand
   size and inter-demand interval.

2. **Syntetos, A.A., Boylan, J.E., & Croston, J.D. (2005).** On the
   categorization of demand patterns. *Journal of the Operational
   Research Society*, 56(5), 495–503.
   DOI: 10.1057/palgrave.jors.2601841
   — Source of the ADI/CV² classification scheme and the 1.32 / 0.49
   thresholds used in this project (`src/analysis/demand_classification.py`).

3. **Teunter, R.H., Syntetos, A.A., & Babai, M.Z. (2011).** Intermittent
   demand: Linking forecasting to inventory obsolescence. *European
   Journal of Operational Research*, 214(3), 606–615.
   — Source of the TSB method planned for Phase 2.

## Dataset

4. **Makridakis, S., Spiliotis, E., & Assimakopoulos, V. (2022).** M5
   accuracy competition: Results, findings, and conclusions.
   *International Journal of Forecasting*, 38(4), 1346–1364.
   DOI: 10.1016/j.ijforecast.2021.11.013

5. **Makridakis, S., Spiliotis, E., & Assimakopoulos, V. (2022).** The
   M5 competition: Background, organization, and implementation.
   *International Journal of Forecasting*, 38(4), 1325–1336.
   DOI: 10.1016/j.ijforecast.2021.07.007
   — Both describe the dataset this project uses (Walmart M5, 3,049
   products, 10 stores, 3 states, 2011-01-29 to 2016-06-19).

## Machine learning / deep learning models (Phase 2–3)

6. **Chen, T., & Guestrin, C. (2016).** XGBoost: A scalable tree
   boosting system. *Proceedings of the 22nd ACM SIGKDD International
   Conference on Knowledge Discovery and Data Mining (KDD '16)*, 785–794.
   DOI: 10.1145/2939672.2939785

7. **Hochreiter, S., & Schmidhuber, J. (1997).** Long short-term memory.
   *Neural Computation*, 9(8), 1735–1780.
   DOI: 10.1162/neco.1997.9.8.1735

8. **Taylor, S.J., & Letham, B. (2018).** Forecasting at scale. *The
   American Statistician*, 72(1), 37–45.
   DOI: 10.1080/00031305.2017.1380080

## Forecast evaluation methodology

9. **Hyndman, R.J., & Koehler, A.B. (2006).** Another look at measures
   of forecast accuracy. *International Journal of Forecasting*, 22(4),
   679–688.
   DOI: 10.1016/j.ijforecast.2006.03.001
   — Basis for this project's decision to avoid MAPE where zero-demand
   periods make it degenerate (see `config.yaml` metrics notes, Phase 2).

10. **Hyndman, R.J., & Athanasopoulos, G. (2021).** *Forecasting:
    Principles and Practice* (3rd ed.). OTexts.
    Available at: https://otexts.com/fpp3/
    — General walk-forward / backtesting methodology reference.

## Dataset license note

The M5 dataset is distributed through Kaggle under that competition's
terms of use (provided by Walmart / the Makridakis Open Forecasting
Center for the M5 Accuracy competition). This project uses it for
non-commercial academic coursework. Before any external
publication or redistribution of derived data, check the current terms
at the competition rules page:
https://www.kaggle.com/c/m5-forecasting-accuracy/rules
— this note reflects what's publicly documented there as of this
search; it is not a substitute for reading the current rules yourself.
