# RiverDemand
### Forecasting Intermittent Retail Demand for Inventory Decision Support

**RiverDemand** is a retail demand forecasting project built on the Walmart M5 Forecasting–Accuracy dataset. It combines demand-pattern classification, classical time-series forecasting, intermittent-demand methods, and chronological walk-forward evaluation to investigate how forecasting performance varies across retail demand patterns.

The long-term goal is to translate reliable forecasts into more informed inventory replenishment decisions.

## Project Status

**Current milestone: Phase 2 — Classical and intermittent-demand forecasting**

| Component | Status |
|---|---|
| Data preprocessing and validation | Implemented |
| Exploratory data analysis | Implemented |
| Demand classification using ADI and CV² | Implemented |
| Classical forecasting and walk-forward evaluation | Implemented |
| Croston and TSB intermittent-demand models | Implemented and evaluated |
| XGBoost pooled forecasting (20 representative products) | Implemented and evaluated on development folds |
| LSTM/GRU deep-learning forecasting | Planned |
| Prophet comparison | Planned |
| Inventory decision support and interactive dashboard | Planned |

## Problem Statement

Retail products frequently exhibit different demand patterns. Some sell consistently, while others have long periods of zero sales interrupted by irregular demand. Applying the same forecasting method to every product can produce inconsistent results.

RiverDemand investigates three questions:

1. How can retail products be classified according to their demand characteristics?
2. How do classical and intermittent-demand forecasting methods compare under chronological validation?
3. Does forecasting performance vary by demand category, and how can these differences inform future inventory decisions?

## Dataset and Scope

The project uses the Walmart M5 Forecasting–Accuracy dataset, which contains retail sales histories, calendar information, and selling prices.

The analysis deliberately focuses on a manageable subset rather than the entire catalog.

| Property | Value |
|---|---|
| Store | `CA_1` |
| Department | `FOODS_3` |
| Products analyzed | 823 |
| Daily observations per product | 1,941 |
| Date range | 2011-01-29 to 2016-05-22 |
| Product-level panel size | 1,597,443 item-date rows |
| Representative products | 20 — five per demand category |

The products are classified using Average Demand Interval (ADI) and squared coefficient of variation (CV²) into four demand categories: **Smooth, Intermittent, Erratic, and Lumpy**.

| Demand category | Products | Share |
|---|---:|---:|
| Intermittent | 460 | 55.89% |
| Lumpy | 256 | 31.11% |
| Smooth | 80 | 9.72% |
| Erratic | 27 | 3.28% |

Twenty representative products—five per category—are selected for detailed item-level experiments.

## Methodology

### 1. Data Engineering and Exploratory Analysis

The preprocessing workflow loads the M5 sales, calendar, and selling-price files; restricts the data to the selected store and department; transforms the sales history into a daily item-date panel; and validates dates, records, and data quality.

Exploratory analysis examines missing values, sales distributions, temporal patterns, demand-category frequencies, and representative product histories.

### 2. Demand Classification

ADI and CV² are used to characterize demand frequency and variability. This classification provides a structured way to examine how forecasting behavior differs between continuous and intermittent sales patterns.

### 3. Classical Forecasting

The aggregate daily demand series is evaluated using:

- Seasonal Naive
- ARIMA
- SARIMA
- Holt-Winters exponential smoothing

Stationarity tests, ACF/PACF analysis, residual diagnostics, and convergence checks support model development and interpretation.

### 4. Intermittent-Demand Forecasting

For the 20 representative products, the experiments compare:

- Seasonal Naive
- Croston's method
- Teunter–Syntetos–Babai (TSB)

These methods are evaluated across the four demand categories to investigate whether their relative performance depends on the characteristics of the product's sales history.

### 5. Leakage-Aware Evaluation

Model comparisons use expanding-window walk-forward validation rather than random train/test splitting. Each development fold uses historical observations to forecast a subsequent 28-day period.

The final 28 days are reserved as an untouched holdout. The reported development results do not use that holdout for model selection.

## Classical Forecasting Results

The following table summarizes the mean metrics from five expanding-window folds for the **aggregate CA_1 / FOODS_3 daily sales series**.

| Model | MAE ↓ | RMSE ↓ | MAPE ↓ |
|---|---:|---:|---:|
| Seasonal Naive | 256.55 | 329.05 | 12.08% |
| ARIMA(6,1,1) | 245.61 | 333.37 | 11.32% |
| SARIMA(6,1,1)(0,0,2,7) | 227.96 | 302.22 | 10.64% |
| Holt-Winters, additive trend | **193.35** | **266.44** | **9.00%** |

*Lower values are better. MAPE excludes observations with zero actual demand. These are development-period aggregate-series results, not item-level scores or final-holdout results.*

Holt-Winters achieved the lowest mean MAE, RMSE, and MAPE among these aggregate-series candidates. SARIMA also improved on the Seasonal Naive baseline across all three reported metrics. ARIMA improved MAE and MAPE but did not improve RMSE.

These findings apply to the selected aggregate series and evaluation design; they do not establish a universally best forecasting method for every individual product.

## Intermittent-Demand Results

The table below gives category-level macro-average MAE and RMSE across the five selected products in each category and the walk-forward folds.

| Category | Method | MAE ↓ | RMSE ↓ |
|---|---|---:|---:|
| Smooth | Seasonal Naive | 2.94 | 3.87 |
| Smooth | Croston | 2.50 | 3.22 |
| Smooth | TSB | **2.30** | **3.04** |
| Intermittent | Seasonal Naive | 1.10 | 1.55 |
| Intermittent | Croston | 1.03 | 1.32 |
| Intermittent | TSB | **0.95** | **1.22** |
| Erratic | Seasonal Naive | 4.04 | 5.12 |
| Erratic | Croston | 4.39 | 5.11 |
| Erratic | TSB | **3.39** | **4.26** |
| Lumpy | Seasonal Naive | 1.29 | 1.80 |
| Lumpy | Croston | 1.59 | 1.86 |
| Lumpy | TSB | **1.19** | **1.48** |

TSB outperformed the Seasonal Naive baseline on both MAE and RMSE in all four categories in this selected sample. Croston improved MAE for Smooth and Intermittent products but did not do so for Erratic and Lumpy products.

These category-level findings are exploratory and should not be generalized to all 823 products without further evaluation.

## Diagnostics and Limitations

The project includes residual autocorrelation checks and model-convergence diagnostics. Some SARIMA folds exhibited convergence issues, and significant residual autocorrelation remained in the diagnostic tests. These limitations are documented rather than treating the models as statistically perfect.

Other limitations include the restricted store-department scope, the small representative product sample used in intermittent-demand comparisons, and the absence of reported final-holdout performance at this stage.

The forecasting experiments establish a reproducible baseline for subsequent model extensions; they do not yet establish improvements in real-world inventory costs or service levels.

## Technology Stack

- **Language:** Python
- **Data processing:** pandas, NumPy
- **Statistical forecasting:** statsmodels, pmdarima
- **Evaluation:** scikit-learn and project-specific backtesting utilities
- **Visualization:** Matplotlib and Seaborn
- **Testing:** pytest
- **Notebooks:** Jupyter

See `requirements.txt` for the project's dependency specification.

## Getting Started

### Prerequisites

Install a compatible Python version and Git. Clone the repository:

```bash
git clone https://github.com/nitishhkr11/intermittent-retail-demand-forecasting.git
cd intermittent-retail-demand-forecasting
```

Create and activate a virtual environment.

**Windows PowerShell**

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

### Prepare the Dataset

Download the Walmart M5 Forecasting–Accuracy dataset from Kaggle, subject to its applicable terms. Place these files in `data/raw/`:

```text
sales_train_evaluation.csv
calendar.csv
sell_prices.csv
```

The raw dataset is not included in this repository. The `.gitignore` configuration excludes raw, intermediate, and processed data files.

### Run the Data Pipeline

From the repository root, run:

```bash
python -m src.data.preprocessor
python -m src.analysis.demand_classification
```

Then open the notebooks in order:

1. `notebooks/01_data_understanding.ipynb`
2. `notebooks/02_eda.ipynb`
3. `notebooks/03_demand_classification.ipynb`
4. `notebooks/04_classical_models.ipynb`
5. `notebooks/05_intermittent_models.ipynb`

The notebooks document the analysis and forecasting experiments. Run them using a kernel associated with the project's environment.

### Run Tests

```bash
pytest tests/
```

## Repository Structure

```text
intermittent-retail-demand-forecasting/
├── app/                  # Streamlit dashboard; implementation planned
├── artifacts/            # Generated forecasts, metrics and model outputs
├── config.yaml            # Centralized project configuration
├── data/                  # Local raw, intermediate and processed data
├── notebooks/             # EDA, classification and forecasting experiments
├── reports/
│   ├── phase1/            # Synopsis analysis and figures
│   ├── phase2/            # Forecasting diagnostics and comparison figures
│   ├── phase3/            # XGBoost Step 1 report and figures; further work pending
│   └── final/             # Final project deliverables
├── src/
│   ├── analysis/          # Demand classification, EDA and diagnostics
│   ├── data/              # Loading, preprocessing, validation and splitting
│   ├── evaluation/        # Metrics, backtesting and model comparisons
│   ├── inventory/         # Inventory decision-support components
│   ├── models/            # Classical and intermittent-demand models
│   ├── utils/             # Configuration, paths, logging and plotting
│   └── pipeline.py        # Pipeline orchestration
├── tests/                 # Automated tests
├── requirements.txt
├── LICENSE
└── README.md
```

## Roadmap

### Phase 2 — Classical and Intermittent-Demand Forecasting

Completed components include chronological evaluation, aggregate-series comparisons, Croston/TSB experiments, and forecasting diagnostics.

### Phase 3 — Advanced Forecasting and Inventory Decision Support

**Completed: Step 1 — XGBoost forecasting**

- Implemented a pooled XGBoost regressor using leakage-safe lag, rolling, calendar, and demand-category features.
- Evaluated the model on 20 representative products using five expanding-window folds with a 28-day forecast horizon.
- Compared development-fold MAE, RMSE, and MAPE against Seasonal Naive, Croston, and TSB.
- Added automated tests, diagnostic figures, and the report `reports/phase3/XGBoost_Step1_Report.md`.
- The full automated test suite passed 363 tests in the reported run.

The XGBoost results are development-fold results only. The final 28-day holdout has not been evaluated, and no universal winning model has been established.

**Remaining work**

- Implement and evaluate LSTM/GRU forecasting.
- Evaluate Prophet where appropriate.
- Complete cross-model comparison using compatible evaluation periods and metrics.
- Develop inventory decision support with explicit lead-time, safety-stock, and replenishment assumptions.
- Complete the interactive Streamlit dashboard.
- Freeze model selection, conduct the final holdout evaluation under the documented protocol, and prepare the final report and presentation.

These extensions will be evaluated before being presented as completed results.

## License

This project is distributed under the MIT License. See [`LICENSE`](LICENSE) for details.

## Author

**Nitish Kumar**

GitHub: [@nitishhkr11](https://github.com/nitishhkr11)