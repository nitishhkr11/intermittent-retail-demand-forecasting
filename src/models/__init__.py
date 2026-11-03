"""Model layer — forecasting model wrappers behind a common interface.

base.py: BaseForecaster fit/predict interface.
baseline.py: Seasonal Naive (Phase 2 Step 2's mandatory baseline).
arima.py / sarima.py / holt_winters.py: classical aggregate-series models (Phase 2).
croston.py / tsb.py: intermittent-demand models, evaluated at item level (Phase 2).
xgboost_model.py: pooled and per-item gradient boosting (Phase 3 Step 1).
lstm_gru.py: LSTM / GRU direct 28-day forecasters for the aggregate series, PyTorch, optional dependency (Phase 3).
prophet_model.py: Prophet additive / multiplicative forecasters for the aggregate series, optional dependency (Phase 3).
"""
