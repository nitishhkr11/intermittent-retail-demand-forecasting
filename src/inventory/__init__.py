"""Inventory decision-support layer: reorder point, safety stock and their backtest.

See `src.inventory.reorder` for the method and its assumptions. Everything operational (lead time, review
period, service level) is a user-supplied or illustrative input; no cost or savings model is included.
"""

from src.inventory.reorder import (
    ASSUMPTIONS_TEXT,
    FORECAST_VS_DECISION_TEXT,
    SERVICE_LEVEL_TEXT,
    InventoryAssumptions,
    ReorderPointResult,
    assumptions_from_config,
    bootstrap_safety_stock_interval,
    compute_reorder_point,
    cumulative_error_samples,
    expected_demand_over_interval,
    explain_result,
    protection_interval_days,
    reorder_point_at_origin,
    safety_stock_from_errors,
    service_level_backtest,
    summarize_service_level_backtest,
)

__all__ = [
    "ASSUMPTIONS_TEXT",
    "FORECAST_VS_DECISION_TEXT",
    "SERVICE_LEVEL_TEXT",
    "InventoryAssumptions",
    "ReorderPointResult",
    "assumptions_from_config",
    "bootstrap_safety_stock_interval",
    "compute_reorder_point",
    "cumulative_error_samples",
    "expected_demand_over_interval",
    "explain_result",
    "protection_interval_days",
    "reorder_point_at_origin",
    "safety_stock_from_errors",
    "service_level_backtest",
    "summarize_service_level_backtest",
]
