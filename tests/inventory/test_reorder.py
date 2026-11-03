"""Tests for src/inventory: reorder point, safety stock, cumulative errors, the backtest of the rule."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.inventory import (
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
from src.utils.config import load_config


# ------------------------------------------------------------------------------------------ helpers
def make_table(errors_by_fold, forecast=2.0, model="M", item="A", horizon=28, start="2020-01-01"):
    """Item-level table; `errors_by_fold[k]` is the per-day (actual - forecast) pattern, length `horizon`."""
    rows = []
    day0 = pd.Timestamp(start)
    for fold, errors in enumerate(errors_by_fold, start=1):
        for i in range(horizon):
            rows.append(
                {
                    "item_id": item, "demand_class": "Lumpy", "model": model, "fold": fold,
                    "date": day0 + pd.Timedelta(days=(fold - 1) * horizon + i),
                    "actual": forecast + errors[i], "predicted": forecast, "success": True, "error": "",
                }
            )
    return pd.DataFrame(rows)


# ------------------------------------------------------------------------------------------ assumptions
def test_assumptions_defaults_and_protection_days():
    a = InventoryAssumptions(lead_time_days=7)
    assert a.review_period_days == 0 and a.service_level == 0.95 and a.on_hand is None and a.on_order == 0.0
    assert a.protection_days == 7
    assert InventoryAssumptions(lead_time_days=5, review_period_days=7).protection_days == 12


@pytest.mark.parametrize(
    "kwargs, field",
    [
        ({"lead_time_days": 0}, "lead_time_days"),
        ({"lead_time_days": -3}, "lead_time_days"),
        ({"lead_time_days": 2.5}, "lead_time_days"),
        ({"lead_time_days": True}, "lead_time_days"),
        ({"lead_time_days": "7"}, "lead_time_days"),
        ({"lead_time_days": 7, "review_period_days": -1}, "review_period_days"),
        ({"lead_time_days": 7, "review_period_days": 1.5}, "review_period_days"),
        ({"lead_time_days": 7, "service_level": 1.0}, "service_level"),
        ({"lead_time_days": 7, "service_level": 0.2}, "service_level"),
        ({"lead_time_days": 7, "service_level": float("nan")}, "service_level"),
        ({"lead_time_days": 7, "service_level": "high"}, "service_level"),
        ({"lead_time_days": 7, "on_hand": -1.0}, "on_hand"),
        ({"lead_time_days": 7, "on_hand": float("inf")}, "on_hand"),
        ({"lead_time_days": 7, "on_order": -5}, "on_order"),
    ],
)
def test_invalid_assumptions_name_the_offending_field(kwargs, field):
    with pytest.raises(ValueError, match=field):
        InventoryAssumptions(**kwargs)


def test_protection_interval_longer_than_the_forecast_horizon_is_refused():
    with pytest.raises(ValueError, match="28"):
        InventoryAssumptions(lead_time_days=20, review_period_days=9)
    with pytest.raises(ValueError, match="28"):
        protection_interval_days(29)
    assert InventoryAssumptions(lead_time_days=21, review_period_days=7).protection_days == 28
    assert protection_interval_days(7, 7) == 14


def test_assumptions_accept_numpy_integers():
    assert InventoryAssumptions(lead_time_days=np.int64(3), review_period_days=np.int32(2)).protection_days == 5


def test_assumptions_are_immutable():
    a = InventoryAssumptions(lead_time_days=7)
    with pytest.raises(Exception):
        a.lead_time_days = 3  # type: ignore[misc]


def test_assumptions_from_config_uses_the_config_defaults_and_overrides():
    config = load_config()
    a = assumptions_from_config(config)
    section = config["inventory"]
    assert (a.lead_time_days, a.review_period_days, a.service_level) == (
        section["default_lead_time_days"], section["default_review_period_days"], section["default_service_level"],
    )
    b = assumptions_from_config(config, lead_time_days=3, service_level=0.9, on_hand=10.0)
    assert (b.lead_time_days, b.service_level, b.on_hand) == (3, 0.9, 10.0)
    with pytest.raises(ValueError, match="Unknown"):
        assumptions_from_config(config, lead_days=3)
    with pytest.raises(ValueError, match="inventory"):
        assumptions_from_config({})
    with pytest.raises(ValueError, match="default_service_level"):
        assumptions_from_config({"inventory": {"default_lead_time_days": 7, "default_review_period_days": 0}})


# ------------------------------------------------------------------------------------------ building blocks
def test_expected_demand_is_the_sum_of_the_first_days():
    assert expected_demand_over_interval([1, 2, 3, 4], 3) == 6.0
    assert expected_demand_over_interval(np.array([0.0, 0.0]), 2) == 0.0


@pytest.mark.parametrize("forecast, days", [([1, 2], 3), ([], 1), ([1, np.nan], 2), ([1, -1], 2)])
def test_expected_demand_rejects_bad_forecasts(forecast, days):
    with pytest.raises(ValueError):
        expected_demand_over_interval(forecast, days)


def test_expected_demand_never_extrapolates_and_ignores_days_after_the_interval():
    assert expected_demand_over_interval([1, 1, -9], 2) == 2.0  # a later day is outside the interval


def test_safety_stock_is_the_higher_empirical_quantile_floored_at_zero():
    errors = list(range(-5, 15))  # 20 samples: -5 .. 14
    assert safety_stock_from_errors(errors, 0.90) == pytest.approx(np.quantile(errors, 0.90, method="higher"))
    assert safety_stock_from_errors([-4.0, -2.0, -1.0], 0.95) == 0.0  # never negative
    assert safety_stock_from_errors([7.0], 0.95) == 7.0  # a single sample is allowed (and warned about elsewhere)
    assert safety_stock_from_errors([0.0, 0.0], 0.9) == 0.0


def test_safety_stock_increases_with_the_service_level():
    errors = np.random.default_rng(0).gamma(2.0, 3.0, 400) - 4.0
    stocks = [safety_stock_from_errors(errors, level) for level in (0.6, 0.8, 0.9, 0.95, 0.99)]
    assert stocks == sorted(stocks) and stocks[-1] > stocks[0]


@pytest.mark.parametrize("bad", [[], [np.nan, 1.0], [np.inf], [[1.0, 2.0]]])
def test_safety_stock_rejects_invalid_samples(bad):
    with pytest.raises(ValueError):
        safety_stock_from_errors(bad, 0.9)


@pytest.mark.parametrize("level", [0.0, 0.49, 1.0, 1.2, -0.5, float("nan")])
def test_safety_stock_rejects_service_levels_outside_the_supported_range(level):
    with pytest.raises(ValueError, match="service_level"):
        safety_stock_from_errors([1.0, 2.0, 3.0], level)


# ------------------------------------------------------------------------------------------ compute_reorder_point
def test_reorder_point_hand_example():
    forecast = [2.0] * 28
    errors = np.arange(-5, 15, dtype=float)  # 20 samples
    a = InventoryAssumptions(lead_time_days=7, service_level=0.90)
    result = compute_reorder_point(forecast, errors, a)
    expected_ss = float(np.quantile(errors, 0.90, method="higher"))
    assert isinstance(result, ReorderPointResult)
    assert result.protection_days == 7 and result.expected_demand == 14.0
    assert result.safety_stock == pytest.approx(expected_ss)
    assert result.reorder_point == pytest.approx(14.0 + expected_ss)
    assert result.n_error_samples == 20 and result.service_level == 0.90 and result.error_quantile_level == 0.90
    assert result.order_up_to_level is None and result.inventory_position is None and result.suggested_order_quantity is None
    assert result.warnings == ()


def test_reorder_point_is_never_negative_and_zero_demand_is_handled():
    a = InventoryAssumptions(lead_time_days=3)
    result = compute_reorder_point([0.0] * 28, [-3.0] * 30, a)
    assert result.reorder_point == 0.0 and result.safety_stock == 0.0 and result.expected_demand == 0.0
    assert any("zero over the whole protection interval" in w for w in result.warnings)
    assert any("Safety stock is zero" in w for w in result.warnings)


def test_zero_forecast_with_positive_errors_gets_stock_from_the_errors_alone():
    result = compute_reorder_point([0.0] * 7, [2.0] * 25, InventoryAssumptions(lead_time_days=7))
    assert result.expected_demand == 0.0 and result.safety_stock == 2.0 and result.reorder_point == 2.0


def test_reorder_point_warns_on_few_error_samples_and_threshold_is_configurable():
    a = InventoryAssumptions(lead_time_days=2)
    few = compute_reorder_point([1.0] * 5, [1.0, 2.0, 3.0], a)
    assert any("Only 3 cumulative-error sample" in w for w in few.warnings)
    enough = compute_reorder_point([1.0] * 5, [1.0, 2.0, 3.0], a, min_error_windows_warning=3)
    assert not any("cumulative-error" in w for w in enough.warnings)


def test_reorder_point_with_periodic_review_reports_the_order_up_to_level():
    a = InventoryAssumptions(lead_time_days=5, review_period_days=2, service_level=0.8, on_hand=4.0, on_order=3.0)
    result = compute_reorder_point([1.0] * 28, np.arange(30, dtype=float), a)
    assert result.protection_days == 7 and result.expected_demand == 7.0
    assert result.order_up_to_level == pytest.approx(result.reorder_point)
    assert result.inventory_position == 7.0
    assert result.suggested_order_quantity == pytest.approx(max(0.0, result.reorder_point - 7.0))


def test_suggested_order_is_zero_when_stock_is_sufficient_and_absent_without_on_hand():
    a = InventoryAssumptions(lead_time_days=2, on_hand=1000.0)
    result = compute_reorder_point([1.0] * 5, [1.0] * 30, a)
    assert result.suggested_order_quantity == 0.0 and result.inventory_position == 1000.0
    assert compute_reorder_point([1.0] * 5, [1.0] * 30, InventoryAssumptions(lead_time_days=2)).suggested_order_quantity is None


def test_on_order_counts_towards_the_inventory_position():
    base = compute_reorder_point([2.0] * 7, [1.0] * 30, InventoryAssumptions(lead_time_days=7, on_hand=5.0))
    with_order = compute_reorder_point([2.0] * 7, [1.0] * 30, InventoryAssumptions(lead_time_days=7, on_hand=5.0, on_order=6.0))
    assert with_order.inventory_position == base.inventory_position + 6.0
    assert with_order.suggested_order_quantity == pytest.approx(max(0.0, base.suggested_order_quantity - 6.0))


def test_reorder_point_rejects_short_forecast_bad_errors_and_wrong_assumption_type():
    a = InventoryAssumptions(lead_time_days=7)
    with pytest.raises(ValueError, match="forecast has 3 day"):
        compute_reorder_point([1.0, 1.0, 1.0], [1.0] * 25, a)
    with pytest.raises(ValueError, match="error_samples"):
        compute_reorder_point([1.0] * 7, [], a)
    with pytest.raises(ValueError, match="error_samples"):
        compute_reorder_point([1.0] * 7, [1.0, np.nan], a)
    with pytest.raises(ValueError, match="InventoryAssumptions"):
        compute_reorder_point([1.0] * 7, [1.0] * 25, {"lead_time_days": 7})  # type: ignore[arg-type]


def test_higher_service_level_never_lowers_the_reorder_point():
    errors = np.random.default_rng(3).normal(0, 5, 200)
    points = [
        compute_reorder_point([3.0] * 28, errors, InventoryAssumptions(lead_time_days=7, service_level=s)).reorder_point
        for s in (0.5, 0.8, 0.9, 0.95, 0.99)
    ]
    assert points == sorted(points)


def test_to_dict_is_json_friendly():
    import json

    result = compute_reorder_point([1.0] * 7, [1.0] * 30, InventoryAssumptions(lead_time_days=7))
    record = result.to_dict()
    json.dumps(record)
    assert record["warnings"] == [] and record["reorder_point"] == result.reorder_point


# ------------------------------------------------------------------------------------------ cumulative errors
def test_cumulative_errors_cover_every_overlapping_window_with_sign_actual_minus_forecast():
    pattern = np.zeros(28)
    pattern[3] = 5.0  # one day with demand 5 above the forecast
    table = make_table([pattern, pattern])
    samples = cumulative_error_samples(table, 7, model="M", item_id="A")
    assert len(samples) == 2 * (28 - 7 + 1)
    # day index 3 lies inside windows starting at 0..3 -> 4 windows per fold carry +5
    assert int((samples == 5.0).sum()) == 2 * 4 and int((samples == 0.0).sum()) == 2 * (22 - 4)
    assert (samples >= 0).all()


def test_cumulative_errors_by_fold_and_window_length():
    table = make_table([np.full(28, 1.0), np.full(28, -2.0), np.full(28, 0.5)])
    only_second = cumulative_error_samples(table, 4, folds=[2], model="M", item_id="A")
    assert len(only_second) == 25 and np.allclose(only_second, -8.0)
    both = cumulative_error_samples(table, 28, folds=[1, 3], model="M", item_id="A")
    assert both.tolist() == [28.0, 14.0]
    assert cumulative_error_samples(table, 1, folds=[2], model="M", item_id="A").tolist() == [-2.0] * 28


def test_cumulative_errors_skip_failed_folds_instead_of_filling_them():
    table = make_table([np.ones(28), np.ones(28)])
    table.loc[table["fold"] == 2, "predicted"] = np.nan
    table.loc[table["fold"] == 2, "success"] = False
    samples = cumulative_error_samples(table, 7, model="M", item_id="A")
    assert len(samples) == 22
    assert cumulative_error_samples(table, 7, model="M", item_id="A", folds=[2]).size == 0


def test_cumulative_errors_refuse_ambiguous_or_invalid_requests():
    table = make_table([np.ones(28)])
    other = make_table([np.ones(28)], item="B")
    two_items = pd.concat([table, other])
    with pytest.raises(ValueError, match="several items"):
        cumulative_error_samples(two_items, 7, model="M")
    two_models = pd.concat([table, make_table([np.ones(28)], model="N")])
    with pytest.raises(ValueError, match="several models"):
        cumulative_error_samples(two_models, 7, item_id="A")
    with pytest.raises(ValueError, match="window_days"):
        cumulative_error_samples(table, 0, model="M", item_id="A")
    with pytest.raises(ValueError, match="at most 28"):
        cumulative_error_samples(table, 29, model="M", item_id="A")
    with pytest.raises(ValueError, match="No rows"):
        cumulative_error_samples(table, 7, model="missing", item_id="A")
    with pytest.raises(ValueError, match="missing column"):
        cumulative_error_samples(table.drop(columns="actual"), 7, model="M", item_id="A")


def test_cumulative_errors_detect_gaps_in_the_forecast_days():
    table = make_table([np.ones(28)])
    table = table.drop(index=5)
    with pytest.raises(ValueError, match="consecutive"):
        cumulative_error_samples(table, 7, model="M", item_id="A")


def test_cumulative_errors_work_on_an_aggregate_table_without_item_id():
    table = make_table([np.full(28, 3.0)]).drop(columns=["item_id", "demand_class"])
    samples = cumulative_error_samples(table, 7, model="M")
    assert len(samples) == 22 and np.allclose(samples, 21.0)
    with pytest.raises(ValueError, match="no item_id"):
        cumulative_error_samples(table, 7, model="M", item_id="A")


# ------------------------------------------------------------------------------------------ bootstrap
def test_bootstrap_interval_is_seeded_ordered_and_contains_the_point_estimate_for_smooth_errors():
    errors = np.random.default_rng(1).normal(10, 4, 300)
    low, high = bootstrap_safety_stock_interval(errors, 0.9, n_resamples=300, seed=7)
    assert bootstrap_safety_stock_interval(errors, 0.9, n_resamples=300, seed=7) == (low, high)
    assert low <= high and low <= safety_stock_from_errors(errors, 0.9) <= high
    assert bootstrap_safety_stock_interval(errors, 0.9, n_resamples=300, seed=8) != (low, high)


def test_bootstrap_does_not_touch_the_global_random_state():
    np.random.seed(123)
    expected = np.random.random(3)
    np.random.seed(123)
    bootstrap_safety_stock_interval(np.arange(30.0), 0.9, n_resamples=50)
    assert np.allclose(np.random.random(3), expected)


def test_bootstrap_of_a_single_value_and_of_constant_errors_is_degenerate_not_an_error():
    assert bootstrap_safety_stock_interval([4.0], 0.9, n_resamples=20) == (4.0, 4.0)
    assert bootstrap_safety_stock_interval([-2.0] * 10, 0.9, n_resamples=20) == (0.0, 0.0)


def test_block_bootstrap_is_wider_than_iid_for_dependent_samples():
    rng = np.random.default_rng(0)
    base = np.cumsum(rng.normal(0, 1, 60))  # strongly dependent, like overlapping window errors
    iid = bootstrap_safety_stock_interval(base, 0.9, n_resamples=400, seed=1, block_size=1)
    blocked = bootstrap_safety_stock_interval(base, 0.9, n_resamples=400, seed=1, block_size=14)
    assert (blocked[1] - blocked[0]) >= (iid[1] - iid[0]) * 0.8  # never dramatically narrower


@pytest.mark.parametrize(
    "kwargs", [{"n_resamples": 0}, {"block_size": 0}, {"confidence": 1.0}, {"confidence": 0.0}]
)
def test_bootstrap_rejects_invalid_settings(kwargs):
    with pytest.raises(ValueError):
        bootstrap_safety_stock_interval([1.0, 2.0, 3.0], 0.9, **kwargs)


# ------------------------------------------------------------------------------------------ backtest of the rule
def iid_table(n_items=6, seed=0, scale=3.0):
    rng = np.random.default_rng(seed)
    parts = []
    for i in range(n_items):
        errors = [rng.gamma(2.0, scale, 28) - 2 * scale for _ in range(5)]
        parts.append(make_table(errors, forecast=6.0, item=f"I{i}"))
    return pd.concat(parts, ignore_index=True)


def test_backtest_shapes_and_status_for_cells_without_enough_history():
    table = iid_table(n_items=2)
    result = service_level_backtest(table, 7, (0.8, 0.95), model="M", min_calibration_folds=2)
    assert len(result) == 2 * 5 * 2
    early = result[result["fold"] <= 2]
    assert (early["status"] != "ok").all() and early["achieved_cycle_service"].isna().all()
    ok = result[result["status"] == "ok"]
    assert sorted(ok["fold"].unique()) == [3, 4, 5]
    assert ok["achieved_cycle_service"].between(0, 1).all()
    assert (ok["n_test_windows"] == 22).all()
    assert (ok["n_calibration_samples"] == ok["fold"].sub(1) * 22).all()


def test_backtest_calibrates_on_earlier_folds_only():
    base = iid_table(n_items=1, seed=4)
    perturbed = base.copy()
    mask = perturbed["fold"] == 4
    perturbed.loc[mask, "actual"] = perturbed.loc[mask, "actual"] + 500.0  # the TEST fold gets huge demand
    a = service_level_backtest(base, 7, (0.9,), model="M")
    b = service_level_backtest(perturbed, 7, (0.9,), model="M")
    for fold in (3, 4):  # fold 3 is calibrated on folds 1-2; fold 4's own data must not change fold 4's stock
        sa = a[a["fold"] == fold]["safety_stock"].iloc[0]
        sb = b[b["fold"] == fold]["safety_stock"].iloc[0]
        assert sa == sb
    assert b[b["fold"] == 4]["achieved_cycle_service"].iloc[0] == 0.0  # ...but it shows up as stock-outs
    assert a[a["fold"] == 5]["safety_stock"].iloc[0] != b[b["fold"] == 5]["safety_stock"].iloc[0]  # later folds DO see it


def test_backtest_achieves_roughly_the_target_on_stationary_errors():
    table = iid_table(n_items=12, seed=2)
    result = service_level_backtest(table, 7, (0.8, 0.95), model="M")
    summary = summarize_service_level_backtest(result, n_resamples=200, seed=3)
    assert set(summary["target_cycle_service"]) == {0.8, 0.95}
    for row in summary.itertuples():
        assert abs(row.gap) < 0.12  # stationary errors: achieved close to the target
        assert row.ci_low <= row.achieved_cycle_service <= row.ci_high
        assert row.n_cells == 12 * 3
    assert summary.sort_values("target_cycle_service")["achieved_cycle_service"].is_monotonic_increasing


def test_backtest_shows_the_shortfall_when_errors_drift_upwards():
    drifting = [np.full(28, d) for d in (0.0, 0.0, 0.0, 0.0, 6.0)]
    table = make_table(drifting)
    result = service_level_backtest(table, 7, (0.95,), model="M")
    last = result[result["fold"] == 5].iloc[0]
    assert last["achieved_cycle_service"] == 0.0 and last["n_stockout_windows"] == last["n_test_windows"]
    summary = summarize_service_level_backtest(result, n_resamples=50)
    assert summary["gap"].iloc[0] < 0  # fell short of the target, and the summary says so


def test_backtest_marks_failed_test_folds_and_small_samples():
    table = iid_table(n_items=1)
    table.loc[table["fold"] == 4, "success"] = False
    table.loc[table["fold"] == 4, "predicted"] = np.nan
    result = service_level_backtest(table, 7, (0.9,), model="M")
    assert result[result["fold"] == 4]["status"].iloc[0] == "test fold has failed or missing forecasts"
    summary = summarize_service_level_backtest(result, n_resamples=50)
    assert "indicative" in summary["note"].iloc[0] and summary["n_cells"].iloc[0] == 2


def test_backtest_validates_its_inputs():
    table = iid_table(n_items=1)
    with pytest.raises(ValueError, match="service_level"):
        service_level_backtest(table, 7, (1.0,), model="M")
    with pytest.raises(ValueError, match="window_days"):
        service_level_backtest(table, 0, model="M")
    with pytest.raises(ValueError, match="No rows"):
        service_level_backtest(table, 7, model="ghost")
    two = pd.concat([table, make_table([np.ones(28)] * 5, model="N", item="I0")])
    with pytest.raises(ValueError, match="several models"):
        service_level_backtest(two, 7)


# ------------------------------------------------------------------------------------------ origin helper and text
def test_reorder_point_at_origin_uses_only_earlier_folds_and_reports_realised_demand_separately():
    table = iid_table(n_items=1, seed=5)
    a = InventoryAssumptions(lead_time_days=7, service_level=0.9)
    result, context = reorder_point_at_origin(table, model="M", item_id="I0", fold=4, assumptions=a)
    assert context["calibration_folds"] == [1, 2, 3] and result.n_error_samples == 3 * 22
    fold4 = table[table["fold"] == 4].sort_values("date")
    assert result.expected_demand == pytest.approx(fold4["predicted"].iloc[:7].sum())
    assert context["realised_demand_over_protection"] == pytest.approx(fold4["actual"].iloc[:7].sum())
    assert context["forecast_start"] == fold4["date"].iloc[0].date().isoformat()
    changed = table.copy()
    changed.loc[changed["fold"] == 4, "actual"] += 99.0
    again, _ = reorder_point_at_origin(changed, model="M", item_id="I0", fold=4, assumptions=a)
    assert again.reorder_point == result.reorder_point  # the decision never sees the protected fold's actuals


def test_reorder_point_at_origin_refuses_the_first_fold_unknown_folds_and_failed_forecasts():
    table = iid_table(n_items=1)
    a = InventoryAssumptions(lead_time_days=7)
    with pytest.raises(ValueError, match="no earlier fold"):
        reorder_point_at_origin(table, model="M", item_id="I0", fold=1, assumptions=a)
    with pytest.raises(ValueError, match="not in the table"):
        reorder_point_at_origin(table, model="M", item_id="I0", fold=9, assumptions=a)
    broken = table.copy()
    broken.loc[broken["fold"] == 3, "success"] = False
    broken.loc[broken["fold"] == 3, "predicted"] = np.nan
    with pytest.raises(ValueError, match="failed"):
        reorder_point_at_origin(broken, model="M", item_id="I0", fold=3, assumptions=a)


def test_explain_result_states_assumptions_service_level_meaning_and_warnings():
    a = InventoryAssumptions(lead_time_days=5, review_period_days=2, service_level=0.9, on_hand=3.0)
    result = compute_reorder_point([2.0] * 28, [1.0, 2.0, 3.0], a)
    text = explain_result(result, a, item_label="FOODS_3_042", unit="units")
    for phrase in (
        "FOODS_3_042", "7 day", "lead time 5", "review period 2", "Safety stock", "Reorder point", "order-up-to",
        "Inventory position", "cycle service level", "not the fill rate", "illustrative", "no savings", "Warnings",
        "forecast is an estimate",
    ):
        assert phrase.lower() in text.lower(), phrase
    assert "saving" not in text.lower().replace("no savings", "")


def test_explain_result_when_no_order_is_needed():
    a = InventoryAssumptions(lead_time_days=2, on_hand=500.0)
    result = compute_reorder_point([1.0] * 5, [1.0] * 30, a)
    assert "no order is needed" in explain_result(result, a)
