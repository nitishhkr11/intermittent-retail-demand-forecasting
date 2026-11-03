"""Unit tests for src/models/xgboost_model.py (Phase 3 Step 1).

Small, fully deterministic synthetic series only -- no real project data
and no download of the M5 dataset (the notebook-execution test covers the
real 20-item run). Values come from fixed repeating blocks, never from
`np.random`, so nothing here can flake on a random seed.

Most behavioural tests inject a recording stub in place of `XGBRegressor`
(the `regressor_factory` argument). The stub shows exactly which feature
matrix the model was fit on and which feature matrix it was asked to
predict on at every recursive step, so leakage and recursion rules are
asserted on the real inputs rather than inferred from outputs. A few
tests use the real `XGBRegressor` with a handful of trees to cover the
actual xgboost integration (categorical dtypes, determinism, importances).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.analysis.demand_classification import classify_series
from src.evaluation.backtesting import run_backtest, summarize_backtest
from src.evaluation.metrics import compute_metrics
from src.models.base import BaseForecaster
from src.models.xgboost_model import (
    CALENDAR_BASE_COLUMNS,
    DEMAND_CLASS_LEVELS,
    EVENT_FLAG_COLUMNS,
    PooledXGBoostForecaster,
    XGBoostConfig,
    XGBoostForecaster,
    build_event_calendar,
    build_training_frame,
    calendar_features,
    fold_local_demand_classes,
    lag_roll_feature_matrix,
    lag_roll_features_for_next_step,
    load_event_calendar,
)
from src.utils.config import load_config

# Small and quick: a few shallow trees, calendar events off unless a test turns them on.
FAST = dict(n_estimators=8, max_depth=3, use_event_calendar=False)
CFG = XGBoostConfig(**FAST)
LAGS = (1, 7, 14, 28)
WINDOWS = (7, 14, 28)

BLOCK_A = [0, 2, 0, 0, 5, 1, 0, 3, 0, 0, 4, 0, 1]          # intermittent-looking, 13-day period
BLOCK_B = [3, 4, 2, 5, 3, 4, 6, 3, 2, 4]                   # smooth-looking, 10-day period


def _pattern(n: int, block) -> np.ndarray:
    reps = int(np.ceil(n / len(block)))
    return np.asarray((list(block) * reps)[:n], dtype=float)


def _series(values, start="2020-01-01", name=None) -> pd.Series:
    idx = pd.date_range(start, periods=len(values), freq="D")
    return pd.Series(np.asarray(values, dtype=float), index=idx, name=name)


class _StubRegressor:
    """Stands in for XGBRegressor and records every matrix it is fit or asked to predict on."""

    def __init__(self, predict_fn=None):
        self.predict_fn = predict_fn or (lambda X: np.full(len(X), 2.0))
        self.fit_X = None
        self.fit_y = None
        self.predict_X = []

    def fit(self, X, y):
        self.fit_X = X.copy()
        self.fit_y = np.asarray(y, dtype=float).copy()
        return self

    def predict(self, X):
        self.predict_X.append(X.copy())
        return np.asarray(self.predict_fn(X), dtype=float)


def _event_calendar(index: pd.DatetimeIndex, **flags_on) -> pd.DataFrame:
    """All-zero flag table over `index`; `flags_on={'is_christmas': [timestamp, ...]}` switches days on."""
    cal = pd.DataFrame(0, index=index, columns=list(EVENT_FLAG_COLUMNS))
    for column, days in flags_on.items():
        cal.loc[pd.DatetimeIndex(days), column] = 1
    return cal


# ---- configuration ----------------------------------------------------------------

def test_default_config_matches_the_documented_values():
    cfg = XGBoostConfig()
    assert cfg.lags == (1, 7, 14, 28)
    assert cfg.rolling_mean_windows == (7, 14, 28)
    assert cfg.max_lookback == 28
    assert cfg.objective == "reg:squarederror"
    assert cfg.random_state == 42
    assert cfg.n_jobs == 1


def test_config_yaml_xgboost_section_equals_the_code_defaults():
    # Guards drift between config.yaml and the dataclass defaults the docs describe.
    assert XGBoostConfig.from_config(load_config()) == XGBoostConfig()


def test_from_config_reads_the_xgboost_section():
    cfg = XGBoostConfig.from_config(
        {
            "xgboost": {
                "lags": [1, 7],
                "rolling_mean_windows": [7],
                "use_event_calendar": False,
                "hyperparameters": {"n_estimators": 11, "max_depth": 2, "random_state": 7},
            }
        }
    )
    assert cfg.lags == (1, 7) and cfg.rolling_mean_windows == (7,)
    assert cfg.use_event_calendar is False
    assert (cfg.n_estimators, cfg.max_depth, cfg.random_state) == (11, 2, 7)
    assert cfg.max_lookback == 7


def test_from_config_rejects_unknown_keys_and_a_missing_section():
    with pytest.raises(KeyError):
        XGBoostConfig.from_config({})
    with pytest.raises(ValueError):
        XGBoostConfig.from_config({"xgboost": {"hyperparameters": {"n_estimator": 5}}})  # typo
    with pytest.raises(ValueError):
        XGBoostConfig.from_config({"xgboost": {"lag": [1]}})  # typo in a top-level key


@pytest.mark.parametrize(
    "kwargs",
    [
        {"lags": ()},
        {"lags": (1, 1)},
        {"lags": (0, 7)},
        {"rolling_mean_windows": ()},
        {"rolling_mean_windows": (-7,)},
        {"n_estimators": 0},
        {"learning_rate": 0.0},
        {"max_depth": 0},
        {"n_jobs": 0},
    ],
)
def test_config_rejects_invalid_values(kwargs):
    with pytest.raises(ValueError):
        XGBoostConfig(**kwargs)


def test_feature_names_pooled_adds_item_and_class_only_when_pooled():
    cfg = XGBoostConfig()
    assert cfg.feature_names(pooled=False) == cfg.numeric_feature_names()
    assert cfg.feature_names(pooled=True) == cfg.numeric_feature_names() + ["item_id", "demand_class"]
    assert len(cfg.feature_names(pooled=True)) == 18
    assert XGBoostConfig(use_event_calendar=False).feature_names(pooled=False) == (
        cfg.lag_roll_names + list(CALENDAR_BASE_COLUMNS)
    )


def test_regressor_params_pin_the_seed_and_single_thread():
    from src.models.xgboost_model import _default_regressor

    reg = _default_regressor(XGBoostConfig())
    params = reg.get_params()
    assert params["random_state"] == 42
    assert params["n_jobs"] == 1
    assert params["enable_categorical"] is True
    assert params["objective"] == "reg:squarederror"


# ---- lag / rolling features: hand-computed correctness ----------------------------

def test_lag_and_rolling_features_match_a_hand_computation():
    y = np.arange(1.0, 41.0)  # y[t] = t + 1
    features = lag_roll_feature_matrix(y, LAGS, WINDOWS)
    assert features.shape == (40, 7)
    t = 30
    # Independent computation straight from the definitions.
    expected = [y[t - 1], y[t - 7], y[t - 14], y[t - 28]]
    expected += [y[t - 7 : t].mean(), y[t - 14 : t].mean(), y[t - 28 : t].mean()]
    np.testing.assert_allclose(features[t], expected)
    # Spot values: lag_7 of y[30] = 31 is y[23] = 24; roll_7 = mean(24..30) = 27.
    assert features[t, 1] == 24.0
    assert features[t, 4] == 27.0


def test_rolling_window_ends_the_day_before_the_target():
    y = np.array([0.0] * 10 + [100.0] + [0.0] * 10)  # one spike at t = 10
    features = lag_roll_feature_matrix(y, (1,), (7,))
    spike_day_roll7 = features[10, 1]
    assert spike_day_roll7 == 0.0, "the spike day's own sales leaked into its rolling mean"
    assert features[11, 0] == 100.0          # lag_1 of the next day does see it
    assert features[11, 1] == pytest.approx(100.0 / 7)


def test_features_without_full_history_are_nan_not_imputed():
    y = np.arange(1.0, 41.0)
    features = lag_roll_feature_matrix(y, LAGS, WINDOWS)
    for col, need in enumerate(list(LAGS) + list(WINDOWS)):
        assert np.isnan(features[:need, col]).all(), f"column {col} was filled before {need} days of history"
        assert not np.isnan(features[need:, col]).any()


def test_no_target_leakage_changing_a_days_sales_never_changes_that_rows_features():
    y = _pattern(80, BLOCK_A)
    base = lag_roll_feature_matrix(y, LAGS, WINDOWS)
    for t in range(28, 70, 3):
        changed = y.copy()
        changed[t] = 1_000_000.0
        features = lag_roll_feature_matrix(changed, LAGS, WINDOWS)
        np.testing.assert_array_equal(features[t], base[t])  # own row: unchanged
        # control: the change does reach later rows through lag_1 (so the check can fail)
        assert features[t + 1, 0] == 1_000_000.0


def test_lag_roll_features_are_additive_for_short_series():
    # Series no longer than the lag: all-NaN column, no exception.
    features = lag_roll_feature_matrix(np.arange(5.0), (7,), (14,))
    assert features.shape == (5, 2) and np.isnan(features).all()


def test_next_step_features_match_the_training_matrix_wherever_both_exist():
    y = np.arange(1.0, 70.0) * 1.3 + (np.arange(69) % 5)
    matrix = lag_roll_feature_matrix(y, LAGS, WINDOWS)
    for t in range(28, 69):
        nxt = lag_roll_features_for_next_step(y[None, :t], LAGS, WINDOWS)[0]
        np.testing.assert_allclose(matrix[t], nxt, rtol=1e-12)


def test_next_step_features_validate_their_input():
    with pytest.raises(ValueError):
        lag_roll_features_for_next_step(np.arange(40.0), LAGS, WINDOWS)  # not 2-D
    with pytest.raises(ValueError):
        lag_roll_features_for_next_step(np.ones((2, 10)), LAGS, WINDOWS)  # shorter than the longest lag


# ---- training frame: warm-up rows are dropped explicitly ---------------------------

def test_training_frame_drops_exactly_the_warmup_rows_and_has_no_nan():
    histories = {
        "A": _series(_pattern(60, BLOCK_A), name="A"),
        "B": _series(_pattern(60, BLOCK_B), name="B"),
    }
    frame, n_dropped = build_training_frame(histories, CFG)
    assert n_dropped == 2 * 28
    assert len(frame) == 2 * (60 - 28)
    assert not frame[CFG.numeric_feature_names()].isna().any().any()
    first = frame[frame["item_id"] == "A"].iloc[0]
    assert first["date"] == pd.Timestamp("2020-01-01") + pd.Timedelta(days=28)
    values = histories["A"].to_numpy()
    assert first["y"] == values[28]
    assert first["lag_28"] == values[0] and first["lag_1"] == values[27]
    assert first["roll_mean_28"] == pytest.approx(values[0:28].mean())


def test_training_frame_requires_more_history_than_the_longest_window():
    with pytest.raises(ValueError):
        build_training_frame({"A": _series(_pattern(28, BLOCK_A), name="A")}, CFG)


# ---- calendar features -------------------------------------------------------------

def test_calendar_features_are_functions_of_the_date_alone():
    index = pd.DatetimeIndex(["2020-02-29", "2020-12-25", "2021-01-03"])
    out = calendar_features(index, None, use_events=False)
    assert out["dayofweek"].tolist() == [5, 4, 6]       # Sat, Fri, Sun (Monday = 0)
    assert out["month"].tolist() == [2, 12, 1]
    assert out["dayofmonth"].tolist() == [29, 25, 3]
    assert list(out.columns) == list(CALENDAR_BASE_COLUMNS)


def test_build_event_calendar_flags_event_types_snap_and_the_christmas_closure():
    dates = pd.to_datetime(["2020-02-02", "2020-02-14", "2020-04-12", "2020-12-25", "2020-12-26"])
    rows = pd.DataFrame(
        {
            "date": dates,
            "event_name_1": ["SuperBowl", "ValentinesDay", "Easter", "Christmas", None],
            "event_type_1": ["Sporting", "Cultural", "Religious", "National", None],
            "event_name_2": [None, None, "OrthodoxEaster", None, None],
            "event_type_2": [None, None, "Cultural", None, None],
            "snap_CA": [0, 1, 0, 0, 1],
        }
    )
    # The panel repeats each date once per item; identical repeats must collapse to one row.
    raw = pd.concat([rows, rows, rows], ignore_index=True)
    cal = build_event_calendar(raw)
    assert list(cal.columns) == list(EVENT_FLAG_COLUMNS)
    assert cal.index.tolist() == list(dates)
    assert cal["event_sporting"].tolist() == [1, 0, 0, 0, 0]
    assert cal["event_cultural"].tolist() == [0, 1, 1, 0, 0]   # Easter's second event is Cultural
    assert cal["event_religious"].tolist() == [0, 0, 1, 0, 0]
    assert cal["event_national"].tolist() == [0, 0, 0, 1, 0]
    assert cal["is_christmas"].tolist() == [0, 0, 0, 1, 0]
    assert cal["snap_ca"].tolist() == [0, 1, 0, 0, 1]
    assert set(cal.dtypes.astype(str)) == {"int64"}


def test_build_event_calendar_rejects_inconsistent_and_incomplete_input():
    base = pd.DataFrame(
        {
            "date": pd.to_datetime(["2020-01-01", "2020-01-01"]),
            "event_name_1": [None, "NewYear"],
            "event_type_1": [None, "National"],
            "event_name_2": [None, None],
            "event_type_2": [None, None],
            "snap_CA": [0, 0],
        }
    )
    with pytest.raises(ValueError, match="inconsistent"):
        build_event_calendar(base)  # same date, two different event rows
    with pytest.raises(ValueError, match="missing column"):
        build_event_calendar(base.drop(columns=["snap_CA"]))


def test_calendar_features_do_not_zero_fill_missing_dates():
    index = pd.date_range("2020-03-01", periods=5, freq="D")
    cal = _event_calendar(index[:3])
    with pytest.raises(ValueError, match="does not cover"):
        calendar_features(index, cal, use_events=True)
    with pytest.raises(ValueError, match="event calendar is required"):
        calendar_features(index, None, use_events=True)


def test_load_event_calendar_reads_only_calendar_columns_and_filters_dates_at_read_time(tmp_path, monkeypatch):
    dates = pd.date_range("2020-01-01", periods=90, freq="D")
    panel = pd.DataFrame(
        {
            "item_id": "X",
            "date": dates,
            "sales": 7_777,  # sentinel: must never be loaded
            "event_name_1": None,
            "event_type_1": None,
            "event_name_2": None,
            "event_type_2": None,
            "snap_CA": 0,
        }
    )
    path = tmp_path / "panel.parquet"
    panel.to_parquet(path)

    seen = {}
    real_read = pd.read_parquet

    def spy(*args, **kwargs):
        seen.update(kwargs)
        return real_read(*args, **kwargs)

    monkeypatch.setattr(pd, "read_parquet", spy)
    cal = load_event_calendar(path, last_date="2020-03-15")

    assert "sales" not in seen["columns"] and "item_id" not in seen["columns"]
    assert seen["filters"] == [("date", "<=", pd.Timestamp("2020-03-15"))]
    assert cal.index.max() == pd.Timestamp("2020-03-15")
    assert len(cal) == 75
    assert list(cal.columns) == list(EVENT_FLAG_COLUMNS)

    with pytest.raises(FileNotFoundError):
        load_event_calendar(tmp_path / "missing.parquet", "2020-03-15")


# ---- fit / predict contract --------------------------------------------------------

def test_xgboost_forecaster_is_a_baseforecaster_with_the_expected_names():
    assert issubclass(XGBoostForecaster, BaseForecaster)
    assert XGBoostForecaster.name == "XGBoost(per-item)"
    assert PooledXGBoostForecaster(CFG).name == "XGBoost(pooled)"
    poisson = XGBoostConfig(objective="count:poisson", **{k: v for k, v in FAST.items()})
    assert PooledXGBoostForecaster(poisson).name == "XGBoost(pooled,poisson)"
    assert PooledXGBoostForecaster(CFG, name="custom").name == "custom"


def test_predict_before_fit_raises():
    with pytest.raises(RuntimeError, match="before fit"):
        XGBoostForecaster(CFG).predict(7)
    with pytest.raises(RuntimeError, match="before fit"):
        PooledXGBoostForecaster(CFG).predict(7)
    with pytest.raises(RuntimeError):
        PooledXGBoostForecaster(CFG).forecast_dates(7)


def test_horizon_must_be_positive():
    model = XGBoostForecaster(CFG).fit(_series(_pattern(80, BLOCK_A), name="A"))
    for bad in (0, -1):
        with pytest.raises(ValueError, match="horizon"):
            model.predict(bad)


def test_output_dates_and_length_follow_the_training_end_date():
    y = _series(_pattern(100, BLOCK_A), start="2020-01-01", name="A")  # ends 2020-04-09
    pooled = PooledXGBoostForecaster(CFG).fit({"A": y, "B": _series(_pattern(100, BLOCK_B), name="B")})
    dates = pooled.forecast_dates(28)
    assert dates[0] == pd.Timestamp("2020-04-10") and dates[-1] == pd.Timestamp("2020-05-07")
    assert len(dates) == 28 and dates.is_monotonic_increasing
    out = pooled.predict(28)
    assert set(out) == {"A", "B"}
    assert all(v.shape == (28,) for v in out.values())
    assert pooled.fit_diagnostics()["train_end"] == "2020-04-09"

    single = XGBoostForecaster(CFG).fit(y)
    assert single.predict(5).shape == (5,)


def test_predictions_are_finite_and_non_negative_with_the_real_regressor():
    pooled = PooledXGBoostForecaster(CFG).fit(
        {"A": _series(_pattern(120, BLOCK_A), name="A"), "B": _series(_pattern(120, BLOCK_B), name="B")}
    )
    for forecast in pooled.predict(28).values():
        assert np.isfinite(forecast).all()
        assert (forecast >= 0).all()


def test_fixed_seed_gives_identical_forecasts_on_repeat_fits():
    histories = {"A": _series(_pattern(120, BLOCK_A), name="A"), "B": _series(_pattern(120, BLOCK_B), name="B")}
    cfg = XGBoostConfig(n_estimators=20, max_depth=4, use_event_calendar=False)  # subsample < 1: seed matters
    first = PooledXGBoostForecaster(cfg).fit(histories).predict(28)
    second = PooledXGBoostForecaster(cfg).fit(histories).predict(28)
    for item in histories:
        np.testing.assert_array_equal(first[item], second[item])


@pytest.mark.parametrize(
    "bad_history",
    [
        pd.Series([], dtype=float, index=pd.DatetimeIndex([])),
        _series([1.0] * 40).where(lambda s: s.index != pd.Timestamp("2020-01-10")),          # NaN
        _series([1.0] * 39 + [-1.0]),                                                         # negative
        pd.Series(np.ones(40)),                                                               # RangeIndex
        _series(np.ones(40)).drop(pd.Timestamp("2020-01-10")),                                # date gap
        _series(np.ones(40)).iloc[::-1],                                                      # descending
        pd.concat([_series(np.ones(35)), _series(np.ones(5), start="2020-02-04")]),           # duplicate dates
        _series(np.ones(28)),                                                                 # no row left after warm-up
    ],
    ids=["empty", "nan", "negative", "no-date-index", "gap", "descending", "duplicate-dates", "too-short"],
)
def test_fit_rejects_invalid_histories(bad_history):
    with pytest.raises(ValueError):
        XGBoostForecaster(CFG).fit(bad_history)


def test_fit_rejects_a_non_series_and_an_empty_mapping():
    with pytest.raises(TypeError):
        PooledXGBoostForecaster(CFG).fit({"A": [1.0, 2.0, 3.0]})
    with pytest.raises(ValueError):
        PooledXGBoostForecaster(CFG).fit({})


def test_pooled_fit_requires_one_shared_forecast_origin():
    a = _series(_pattern(80, BLOCK_A), name="A")
    b = _series(_pattern(79, BLOCK_B), name="B")
    with pytest.raises(ValueError, match="same date"):
        PooledXGBoostForecaster(CFG).fit({"A": a, "B": b})


# ---- what the model is fit on and asked to predict on ------------------------------

def test_fit_matrix_has_the_documented_columns_dtypes_and_no_nan():
    stub = _StubRegressor()
    histories = {"B": _series(_pattern(70, BLOCK_B), name="B"), "A": _series(_pattern(70, BLOCK_A), name="A")}
    PooledXGBoostForecaster(CFG, regressor_factory=lambda: stub).fit(histories)

    X = stub.fit_X
    assert list(X.columns) == CFG.feature_names(pooled=True)
    assert len(X) == 2 * (70 - 28) and len(stub.fit_y) == len(X)
    assert not X.isna().any().any()
    assert isinstance(X["item_id"].dtype, pd.CategoricalDtype)
    assert isinstance(X["demand_class"].dtype, pd.CategoricalDtype)
    assert list(X["item_id"].cat.categories) == ["A", "B"]                 # sorted, same order at predict time
    assert list(X["demand_class"].cat.categories) == list(DEMAND_CLASS_LEVELS)
    # item-major rows: item A first, target values are A's sales after the warm-up rows
    np.testing.assert_array_equal(stub.fit_y[: 70 - 28], _pattern(70, BLOCK_A)[28:])
    np.testing.assert_array_equal(stub.fit_y[70 - 28 :], _pattern(70, BLOCK_B)[28:])
    # every row's identity, class and lag features belong to the item whose target it carries
    labels = X["item_id"].astype(str).to_numpy()
    assert (labels[: 70 - 28] == "A").all() and (labels[70 - 28 :] == "B").all()
    assert X["lag_1"].iloc[0] == _pattern(70, BLOCK_A)[27]
    assert X["lag_1"].iloc[70 - 28] == _pattern(70, BLOCK_B)[27]
    class_by_label = {
        item: classify_series(pd.Series(_pattern(70, block)))["demand_class"]
        for item, block in (("A", BLOCK_A), ("B", BLOCK_B))
    }
    assert X["demand_class"].astype(str).to_numpy().tolist() == (
        [class_by_label["A"]] * (70 - 28) + [class_by_label["B"]] * (70 - 28)
    )


def test_per_item_fit_matrix_has_no_item_or_class_columns():
    stub = _StubRegressor()
    XGBoostForecaster(CFG, regressor_factory=lambda: stub).fit(_series(_pattern(70, BLOCK_A), name="A"))
    assert list(stub.fit_X.columns) == CFG.feature_names(pooled=False)


def test_recursive_forecast_feeds_back_its_own_predictions():
    y = _series(np.arange(1.0, 61.0), name="A")  # strictly increasing actuals, last value 60
    stub = _StubRegressor(predict_fn=lambda X: X["lag_1"].to_numpy() + 100.0)
    preds = XGBoostForecaster(CFG, regressor_factory=lambda: stub).fit(y).predict(5)
    # step 1 sees the last actual (60); every later step sees the PREVIOUS PREDICTION.
    np.testing.assert_allclose(preds, [160.0, 260.0, 360.0, 460.0, 560.0])
    assert [call["lag_1"].iloc[0] for call in stub.predict_X] == [60.0, 160.0, 260.0, 360.0, 460.0]


def test_recursion_state_mixes_history_and_predictions_exactly_as_defined():
    y_values = np.arange(1.0, 61.0)
    stub = _StubRegressor(predict_fn=lambda X: X["lag_1"].to_numpy() + 100.0)
    h = 10
    preds = XGBoostForecaster(CFG, regressor_factory=lambda: stub).fit(_series(y_values, name="A")).predict(h)

    seen = np.concatenate([y_values, preds])  # what a correct recursion has available, in time order
    for step, X in enumerate(stub.predict_X):
        t = len(y_values) + step  # position of the day being forecast
        assert X["lag_1"].iloc[0] == seen[t - 1]
        assert X["lag_7"].iloc[0] == seen[t - 7]       # actual for steps 0-6, own prediction from step 7
        assert X["lag_14"].iloc[0] == seen[t - 14]
        assert X["lag_28"].iloc[0] == seen[t - 28]
        assert X["roll_mean_7"].iloc[0] == pytest.approx(seen[t - 7 : t].mean())
        assert X["roll_mean_28"].iloc[0] == pytest.approx(seen[t - 28 : t].mean())
    # the first prediction is used as a lag_7 value exactly 7 steps later
    assert stub.predict_X[7]["lag_7"].iloc[0] == preds[0]


def test_each_items_forecast_comes_from_that_items_own_rows():
    # Feature rows are keyed by item. The stub's output depends on the item's own lag_1 and its
    # identity code, so any misalignment between items, features and returned arrays shows up.
    stub = _StubRegressor(
        predict_fn=lambda X: X["lag_1"].to_numpy() + 1000.0 * (X["item_id"].cat.codes.to_numpy() + 1)
    )
    histories = {"B": _series(np.arange(1.0, 61.0), name="B"), "A": _series(np.arange(101.0, 161.0), name="A")}
    out = PooledXGBoostForecaster(CFG, regressor_factory=lambda: stub).fit(histories).predict(3)
    # A (code 0): 160+1000, then each step feeds back its own previous prediction.
    np.testing.assert_allclose(out["A"], [1160.0, 2160.0, 3160.0])
    # B (code 1): 60+2000, 2060+2000, 4060+2000.
    np.testing.assert_allclose(out["B"], [2060.0, 4060.0, 6060.0])


def test_negative_raw_predictions_are_clipped_to_zero_and_the_clipped_value_is_fed_back():
    stub = _StubRegressor(predict_fn=lambda X: np.full(len(X), -3.0))
    model = XGBoostForecaster(CFG, regressor_factory=lambda: stub).fit(_series(_pattern(80, BLOCK_B), name="A"))
    preds = model.predict(5)
    np.testing.assert_array_equal(preds, np.zeros(5))
    assert stub.predict_X[1]["lag_1"].iloc[0] == 0.0, "the unclipped -3 must never become a lag"
    diag = model.fit_diagnostics()
    assert diag["n_predictions"] == 5
    assert diag["n_predictions_clipped_at_zero"] == 5
    assert diag["min_raw_prediction"] == -3.0


def test_non_negative_predictions_pass_through_unchanged():
    stub = _StubRegressor(predict_fn=lambda X: np.full(len(X), 1.5))
    model = XGBoostForecaster(CFG, regressor_factory=lambda: stub).fit(_series(_pattern(80, BLOCK_B), name="A"))
    np.testing.assert_array_equal(model.predict(4), np.full(4, 1.5))
    assert model.fit_diagnostics()["n_predictions_clipped_at_zero"] == 0


@pytest.mark.parametrize("bad_value", [np.nan, np.inf, -np.inf])
def test_non_finite_predictions_raise_instead_of_being_clipped_or_replaced(bad_value):
    stub = _StubRegressor(predict_fn=lambda X: np.full(len(X), bad_value))
    model = XGBoostForecaster(CFG, regressor_factory=lambda: stub).fit(_series(_pattern(80, BLOCK_B), name="A"))
    with pytest.raises(ValueError, match="non-finite"):
        model.predict(3)


def test_a_wrong_shaped_regressor_output_raises():
    stub = _StubRegressor(predict_fn=lambda X: np.zeros(len(X) + 1))
    model = XGBoostForecaster(CFG, regressor_factory=lambda: stub).fit(_series(_pattern(80, BLOCK_B), name="A"))
    with pytest.raises(ValueError, match="shape"):
        model.predict(3)


def test_event_flags_enter_the_features_on_the_right_forecast_dates():
    y = _series(_pattern(80, BLOCK_B), start="2020-01-01", name="A")          # ends 2020-03-20
    index = pd.date_range("2020-01-01", periods=80 + 10, freq="D")
    closure = pd.Timestamp("2020-03-24")                                       # forecast step 4 (0-based 3)
    cal = _event_calendar(index, is_christmas=[closure], snap_ca=[pd.Timestamp("2020-03-21")])
    cfg = XGBoostConfig(n_estimators=8, max_depth=3, use_event_calendar=True)
    stub = _StubRegressor()
    XGBoostForecaster(cfg, event_calendar=cal, regressor_factory=lambda: stub).fit(y).predict(5)

    assert list(stub.fit_X.columns) == cfg.feature_names(pooled=False)
    assert [c["is_christmas"].iloc[0] for c in stub.predict_X] == [0, 0, 0, 1, 0]
    assert [c["snap_ca"].iloc[0] for c in stub.predict_X] == [1, 0, 0, 0, 0]
    assert [c["dayofweek"].iloc[0] for c in stub.predict_X] == [
        d.dayofweek for d in pd.date_range("2020-03-21", periods=5, freq="D")
    ]


def test_an_event_calendar_that_misses_forecast_dates_fails_at_predict_not_silently():
    y = _series(_pattern(80, BLOCK_B), name="A")                               # ends 2020-03-20
    cal = _event_calendar(pd.date_range("2020-01-01", periods=82, freq="D"))   # covers only 2 forecast days
    model = XGBoostForecaster(
        XGBoostConfig(n_estimators=8, max_depth=3), event_calendar=cal
    ).fit(y)
    with pytest.raises(ValueError, match="does not cover"):
        model.predict(5)


def test_an_event_calendar_that_misses_training_dates_fails_at_fit():
    y = _series(_pattern(80, BLOCK_B), name="A")
    cal = _event_calendar(pd.date_range("2020-02-01", periods=60, freq="D"))
    with pytest.raises(ValueError, match="does not cover"):
        XGBoostForecaster(XGBoostConfig(n_estimators=8, max_depth=3), event_calendar=cal).fit(y)


# ---- demand class feature: computed from the training window only -------------------

def test_demand_class_feature_comes_from_the_training_window_only():
    smooth_then_sparse = np.concatenate([np.full(100, 5.0) + (np.arange(100) % 3), np.zeros(98), [9.0, 9.0]])
    full = _series(smooth_then_sparse, name="A")
    prefix = full.iloc[:100]
    assert classify_series(pd.Series(prefix.to_numpy()))["demand_class"] == "Smooth"
    assert classify_series(pd.Series(full.to_numpy()))["demand_class"] != "Smooth"  # later data would change the label

    pooled = PooledXGBoostForecaster(CFG).fit({"A": prefix})
    table = pooled.item_class_table()
    assert table.loc[0, "demand_class"] == "Smooth"
    assert table.loc[0, "n_train_obs"] == 100
    pd.testing.assert_frame_equal(table, fold_local_demand_classes({"A": prefix}), check_dtype=False)


def test_fitted_class_values_are_what_the_model_receives():
    stub = _StubRegressor()
    histories = {"A": _series(np.full(60, 4.0), name="A"), "B": _series(_pattern(60, BLOCK_A), name="B")}
    PooledXGBoostForecaster(CFG, regressor_factory=lambda: stub).fit(histories).predict(2)
    classes = stub.predict_X[0].set_index(stub.predict_X[0]["item_id"].astype(str))["demand_class"].astype(str)
    assert classes["A"] == "Smooth"
    assert classes["B"] == classify_series(pd.Series(histories["B"].to_numpy()))["demand_class"]


# ---- integration with the existing, unmodified backtest framework --------------------

def test_per_item_model_runs_through_the_unmodified_run_backtest():
    series = _series(_pattern(150, BLOCK_A), name="A")
    results = run_backtest(series, lambda: XGBoostForecaster(CFG), horizon=10, n_folds=3)
    assert len(results) == 3
    assert all(r.success for r in results)
    assert all(r.model_name == "XGBoost(per-item)" for r in results)
    for r in results:
        assert len(r.predictions) == 10 and r.dates[0] == r.origin_date + pd.Timedelta(days=1)
        assert r.diagnostics["train_end"] == str(r.origin_date.date())   # trained on data up to the origin only
    summary = summarize_backtest(results)
    assert summary[["MAE", "RMSE"]].notna().all().all()
    # the folds are exactly the ones every other model gets
    assert results[-1].forecast_end == series.index[-1]


def test_metrics_for_xgboost_forecasts_use_the_existing_functions():
    series = _series(_pattern(150, BLOCK_A), name="A")
    result = run_backtest(series, lambda: XGBoostForecaster(CFG), horizon=10, n_folds=2)[-1]
    metrics = compute_metrics(result.actuals, result.predictions)
    assert metrics.mae == pytest.approx(np.mean(np.abs(result.actuals - result.predictions)))
    assert metrics.rmse == pytest.approx(np.sqrt(np.mean((result.actuals - result.predictions) ** 2)))
    summary = summarize_backtest([result]).iloc[0]
    assert summary["MAE"] == pytest.approx(metrics.mae)


def test_an_all_zero_test_window_gives_undefined_mape_not_a_fabricated_one():
    values = np.concatenate([_pattern(120, BLOCK_B), np.zeros(10)])
    series = _series(values, name="A")
    result = run_backtest(series, lambda: XGBoostForecaster(CFG), horizon=10, n_folds=1)[0]
    row = summarize_backtest([result]).iloc[0]
    assert pd.isna(row["MAPE"])             # existing zero-protected MAPE: undefined on an all-zero window
    assert row["mape_n_excluded"] == 10
    assert np.isfinite(row["MAE"])


def test_a_fold_too_short_to_train_is_recorded_as_failed_not_dropped_or_invented():
    series = _series(_pattern(100, BLOCK_A), name="A")
    # horizon 28, 3 folds -> fold 1 trains on only 16 days, fewer than the 28 the lag features need
    results = run_backtest(series, lambda: XGBoostForecaster(CFG), horizon=28, n_folds=3)
    assert [r.success for r in results] == [False, True, True]
    assert results[0].predictions is None and results[0].error.startswith("ValueError")
    assert len(results[0].actuals) == 28                      # actuals still reported for the failed fold
    summary = summarize_backtest(results)
    assert summary.loc[0, ["MAE", "RMSE", "MAPE"]].isna().all()


# ---- diagnostics and importances -----------------------------------------------------

def test_fit_diagnostics_report_the_training_set_and_settings():
    pooled = PooledXGBoostForecaster(CFG).fit(
        {"A": _series(_pattern(70, BLOCK_A), name="A"), "B": _series(_pattern(70, BLOCK_B), name="B")}
    )
    diag = pooled.fit_diagnostics()
    assert diag["pooled"] is True and diag["n_items"] == 2
    assert diag["n_train_rows"] == 2 * 42
    assert diag["n_warmup_rows_dropped"] == 2 * 28
    assert diag["n_features"] == 12 and diag["random_state"] == 42 and diag["n_jobs"] == 1
    assert diag["train_start"] == "2020-01-01" and diag["train_end"] == "2020-03-10"
    assert diag["xgboost_version"]


def test_feature_importance_is_indexed_by_the_model_features():
    cfg = XGBoostConfig(n_estimators=30, max_depth=3, use_event_calendar=False)
    pooled = PooledXGBoostForecaster(cfg).fit(
        {"A": _series(_pattern(120, BLOCK_A), name="A"), "B": _series(_pattern(120, BLOCK_B), name="B")}
    )
    importance = pooled.feature_importance("gain")
    assert list(importance.index) == cfg.feature_names(pooled=True)
    assert (importance >= 0).all() and importance.sum() > 0
