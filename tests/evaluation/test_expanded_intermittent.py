"""Unit checks for the full-panel seasonal intermittent-demand experiment."""

import numpy as np
import pandas as pd

from src.evaluation.expanded_intermittent import _factories, _run_item_chunk
from src.utils.config import load_config


def test_expanded_factories_include_sba_and_keep_fixed_documented_weights():
    config = load_config()
    factories = _factories(config)
    assert list(factories) == [
        "SeasonalNaive", "Croston(alpha=0.1)", "SBA(alpha=0.1)", "TSB(alpha=0.1,beta=0.1)"
    ]
    assert factories["SBA(alpha=0.1)"]().name == "SBA(alpha=0.1)"


def test_chunk_forecasts_ignore_their_own_test_window_for_fit():
    dates = pd.date_range("2020-01-01", periods=100, freq="D")
    values = np.tile(np.array([0.0, 2.0, 0.0, 0.0, 4.0]), 20)
    base = pd.DataFrame({"item-A": values}, index=dates)
    changed = base.copy()
    changed.iloc[60:88, 0] = 1000.0
    fold = {
        "fold": 1, "year": 2020, "quarter": 1, "train_end": "2020-02-29",
        "forecast_start": "2020-03-01", "forecast_end": "2020-03-28",
        "test_start_pos": 60, "test_end_pos": 88,
    }
    factories = _factories(load_config())
    a = _run_item_chunk(["item-A"], base, [fold], factories, "manifest", lag=7)
    b = _run_item_chunk(["item-A"], changed, [fold], factories, "manifest", lag=7)
    by_model_a = {row["model"]: row["predicted"] for row in a["forecasts"]}
    by_model_b = {row["model"]: row["predicted"] for row in b["forecasts"]}
    assert by_model_a == by_model_b
    assert len(a["scores"]) == 4
    assert {row["fold_local_class"] for row in a["scores"]} == {"Intermittent"}
