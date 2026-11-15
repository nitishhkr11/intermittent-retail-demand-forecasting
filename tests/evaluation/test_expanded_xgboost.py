"""Fast design and sampling tests for the development-only XGBoost ablation study."""

from __future__ import annotations

import pandas as pd

from src.analysis.demand_classification import STANDARD_CATEGORIES
from src.evaluation import expanded_xgboost as xgb_study


def test_ablation_plan_covers_the_required_pooled_feature_sets_and_poisson_objective():
    pooled = [row for row in xgb_study.FEATURE_VARIANTS if row["scope"] == "pooled"]
    assert [row["key"] for row in pooled] == [
        "lags_only", "lags_rolling", "lags_calendar", "lags_calendar_events",
        "lags_calendar_events_poisson",
    ]
    assert [row["objective"] for row in pooled].count("count:poisson") == 1
    assert xgb_study.FEATURE_VARIANTS[-1]["scope"] == "per_item"


def test_per_item_sample_is_deterministic_and_uses_only_first_origin_training_prefix(monkeypatch):
    index = pd.date_range("2020-01-01", periods=100, freq="D")
    class_by_item = {f"{label}_{i}": label for label in STANDARD_CATEGORIES for i in range(4)}
    wide = pd.DataFrame({item: range(100) for item in class_by_item}, index=index)
    seen_lengths = []

    def fake_classify(series):
        seen_lengths.append(len(series))
        return {"demand_class": class_by_item[series.name]}

    monkeypatch.setattr(xgb_study, "classify_series", fake_classify)
    folds = [{"fold": 1, "test_start_pos": 80, "train_end": index[79].date().isoformat()}]
    sample_a = xgb_study.stratified_item_sample(wide, folds, per_class=2, seed=7)
    sample_b = xgb_study.stratified_item_sample(wide, folds, per_class=2, seed=7)
    pd.testing.assert_frame_equal(sample_a, sample_b)
    assert len(sample_a) == 8
    assert sample_a.groupby("sample_class_at_first_origin").size().to_dict() == {label: 2 for label in STANDARD_CATEGORIES}
    assert set(seen_lengths) == {80}
    assert set(sample_a["sample_origin"]) == {index[79].date().isoformat()}
