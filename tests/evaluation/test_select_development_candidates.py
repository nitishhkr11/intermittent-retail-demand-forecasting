from src.evaluation.select_development_candidates import _aggregate_candidates, _class_candidates


def test_aggregate_selection_compares_complete_shared_fold_candidates():
    table = _aggregate_candidates()
    assert not table.empty
    assert (table["folds"] == 16).all()
    assert (table["failures"] == 0).all()
    assert table.iloc[0]["model"] == "LSTM(calendar)"


def test_every_required_demand_class_has_a_full_panel_selected_candidate():
    selected, details = _class_candidates()
    assert set(selected["selection_group"]) == {"Smooth", "Erratic", "Intermittent", "Lumpy"}
    assert all(table.iloc[0]["eligible"] for table in details.values())
    assert all(not table[table.model.str.startswith("XGBoost(per-item")]["eligible"].any() for table in details.values())
