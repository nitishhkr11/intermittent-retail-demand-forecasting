import numpy as np
import pandas as pd

from src.evaluation.expanded_inventory_calibration import METHODS, compare_calibrations


def _forecasts():
    rows = []
    for item in ("a", "b", "c"):
        for fold in range(1, 5):
            for day in range(28):
                predicted = 5.0 + (item == "b") * 3
                actual = predicted + (2.0 if (day + fold) % 5 == 0 else -0.5)
                rows.append({"model": "M", "item_id": item, "fold": fold,
                             "date": pd.Timestamp("2020-01-01") + pd.Timedelta(days=fold * 35 + day),
                             "actual": actual, "predicted": predicted, "success": True,
                             "fold_local_class": "Smooth" if item == "a" else "Erratic"})
    return pd.DataFrame(rows)


def test_calibration_methods_use_only_prior_folds_and_common_eval_cells():
    result = compare_calibrations(_forecasts())
    assert set(result["method"]) == set(METHODS)
    assert set(result["fold"]) == {1, 2, 3, 4}
    assert result.loc[result.fold < 3, "status"].str.startswith("fewer than").all()
    valid = result[result.status == "ok"]
    assert valid.groupby(["item_id", "fold"]).method.nunique().eq(len(METHODS)).all()
    assert (valid.calibration_folds <= valid.fold - 1).all()
    assert np.isfinite(valid["achieved_cycle_service"]).all()


def test_calibration_fails_do_not_turn_into_imputed_service():
    table = _forecasts()
    table.loc[table.fold == 3, "success"] = False
    result = compare_calibrations(table)
    failed = result[(result.fold == 3) & (result.method != "class_normalized")]
    assert failed.status.eq("test fold failed or missing").all()
    assert result.loc[result.fold == 3, "achieved_cycle_service"].isna().all()


def test_class_normalization_never_uses_the_evaluation_folds_actual_scale():
    original = _forecasts()
    changed = original.copy()
    changed.loc[changed.fold == 4, "actual"] *= 1000
    a = compare_calibrations(original)
    b = compare_calibrations(changed)
    a = a[(a.method == "class_normalized") & (a.fold == 4)].sort_values("item_id")
    b = b[(b.method == "class_normalized") & (b.fold == 4)].sort_values("item_id")
    assert np.allclose(a["safety_stock"], b["safety_stock"])
