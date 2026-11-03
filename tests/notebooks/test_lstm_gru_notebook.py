"""Notebook 07 (LSTM / GRU on the aggregate series): delivered-file checks and saved-artifact audit.

Artifacts under `artifacts/` are git-ignored and rebuilt by running the notebook, so tests that need them are skipped (with the command
to create them) when they are absent. The test that re-executes the whole notebook trains 100 small networks (about 20 minutes on one
CPU thread) and therefore runs only when RIVERDEMAND_RUN_NEURAL_NOTEBOOK=1 is set, in addition to the `slow` marker.
"""

from __future__ import annotations

import json
import os
import re

import nbformat
import numpy as np
import pandas as pd
import pytest

from src.data.aggregate_series import load_aggregate_development_series
from src.evaluation.backtesting import generate_rolling_origins
from src.evaluation.forecast_artifacts import audit_forecast_table, forecasts_dir, metrics_by_group, read_forecast_table
from src.utils.config import load_config
from src.utils.paths import PROJECT_ROOT

NOTEBOOK_PATH = PROJECT_ROOT / "notebooks" / "07_lstm_gru.ipynb"
CONFIG = load_config()
PHASE3 = CONFIG["phase3"]
METRICS_DIR = PROJECT_ROOT / PHASE3["metrics_dir"]
FORECASTS = forecasts_dir(CONFIG) / PHASE3["neural_forecasts_filename"]
RUN_COMMAND = "python -m jupyter nbconvert --to notebook --execute --inplace notebooks/07_lstm_gru.ipynb"
VARIANTS = ["LSTM(calendar)", "GRU(calendar)", "LSTM(univariate)", "GRU(univariate)"]
FIGURES = [PHASE3["neural_training_curves_plot_filename"], PHASE3["neural_forecast_plot_filename"],
           PHASE3["neural_comparison_plot_filename"], PHASE3["neural_seed_plot_filename"]]


def _text(nb) -> str:
    parts = []
    for cell in nb.cells:
        if cell.cell_type != "code":
            continue
        for out in cell.get("outputs", []):
            if out.get("output_type") == "stream":
                parts.append(out.get("text", ""))
            elif out.get("output_type") in ("execute_result", "display_data"):
                text = out.get("data", {}).get("text/plain", "")
                parts.append("".join(text) if isinstance(text, list) else text)
    return "\n".join(parts)


def _markdown(nb) -> str:
    return "\n".join(c.source for c in nb.cells if c.cell_type == "markdown").lower()


def _need(path):
    if not path.exists():
        pytest.skip(f"{path.name} not found. Create it with: {RUN_COMMAND}")
    return path


# --------------------------------------------------------------------------------------------------------------- the notebook
def test_notebook_is_the_executed_deliverable_not_the_placeholder():
    nb = nbformat.read(NOTEBOOK_PATH, as_version=4)
    code = [c for c in nb.cells if c.cell_type == "code"]
    assert len(code) >= 12
    assert all(c.get("outputs") is not None and c.execution_count for c in code if c.source.strip()), "some code cells were never executed"
    assert "not yet implemented" not in _markdown(nb)


def test_notebook_explains_the_networks_the_inputs_and_the_limits():
    prose = _markdown(nbformat.read(NOTEBOOK_PATH, as_version=4))
    for phrase in ("lstm", "gru", "window", "28", "early stopping", "seed", "calendar", "univariate", "scaler", "holdout", "limitation", "aggregate"):
        assert phrase in prose, f"notebook 07 never mentions {phrase!r}"


def test_notebook_shows_the_leakage_determinism_and_completeness_checks():
    text = _text(nbformat.read(NOTEBOOK_PATH, as_version=4))
    assert "Perturbation test passed" in text
    assert "bit for bit" in text
    assert "Failed (model, fold) fits: 0 of 20" in text
    assert "no date in the reserved final holdout appears" in text
    assert "Recomputed neural MAE/RMSE equal the saved metrics file" in text
    for name in VARIANTS:
        assert name in text, f"{name} is missing from the executed notebook"
    assert text.count("PASS - ") >= 6 and "FAIL - " not in text


def test_notebook_says_honestly_how_many_folds_were_restored_from_checkpoints():
    text = _text(nbformat.read(NOTEBOOK_PATH, as_version=4))
    assert re.search(r"\d+ of \d+ folds restored from checkpoint", text)
    assert re.search(r"\d+ of 20 folds were restored from checkpoints", text)


def test_notebook_claims_no_universal_winner_and_never_uses_the_holdout():
    nb = nbformat.read(NOTEBOOK_PATH, as_version=4)
    combined = _markdown(nb) + "\n" + _text(nb).lower()
    assert "no significance test" in combined
    assert "not used in this notebook" in combined
    # a winner claim may appear only inside a sentence that negates it ("no model is declared significantly better")
    for phrase in ("universal winner", "significantly better", "best overall"):
        for line in combined.splitlines():
            if phrase in line:
                assert re.search(r"\bnot?\b|\bnothing\b|\bwithout\b", line), f"unnegated claim {phrase!r}: {line.strip()[:160]}"


def test_notebook_outputs_contain_no_errors_or_machine_paths():
    nb = nbformat.read(NOTEBOOK_PATH, as_version=4)
    outputs = [out for c in nb.cells if c.cell_type == "code" for out in c.get("outputs", [])]
    assert not [o for o in outputs if o.get("output_type") == "error"]
    blob = _text(nb)
    assert not re.search(r"/home/|/tmp/|/Users/|[A-Za-z]:\\\\", blob)


# ----------------------------------------------------------------------------------------------------------- saved artifacts
@pytest.fixture(scope="module")
def forecasts():
    return read_forecast_table(_need(FORECASTS), kind="aggregate")


@pytest.fixture(scope="module")
def dev_series():
    return load_aggregate_development_series(CONFIG)


def test_forecast_table_passes_the_full_audit(forecasts, dev_series):
    checks = audit_forecast_table(forecasts, kind="aggregate", dev_series=dev_series)
    assert [(c.name, c.detail) for c in checks if not c.passed] == []


def test_forecasts_cover_four_variants_five_folds_28_days_each(forecasts):
    assert sorted(set(forecasts["model"])) == sorted(VARIANTS)
    assert forecasts.groupby(["model", "fold"]).size().eq(28).all()
    assert forecasts.groupby("model")["fold"].nunique().eq(5).all()
    assert len(forecasts) == 4 * 5 * 28
    assert forecasts["success"].all()
    assert np.isfinite(forecasts["predicted"]).all() and (forecasts["predicted"] >= 0).all()


def test_forecast_windows_match_the_phase_2_protocol_and_stay_inside_development(forecasts, dev_series):
    origins = generate_rolling_origins(n_obs=len(dev_series), horizon=int(CONFIG["backtest"]["horizon"]), n_folds=int(CONFIG["backtest"]["n_folds"]))
    expected = {int(o.fold): (dev_series.index[o.test_start], dev_series.index[o.test_end - 1]) for o in origins}
    dates = pd.to_datetime(forecasts["date"])
    for fold, (start, end) in expected.items():
        part = dates[forecasts["fold"] == fold]
        assert part.min() == start and part.max() == end, f"fold {fold} window differs from the Phase 2 folds"
    assert dates.max() <= pd.Timestamp("2016-04-24")


def test_actuals_in_the_forecast_table_are_the_development_series(forecasts, dev_series):
    part = forecasts.assign(date=pd.to_datetime(forecasts["date"]))
    np.testing.assert_allclose(part["actual"].to_numpy(), dev_series.loc[part["date"]].to_numpy(), rtol=0, atol=1e-9)


def test_saved_metrics_equal_metrics_recomputed_from_the_saved_forecasts(forecasts):
    saved = pd.read_csv(_need(METRICS_DIR / PHASE3["neural_metrics_filename"]))
    recomputed = metrics_by_group(forecasts, ["model", "fold"])
    merged = saved.merge(recomputed, on=["model", "fold"], suffixes=("_saved", "_recomputed"))
    assert len(merged) == len(saved) == 20
    np.testing.assert_allclose(merged["MAE_saved"], merged["MAE_recomputed"], rtol=1e-9)
    np.testing.assert_allclose(merged["RMSE_saved"], merged["RMSE_recomputed"], rtol=1e-9)


def test_seed_metrics_cover_every_network_and_the_ensemble_is_an_average_of_them():
    seeds = pd.read_csv(_need(METRICS_DIR / PHASE3["neural_seed_metrics_filename"]))
    assert len(seeds) == 100
    assert set(seeds["seed"]) == set(CONFIG["neural"]["seeds"])
    assert seeds.groupby(["model", "fold"]).size().eq(len(CONFIG["neural"]["seeds"])).all()
    assert np.isfinite(seeds[["MAE", "RMSE"]].to_numpy()).all()


def test_scalers_were_fitted_before_the_inner_validation_tail_and_before_the_forecast_origin():
    diag = pd.read_csv(_need(METRICS_DIR / PHASE3["neural_diagnostics_filename"]), parse_dates=["history_end", "scaler_fit_end_date"])
    assert len(diag) == 100
    assert (diag["scaler_fit_end_date"] < diag["history_end"]).all(), "a scaler saw the validation tail or later"
    gap = (diag["history_end"] - diag["scaler_fit_end_date"]).dt.days
    assert (gap >= int(CONFIG["neural"]["validation_days"]) - 1).all()
    assert (diag["history_end"] < pd.Timestamp("2016-04-25")).all()
    assert (diag["n_validation_windows"] > 0).all() and (diag["n_train_windows"] > 0).all()
    assert diag["epochs_run"].between(1, int(CONFIG["neural"]["max_epochs"])).all()
    assert (diag["hit_max_epochs"] == (diag["epochs_run"] >= int(CONFIG["neural"]["max_epochs"]))).all()


def test_training_history_is_finite_and_marks_exactly_one_best_epoch_per_network():
    hist = pd.read_csv(_need(METRICS_DIR / PHASE3["neural_history_filename"]))
    assert np.isfinite(hist[["train_loss", "val_loss"]].to_numpy()).all()
    best = hist.groupby(["model", "fold", "seed"])["is_best_epoch"].sum()
    assert len(best) == 100 and best.eq(1).all()


def test_run_metadata_records_the_environment_without_machine_paths():
    meta = json.loads(_need(METRICS_DIR / PHASE3["neural_metadata_filename"]).read_text(encoding="utf-8"))
    assert meta["details"]["variants"] == VARIANTS and meta["details"]["failed_folds"] == 0
    assert meta["details"]["networks_trained"] == 100 and meta["details"]["device"] == "cpu"
    assert meta["details"]["development_end"] == "2016-04-24"
    assert re.fullmatch(r"[0-9a-f]{64}", meta["config_sha256"])
    assert "torch" in meta["environment"]["libraries"]
    assert not re.search(r"/home/|/tmp/|/Users/|[A-Za-z]:\\\\", json.dumps(meta))


def test_the_four_figures_exist_and_are_png_files():
    for name in FIGURES:
        path = _need(PROJECT_ROOT / CONFIG["reports"]["phase3_dir"] / name)
        assert path.read_bytes()[:8] == b"\x89PNG\r\n\x1a\n" and path.stat().st_size > 5_000


# ----------------------------------------------------------------------------------------------------------- full execution
@pytest.mark.slow
@pytest.mark.skipif(os.environ.get("RIVERDEMAND_RUN_NEURAL_NOTEBOOK") != "1",
                    reason="trains 100 networks (~20 min on one CPU thread); set RIVERDEMAND_RUN_NEURAL_NOTEBOOK=1 to run it")
def test_notebook_executes_end_to_end():
    pytest.importorskip("torch", reason="PyTorch is optional; install with: python -m pip install -r requirements-phase3.txt")
    from nbclient import NotebookClient

    nb = nbformat.read(NOTEBOOK_PATH, as_version=4)
    NotebookClient(nb, timeout=7200, kernel_name="python3", resources={"metadata": {"path": str(PROJECT_ROOT)}}).execute()
    assert [o for c in nb.cells if c.cell_type == "code" for o in c.get("outputs", []) if o.get("output_type") == "error"] == []
    text = _text(nb)
    assert "Failed (model, fold) fits: 0 of 20" in text and "Perturbation test passed" in text and "bit for bit" in text
