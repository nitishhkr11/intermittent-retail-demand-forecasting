"""Resumable LSTM/GRU runs: a saved fold is restored only when settings, data and libraries are unchanged."""

from __future__ import annotations

import copy

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("torch")

from src.evaluation.backtesting import BacktestFoldResult  # noqa: E402
from src.models.lstm_gru import (  # noqa: E402
    CHECKPOINT_VERSION, NeuralConfig, NeuralSequenceForecaster, checkpoint_fingerprint, load_checkpoint, save_checkpoint,
)
from src.utils.config import load_config  # noqa: E402

CONFIG = load_config()
VALUES = np.arange(300, dtype=float) % 7 + 10


def _result():
    dates = pd.date_range("2016-01-01", periods=3)
    return BacktestFoldResult("LSTM(calendar)", 2, dates[0], dates[0], dates[-1], dates, np.array([1.0, 2.0, 3.0]), np.array([1.5, 2.5, 3.5]))


def test_fingerprint_is_stable_and_sensitive_to_everything_that_changes_a_result():
    base = checkpoint_fingerprint(CONFIG, "LSTM(calendar)", 2, VALUES)
    assert base == checkpoint_fingerprint(copy.deepcopy(CONFIG), "LSTM(calendar)", 2, VALUES.copy())
    changed = copy.deepcopy(CONFIG)
    changed["neural"]["hidden_size"] += 1
    assert checkpoint_fingerprint(changed, "LSTM(calendar)", 2, VALUES) != base
    assert checkpoint_fingerprint(CONFIG, "GRU(calendar)", 2, VALUES) != base
    assert checkpoint_fingerprint(CONFIG, "LSTM(calendar)", 3, VALUES) != base
    bumped = VALUES.copy()
    bumped[-1] += 1.0
    assert checkpoint_fingerprint(CONFIG, "LSTM(calendar)", 2, bumped) != base
    assert checkpoint_fingerprint(CONFIG, "LSTM(calendar)", 2, VALUES[:-1]) != base
    seeds = copy.deepcopy(CONFIG)
    seeds["neural"]["seeds"] = [9]
    assert checkpoint_fingerprint(seeds, "LSTM(calendar)", 2, VALUES) != base


def test_variants_list_does_not_change_the_fingerprint():
    other = copy.deepcopy(CONFIG)
    other["neural"]["variants"] = other["neural"]["variants"][:1]
    assert checkpoint_fingerprint(other, "LSTM(calendar)", 2, VALUES) == checkpoint_fingerprint(CONFIG, "LSTM(calendar)", 2, VALUES)


def test_round_trip_restores_result_model_and_seconds(tmp_path):
    model = NeuralSequenceForecaster("lstm", "univariate", NeuralConfig.from_config(CONFIG))
    fp = checkpoint_fingerprint(CONFIG, "LSTM(univariate)", 2, VALUES)
    path = save_checkpoint(tmp_path, "LSTM(univariate)", 2, fp, _result(), model, 12.5)
    assert path.exists() and not list(tmp_path.glob("*.tmp"))
    record = load_checkpoint(tmp_path, "LSTM(univariate)", 2, fp)
    assert record is not None and record["seconds"] == 12.5 and record["fold"] == 2
    np.testing.assert_array_equal(record["result"].predictions, [1.5, 2.5, 3.5])
    assert isinstance(record["model"], NeuralSequenceForecaster) and record["model"].name == "LSTM(univariate)"


def test_a_mismatched_fingerprint_missing_or_damaged_file_is_never_restored(tmp_path):
    fp = checkpoint_fingerprint(CONFIG, "GRU(calendar)", 1, VALUES)
    assert load_checkpoint(tmp_path, "GRU(calendar)", 1, fp) is None                       # nothing saved yet
    path = save_checkpoint(tmp_path, "GRU(calendar)", 1, fp, _result(), None, 1.0)
    assert load_checkpoint(tmp_path, "GRU(calendar)", 1, "0" * 64) is None                 # settings or data changed
    assert load_checkpoint(tmp_path, "GRU(calendar)", 2, fp) is None                       # other fold
    path.write_bytes(path.read_bytes()[:20])                                               # truncated write
    assert load_checkpoint(tmp_path, "GRU(calendar)", 1, fp) is None
    path.write_bytes(b"not a pickle")
    assert load_checkpoint(tmp_path, "GRU(calendar)", 1, fp) is None


def test_a_checkpoint_from_another_layout_version_is_ignored(tmp_path):
    import pickle

    fp = checkpoint_fingerprint(CONFIG, "LSTM(calendar)", 1, VALUES)
    path = save_checkpoint(tmp_path, "LSTM(calendar)", 1, fp, _result(), None, 1.0)
    record = pickle.loads(path.read_bytes())
    record["version"] = CHECKPOINT_VERSION + 1
    path.write_bytes(pickle.dumps(record))
    assert load_checkpoint(tmp_path, "LSTM(calendar)", 1, fp) is None


def test_the_checkpoint_folder_is_git_ignored_and_configured():
    assert CONFIG["phase3"]["neural_checkpoint_dir"].startswith("artifacts/models/")
