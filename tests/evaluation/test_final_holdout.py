"""The controlled final-holdout workflow: freezing, refusals, the one-time run, and the fact that it never ran on real data.

Everything here uses SYNTHETIC series in a temporary folder. The real holdout sales are never read by these tests:
`load_holdout_inputs` is only checked for what it exposes BEFORE `read_actuals` is called.
"""

from __future__ import annotations

import copy
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.data.aggregate_series import DevelopmentWindow
from src.evaluation import final_holdout as fh
from src.evaluation.model_comparison import AUDIT_COLUMNS
from src.models.base import BaseForecaster
from src.utils.config import load_config
from src.utils.paths import PROJECT_ROOT

CONFIG = load_config()
DEV_DAYS, HORIZON = 91, 28
DATES = pd.date_range("2016-01-01", periods=DEV_DAYS + HORIZON, freq="D")
DEV_DATES, HOLDOUT_DATES = DATES[:DEV_DAYS], DATES[DEV_DAYS:]
WINDOW = DevelopmentWindow(DEV_DATES[0], DEV_DATES[-1], HOLDOUT_DATES[0], HOLDOUT_DATES[-1], DEV_DAYS, HORIZON)
ITEMS = ["ITEM_A", "ITEM_B", "ITEM_C"]


# ---------------------------------------------------------------------------------------------- synthetic world
def _aggregate_full() -> pd.Series:
    t = np.arange(len(DATES))
    return pd.Series(100 + 10 * np.sin(2 * np.pi * (t % 7) / 7) + 0.5 * t, index=DATES, name="aggregate_sales")


def _item_full(k: int) -> pd.Series:
    t = np.arange(len(DATES))
    return pd.Series(((t + k) % 7 == 0) * (2 + k) + ((t % 5 == 0) * 1.0), index=DATES, name=ITEMS[k], dtype=float)


class Events:
    """Records the order in which forecasts are produced and holdout actuals are read."""

    def __init__(self):
        self.log = []


class StubAggregate(BaseForecaster):
    def __init__(self, name, value, events, fail=False):
        self.name, self.value, self.events, self.fail = name, value, events, fail

    def fit(self, history):
        assert history.index[-1] == DEV_DATES[-1], "fitted on something other than the development window"
        return self

    def predict(self, horizon):
        self.events.log.append(("predict", self.name))
        if self.fail:
            raise RuntimeError("boom")
        return np.full(horizon, float(self.value))


class StubPooled:
    name = "XGBoost(pooled)"

    def __init__(self, events):
        self.events = events

    def fit(self, histories):
        assert all(s.index[-1] == DEV_DATES[-1] for s in histories.values())
        self._means = {k: float(s.mean()) for k, s in histories.items()}
        return self

    def predict(self, horizon):
        self.events.log.append(("predict", self.name))
        return {k: np.full(horizon, m) for k, m in self._means.items()}


def make_config(tmp_path: Path, aggregate=("SeasonalNaive", "ARIMA"), item_level=("SeasonalNaive", "TSB", "XGBoost(pooled)")):
    config = copy.deepcopy(CONFIG)
    config["phase3"]["final_dir"] = str(tmp_path / "final")
    config["final_holdout"]["candidates"] = {"aggregate": list(aggregate), "item_level": list(item_level)}
    return config


def make_summary(rows):
    return pd.DataFrame(
        [{"model": m, "MAE_mean": mae, "RMSE_mean": mae * 1.3, "folds_scored": 5 - failed, "folds_failed": failed}
         for m, mae, failed in rows]
    )


def make_evidence(aggregate=None, item_level=None, audit=None):
    aggregate = aggregate or [("SeasonalNaive", 120.0, 0), ("ARIMA(6, 1, 1)", 95.0, 0)]
    item_level = item_level or [("SeasonalNaive", 1.4, 0), ("TSB(alpha=0.1,beta=0.1)", 1.1, 0), ("XGBoost(pooled)", 1.2, 0)]
    audit = audit if audit is not None else pd.DataFrame(columns=AUDIT_COLUMNS)
    return fh.DevelopmentEvidence(
        summaries={"aggregate": make_summary(aggregate), "item_level": make_summary(item_level)},
        audit=audit, window=WINDOW, artifact_sha256={"seasonal_naive_backtest_forecasts.csv": "ab" * 32},
    )


def make_inputs(events: Events, read_actuals_calls=None):
    aggregate, items = _aggregate_full(), {k: _item_full(i) for i, k in enumerate(ITEMS)}

    def read_actuals():
        events.log.append(("read_actuals",))
        return fh.HoldoutActuals(aggregate.iloc[DEV_DAYS:], {k: s.iloc[DEV_DAYS:] for k, s in items.items()})

    return fh.HoldoutInputs(aggregate.iloc[:DEV_DAYS], {k: s.iloc[:DEV_DAYS] for k, s in items.items()}, None, read_actuals, HORIZON)


def make_factories(events: Events, arima_name="ARIMA(6, 1, 1)", fail_arima=False):
    from src.models.baseline import SeasonalNaive
    from src.models.tsb import TSBForecaster

    return {
        "aggregate": {
            "SeasonalNaive": lambda: SeasonalNaive(season_length=7),
            "ARIMA": lambda: StubAggregate(arima_name, 130.0, events, fail=fail_arima),
        },
        "item_level": {
            "SeasonalNaive": lambda: SeasonalNaive(season_length=7),
            "TSB": lambda: TSBForecaster(alpha=0.1, beta=0.1),
            "XGBoost(pooled)": lambda: StubPooled(events),
        },
    }


@pytest.fixture
def frozen_world(tmp_path):
    config = make_config(tmp_path)
    record = fh.freeze(config, evidence=make_evidence())
    return config, record


# ---------------------------------------------------------------------------------------------- candidates
def test_the_configured_candidates_are_valid():
    candidates = fh.configured_candidates(CONFIG)
    assert set(candidates) == {"aggregate", "item_level"}
    assert "XGBoost(pooled)" in candidates["item_level"] and "Prophet(additive)" in candidates["aggregate"]


@pytest.mark.parametrize(
    "scope,names,message",
    [("aggregate", ["SeasonalNaive", "NHiTS"], "Unknown"), ("aggregate", ["ARIMA", "ARIMA"], "twice"), ("item_level", [], "empty")],
)
def test_bad_candidate_lists_are_refused(tmp_path, scope, names, message):
    config = make_config(tmp_path)
    config["final_holdout"]["candidates"][scope] = names
    with pytest.raises(fh.FinalHoldoutError, match=message):
        fh.configured_candidates(config)


def test_name_matching_never_confuses_neighbouring_models():
    assert fh.name_matches("ARIMA", "ARIMA(6, 1, 1)")
    assert not fh.name_matches("ARIMA", "SARIMA(6, 1, 1)x(0, 0, 2, 7)")
    assert fh.name_matches("XGBoost(pooled)", "XGBoost(pooled)")
    assert not fh.name_matches("XGBoost(pooled)", "XGBoost(pooled,poisson)")
    assert not fh.name_matches("Prophet(additive)", "Prophet(additive,no-regressors)")
    assert fh.name_matches("TSB", "TSB(alpha=0.1,beta=0.1)")
    with pytest.raises(fh.FinalHoldoutError, match="ambiguous"):
        fh.resolve_model_name("TSB", ["TSB(alpha=0.1,beta=0.1)", "TSB(alpha=0.2,beta=0.1)"])
    assert fh.resolve_model_name("Croston", ["TSB(alpha=0.1,beta=0.1)"]) is None


def test_settings_are_json_serialisable_for_every_known_candidate():
    for scope, names in fh.KNOWN_CANDIDATES.items():
        for name in names:
            json.dumps(fh.candidate_settings(scope, name, CONFIG))
    assert fh.candidate_settings("item_level", "TSB", CONFIG) == {"alpha": 0.1, "beta": 0.1}
    assert fh.candidate_settings("aggregate", "ARIMA", CONFIG) == {"order": [6, 1, 1]}
    with pytest.raises(fh.FinalHoldoutError):
        fh.candidate_settings("aggregate", "NHiTS", CONFIG)


def test_default_factories_build_the_models_the_candidates_name():
    factories = fh.default_factories(CONFIG)
    assert factories["aggregate"]["ARIMA"]().name == "ARIMA(6, 1, 1)"
    assert factories["aggregate"]["SARIMA"]().name == "SARIMA(6, 1, 1)x(0, 0, 2, 7)"
    assert factories["aggregate"]["HoltWinters"]().name == "HoltWinters(trend=add)"
    assert factories["item_level"]["Croston"]().name == "Croston(alpha=0.1)"
    assert factories["item_level"]["TSB"]().name == "TSB(alpha=0.1,beta=0.1)"
    for scope in fh.SCOPES:
        assert factories[scope]["SeasonalNaive"]().name == "SeasonalNaive"


def test_default_factories_for_heavy_models_match_the_saved_names():
    pytest.importorskip("xgboost")
    calendar = pd.DataFrame({c: 0 for c in ["snap_ca", "event_cultural", "event_national", "event_religious",
                                            "event_sporting", "is_christmas"]}, index=DATES)
    factories = fh.default_factories(CONFIG, calendar)
    assert factories["item_level"]["XGBoost(pooled)"]().name == "XGBoost(pooled)"
    try:
        import torch  # noqa: F401
    except ImportError:
        pass
    else:
        assert factories["aggregate"]["LSTM(calendar)"]().name == "LSTM(calendar)"
        assert factories["aggregate"]["GRU(calendar)"]().name == "GRU(calendar)"
    try:
        import prophet  # noqa: F401
    except ImportError:
        pass
    else:
        assert factories["aggregate"]["Prophet(additive)"]().name == "Prophet(additive)"


# ---------------------------------------------------------------------------------------------- headline choice
def test_headline_is_the_lowest_development_mae():
    summary = make_summary([("A", 5.0, 0), ("B", 3.0, 0), ("C", 4.0, 0)])
    headline, table = fh.choose_headline(summary, {"A": "A", "B": "B", "C": "C"})
    assert headline == "B" and list(table["eligible_for_headline"]) == [True, True, True]


def test_a_candidate_with_a_failed_development_fold_cannot_be_the_headline_but_is_still_listed():
    summary = make_summary([("A", 5.0, 0), ("B", 1.0, 1)])
    headline, table = fh.choose_headline(summary, {"A": "A", "B": "B"})
    assert headline == "A"
    assert bool(table.set_index("candidate").loc["B", "eligible_for_headline"]) is False


def test_ties_are_broken_alphabetically_and_all_ineligible_is_an_error():
    summary = make_summary([("B", 2.0, 0), ("A", 2.0, 0)])
    assert fh.choose_headline(summary, {"B": "B", "A": "A"})[0] == "A"
    with pytest.raises(fh.NotReadyToFreeze):
        fh.choose_headline(make_summary([("A", 2.0, 5)]), {"A": "A"})
    with pytest.raises(fh.NotReadyToFreeze, match="No development result"):
        fh.choose_headline(summary, {"Z": "Z"})


# ---------------------------------------------------------------------------------------------- freeze
def test_freeze_records_the_choice_without_evaluating_anything(frozen_world):
    config, record = frozen_world
    path = fh.frozen_path(config)
    stored = json.loads(path.read_text(encoding="utf-8"))
    assert stored == json.loads(json.dumps(record))
    assert stored["holdout_evaluated"] is False
    assert stored["scopes"]["aggregate"]["headline_candidate"] == "ARIMA"
    assert stored["scopes"]["aggregate"]["headline_model"] == "ARIMA(6, 1, 1)"
    assert stored["scopes"]["item_level"]["headline_candidate"] == "TSB"
    assert stored["config_sha256"] == fh.config_section_hash(config, fh.FROZEN_CONFIG_SECTIONS)
    assert stored["reserved_holdout"] == {"start": "2016-04-01", "end": "2016-04-28", "days": 28}
    assert stored["development_window"]["end"] == "2016-03-31"
    assert not fh.results_path(config).exists() and not fh.record_path(config).exists()
    for entry in stored["scopes"]["aggregate"]["candidates"]:
        assert "settings" in entry and "dev_MAE_mean" in entry


def test_the_frozen_file_has_no_machine_paths_or_identifiers(frozen_world):
    config, _ = frozen_world
    text = fh.frozen_path(config).read_text(encoding="utf-8")
    for pattern in (r"/home/", r"/tmp/", r"/Users/", r"/mnt/", r"[A-Za-z]:\\", "hostname", "username"):
        assert not re.search(pattern, text), pattern


def test_freeze_does_not_overwrite_without_the_flag_and_not_after_evaluation(frozen_world):
    config, _ = frozen_world
    with pytest.raises(fh.FinalHoldoutError, match="already exists"):
        fh.freeze(config, evidence=make_evidence())
    again = fh.freeze(config, evidence=make_evidence(aggregate=[("SeasonalNaive", 50.0, 0), ("ARIMA(6, 1, 1)", 95.0, 0)]),
                      overwrite=True)
    assert again["scopes"]["aggregate"]["headline_candidate"] == "SeasonalNaive"
    events = Events()
    fh.run(config, confirm=fh.CONFIRMATION_PHRASE, inputs=make_inputs(events), factories=make_factories(events))
    with pytest.raises(fh.FinalHoldoutError, match="already been evaluated"):
        fh.freeze(config, evidence=make_evidence(), overwrite=True)


def test_freeze_refuses_when_a_candidate_has_no_development_result(tmp_path):
    config = make_config(tmp_path)
    audit = pd.DataFrame([{"artifact": "arima_backtest_forecasts.csv", "scope": "aggregate", "status": "missing", "rows": 0,
                           "models": "", "failed_checks": 0,
                           "detail": "Not found. Create it with: python -m jupyter nbconvert --to notebook --execute --inplace notebooks/04_classical_models.ipynb"}])
    evidence = make_evidence(aggregate=[("SeasonalNaive", 120.0, 0)], audit=audit)
    with pytest.raises(fh.NotReadyToFreeze) as info:
        fh.freeze(config, evidence=evidence)
    message = str(info.value)
    assert "aggregate candidate ARIMA" in message and "nbconvert" in message and "Nothing was written" in message
    assert not fh.frozen_path(config).exists()


# ---------------------------------------------------------------------------------------------- status
def test_status_before_freezing_lists_what_is_missing_and_what_to_do(tmp_path):
    config = make_config(tmp_path)
    result = fh.status(config, evidence=make_evidence(aggregate=[("SeasonalNaive", 1.0, 0)]))
    assert not result.frozen and not result.evaluated
    assert any("ARIMA" in m for m in result.missing)
    text = result.to_text()
    assert "freeze" in text and fh.CONFIRMATION_PHRASE in text and "holdout evaluated        : no" in text


def test_status_after_freezing_detects_config_drift(frozen_world):
    config, _ = frozen_world
    ok = fh.status(config)
    assert ok.frozen and ok.config_matches and fh.CONFIRMATION_PHRASE in ok.to_text()
    drifted = copy.deepcopy(config)
    drifted["models"]["arima_order"] = [1, 1, 1]
    bad = fh.status(drifted)
    assert bad.config_matches is False and "NO" in bad.to_text()


# ---------------------------------------------------------------------------------------------- run: refusals
def _no_data_loaded(monkeypatch):
    def boom(*args, **kwargs):
        raise AssertionError("data must not be loaded when the run is refused")

    monkeypatch.setattr(fh, "load_holdout_inputs", boom)


def test_run_refuses_without_the_confirmation_phrase(frozen_world, monkeypatch):
    config, _ = frozen_world
    _no_data_loaded(monkeypatch)
    for wrong in (None, "yes", fh.CONFIRMATION_PHRASE.lower(), fh.CONFIRMATION_PHRASE + " "):
        with pytest.raises(fh.FinalHoldoutError, match="explicit confirmation"):
            fh.run(config, confirm=wrong)
    assert not fh.results_path(config).exists()


def test_run_refuses_without_a_freeze(tmp_path, monkeypatch):
    config = make_config(tmp_path)
    _no_data_loaded(monkeypatch)
    with pytest.raises(fh.FinalHoldoutError, match="Freeze first"):
        fh.run(config, confirm=fh.CONFIRMATION_PHRASE)


def test_run_refuses_after_the_config_changed(frozen_world, monkeypatch):
    config, _ = frozen_world
    _no_data_loaded(monkeypatch)
    config["neural"]["hidden_size"] = 64
    with pytest.raises(fh.FinalHoldoutError, match="has changed since the candidates were frozen"):
        fh.run(config, confirm=fh.CONFIRMATION_PHRASE)


def test_run_refuses_a_second_evaluation_without_override(frozen_world):
    config, _ = frozen_world
    events = Events()
    fh.run(config, confirm=fh.CONFIRMATION_PHRASE, inputs=make_inputs(events), factories=make_factories(events))
    second = Events()
    with pytest.raises(fh.FinalHoldoutError, match="--override-existing"):
        fh.run(config, confirm=fh.CONFIRMATION_PHRASE, inputs=make_inputs(second), factories=make_factories(second))
    assert ("read_actuals",) not in second.log


def test_a_model_that_no_longer_builds_as_frozen_is_refused_before_actuals_are_read(frozen_world):
    config, _ = frozen_world
    events = Events()
    with pytest.raises(fh.FinalHoldoutError, match="Settings changed after the freeze"):
        fh.run(config, confirm=fh.CONFIRMATION_PHRASE, inputs=make_inputs(events),
               factories=make_factories(events, arima_name="ARIMA(1, 1, 1)"))
    assert ("read_actuals",) not in events.log
    assert not fh.results_path(config).exists()


# ---------------------------------------------------------------------------------------------- run: the evaluation
def test_forecasts_are_all_made_before_the_holdout_is_read_and_it_is_read_once(frozen_world):
    config, _ = frozen_world
    events = Events()
    fh.run(config, confirm=fh.CONFIRMATION_PHRASE, inputs=make_inputs(events), factories=make_factories(events))
    kinds = [e[0] for e in events.log]
    assert kinds.count("read_actuals") == 1
    assert kinds.index("read_actuals") == len(kinds) - 1, "every forecast must exist before the holdout is read"
    assert kinds.count("predict") >= 2


def test_results_are_scored_against_the_real_holdout_values_only(frozen_world):
    config, _ = frozen_world
    events = Events()
    results = fh.run(config, confirm=fh.CONFIRMATION_PHRASE, inputs=make_inputs(events), factories=make_factories(events))
    saved = pd.read_csv(fh.results_path(config))
    pd.testing.assert_frame_equal(saved.drop(columns=["error"]), results.drop(columns=["error"]), check_dtype=False)

    forecasts = pd.read_csv(fh.forecasts_path(config), parse_dates=["date"])
    assert forecasts["date"].min() == HOLDOUT_DATES[0] and forecasts["date"].max() == HOLDOUT_DATES[-1]
    aggregate = forecasts[(forecasts["scope"] == "aggregate") & (forecasts["model"] == "SeasonalNaive")]
    assert len(aggregate) == HORIZON
    np.testing.assert_allclose(aggregate["actual"].to_numpy(), _aggregate_full().iloc[DEV_DAYS:].to_numpy())

    # trend of +0.5/day: repeating the last week is short by 3.5, 7, 10.5 and 14 in the four forecast weeks (mean 8.75)
    row = results[(results["scope"] == "aggregate") & (results["model"] == "SeasonalNaive")].iloc[0]
    assert row["MAE"] == pytest.approx(8.75) and row["bias"] == pytest.approx(-8.75)
    assert row["RMSE"] == pytest.approx(np.sqrt(np.mean(np.repeat([3.5, 7.0, 10.5, 14.0], 7) ** 2)))
    # constant 130 forecast: independent recomputation
    arima = results[(results["scope"] == "aggregate") & (results["model"] == "ARIMA(6, 1, 1)")].iloc[0]
    actual = _aggregate_full().iloc[DEV_DAYS:].to_numpy()
    assert arima["MAE"] == pytest.approx(np.mean(np.abs(actual - 130.0)))
    assert arima["RMSE"] == pytest.approx(np.sqrt(np.mean((actual - 130.0) ** 2)))
    assert arima["WAPE"] == pytest.approx(np.sum(np.abs(actual - 130.0)) / actual.sum())
    history = _aggregate_full().iloc[:DEV_DAYS].to_numpy()
    scale = np.mean(np.abs(history[7:] - history[:-7]))
    assert arima["MASE"] == pytest.approx(arima["MAE"] / scale)


def test_every_frozen_candidate_is_reported_and_the_headline_is_flagged_per_scope(frozen_world):
    config, record = frozen_world
    events = Events()
    results = fh.run(config, confirm=fh.CONFIRMATION_PHRASE, inputs=make_inputs(events), factories=make_factories(events))
    assert set(results["scope"]) == {"aggregate", "item_level"}
    assert sorted(results[results["scope"] == "aggregate"]["model"]) == ["ARIMA(6, 1, 1)", "SeasonalNaive"]
    assert sorted(results[results["scope"] == "item_level"]["model"]) == [
        "SeasonalNaive", "TSB(alpha=0.1,beta=0.1)", "XGBoost(pooled)"]
    for scope in fh.SCOPES:
        flagged = results[(results["scope"] == scope) & results["headline"]]
        assert list(flagged["model"]) == [record["scopes"][scope]["headline_model"]]
    assert set(results[results["scope"] == "aggregate"]["averaging"]) == {"single aggregate series"}
    assert set(results[results["scope"] == "item_level"]["averaging"]) == {"equal-weight mean over items"}
    assert (results[results["scope"] == "item_level"]["n_series"] == len(ITEMS)).all()


def test_item_level_results_are_equal_weight_means_over_items(frozen_world):
    config, _ = frozen_world
    events = Events()
    results = fh.run(config, confirm=fh.CONFIRMATION_PHRASE, inputs=make_inputs(events), factories=make_factories(events))
    forecasts = pd.read_csv(fh.forecasts_path(config))
    sub = forecasts[(forecasts["scope"] == "item_level") & (forecasts["model"] == "XGBoost(pooled)")]
    per_item = (sub["actual"] - sub["predicted"]).abs().groupby(sub["item_id"]).mean()
    row = results[(results["scope"] == "item_level") & (results["model"] == "XGBoost(pooled)")].iloc[0]
    assert row["MAE"] == pytest.approx(per_item.mean())


def test_a_failed_candidate_is_reported_as_failed_and_never_replaced(frozen_world):
    config, _ = frozen_world
    events = Events()
    results = fh.run(config, confirm=fh.CONFIRMATION_PHRASE, inputs=make_inputs(events),
                     factories=make_factories(events, fail_arima=True))
    failed = results[results["model"] == "ARIMA(6, 1, 1)"].iloc[0]
    assert bool(failed["success"]) is False and np.isnan(failed["MAE"]) and "boom" in failed["error"]
    forecasts = pd.read_csv(fh.forecasts_path(config))
    assert forecasts[forecasts["model"] == "ARIMA(6, 1, 1)"]["predicted"].isna().all()
    assert bool(results[results["model"] == "SeasonalNaive"].iloc[0]["success"]) is True


def test_the_record_documents_the_single_evaluation(frozen_world):
    config, record = frozen_world
    events = Events()
    fh.run(config, confirm=fh.CONFIRMATION_PHRASE, inputs=make_inputs(events), factories=make_factories(events))
    saved = json.loads(fh.record_path(config).read_text(encoding="utf-8"))
    assert saved["evaluation_count"] == 1 and saved["warning"] is None
    assert saved["details"]["override_used"] is False
    assert saved["details"]["frozen_file_sha256"] == fh.file_sha256(fh.frozen_path(config))
    assert saved["details"]["holdout_window"] == record["reserved_holdout"]
    assert not {"hostname", "node", "user", "username"} & set(saved["environment"])
    # the frozen file itself is never rewritten by a run
    assert json.loads(fh.frozen_path(config).read_text(encoding="utf-8"))["holdout_evaluated"] is False


def test_an_explicit_override_archives_the_old_result_and_warns(frozen_world, capsys):
    config, _ = frozen_world
    first = Events()
    fh.run(config, confirm=fh.CONFIRMATION_PHRASE, inputs=make_inputs(first), factories=make_factories(first))
    original = fh.results_path(config).read_text(encoding="utf-8")
    second = Events()
    fh.run(config, confirm=fh.CONFIRMATION_PHRASE, override_existing=True, inputs=make_inputs(second),
           factories=make_factories(second))
    saved = json.loads(fh.record_path(config).read_text(encoding="utf-8"))
    assert saved["evaluation_count"] == 2 and "more than once" in saved["warning"]
    assert saved["details"]["override_used"] is True and saved["details"]["archived_previous_files"]
    archive = fh.results_path(config).with_name("final_holdout_results.superseded-1.csv")
    assert archive.exists() and archive.read_text(encoding="utf-8") == original
    assert "more than once" in capsys.readouterr().err
    assert fh.status(config).evaluation_count == 2


def test_misaligned_holdout_dates_are_refused(frozen_world):
    config, _ = frozen_world
    events = Events()
    inputs = make_inputs(events)
    shifted = inputs.read_actuals

    def read_actuals():
        actuals = shifted()
        return fh.HoldoutActuals(actuals.aggregate.shift(1, freq="D"), actuals.items)

    inputs.read_actuals = read_actuals
    with pytest.raises(fh.FinalHoldoutError, match="do not match the forecast dates"):
        fh.run(config, confirm=fh.CONFIRMATION_PHRASE, inputs=inputs, factories=make_factories(events))
    assert not fh.results_path(config).exists()


# ---------------------------------------------------------------------------------------------- command line
def test_command_line_refusal_exit_code_and_message(frozen_world, monkeypatch, capsys):
    config, _ = frozen_world
    monkeypatch.setattr(fh, "load_config", lambda: config)
    _no_data_loaded(monkeypatch)
    assert fh.main(["run"]) == 2
    assert fh.CONFIRMATION_PHRASE in capsys.readouterr().err


def test_command_line_status_prints_the_next_step(frozen_world, monkeypatch, capsys):
    config, _ = frozen_world
    monkeypatch.setattr(fh, "load_config", lambda: config)
    assert fh.main(["status"]) == 0
    assert "candidates frozen" in capsys.readouterr().out


# ---------------------------------------------------------------------------------------------- real files: structure only
def test_real_inputs_expose_no_holdout_sales_before_read_actuals():
    from src.data.aggregate_series import aggregate_series_path, load_aggregate_development

    if not aggregate_series_path(CONFIG).exists():
        pytest.skip("the processed aggregate parquet file is not present")
    inputs = fh.load_holdout_inputs(CONFIG)
    _, window = load_aggregate_development(CONFIG)
    assert inputs.aggregate_history.index.max() == window.dev_end
    assert all(s.index.max() == window.dev_end for s in inputs.item_histories.values())
    assert inputs.calendar is not None and "sales" not in inputs.calendar.columns
    assert callable(inputs.read_actuals)          # NOT called: the real holdout is never read by the test suite


def test_only_the_final_holdout_module_can_read_holdout_actuals():
    offenders = []
    for folder in ("src", "app"):
        for path in (PROJECT_ROOT / folder).rglob("*.py"):
            if path.name == "final_holdout.py":
                continue
            text = path.read_text(encoding="utf-8")
            if "HoldoutActuals" in text or "read_actuals" in text:
                offenders.append(path.relative_to(PROJECT_ROOT).as_posix())
    assert offenders == []


def test_no_final_results_exist_in_this_repository():
    """The real holdout has not been evaluated: no results or record file ships with the project."""
    directory = PROJECT_ROOT / CONFIG["phase3"]["final_dir"]
    for key in ("results_filename", "forecasts_filename", "record_filename"):
        assert not (directory / CONFIG["final_holdout"][key]).exists()
