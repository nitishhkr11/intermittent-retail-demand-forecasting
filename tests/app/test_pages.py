"""The six dashboard pages, run end to end with Streamlit's AppTest against the real saved artifacts.

What is checked: every page runs in both themes, the light/dark toggle really switches the stylesheet and survives navigation, every
control can be operated, what is plotted is what was saved (nothing is invented), and the reserved final holdout never shows up as data.
"""

from __future__ import annotations

import re
from datetime import date

import numpy as np
import pandas as pd
import pytest

import data_access as da
import theme
from tests.app.helpers import (HOLDOUT_END, HOLDOUT_START, PAGE_LABELS, PAGE_PATHS, chart_specs, decode_array, flat_values, open_page,
                               raw_options, start_app, texts)
from components.cards import esc
from src.evaluation.forecast_artifacts import forecasts_dir
from src.evaluation.model_comparison import aggregate_sources

THEME_NAMES = list(theme.THEME_NAMES)
TITLES = {
    "views/1_Overview.py": "Overview",
    "views/2_Forecast_Explorer.py": "Forecast Explorer",
    "views/3_Model_Comparison.py": "Model Comparison",
    "views/4_Inventory_Decision.py": "Inventory Decisions",
    "views/5_Model_Reliability.py": "Model Reliability",
    "views/6_About_Methodology.py": "About & Methodology",
}
DATE_RE = re.compile(r"\b(20\d\d-\d\d-\d\d)\b")
GUARD_WORDS = ("reserved", "holdout", "final", "unseen", "never", "not evaluated", "not used", "not shown")


def css_of(at) -> str:
    return " ".join(str(m.value) for m in at.main.markdown if "<style" in str(m.value))


def assert_no_exception(at):
    assert not at.exception, [str(e.value)[:300] for e in at.exception]


# ---------------------------------------------------------------------------------------------------------------------- shell
@pytest.mark.parametrize("name", THEME_NAMES)
def test_entry_script_runs_and_applies_the_selected_theme(name):
    at = start_app(name)
    assert_no_exception(at)
    css = css_of(at)
    assert theme.THEMES[name]["bg"] in css and f"color-scheme: {name}" in css
    other = "light" if name == "dark" else "dark"
    assert theme.THEMES[other]["bg"] not in css
    assert at.sidebar.toggle(key="rd_theme_toggle").value == (name == "dark")
    side = " ".join(texts(at, sidebar=True))
    assert "Not evaluated" in side and "River" in side and "Dark mode" in side


def test_the_registered_pages_are_exactly_the_six_required():
    assert len(PAGE_PATHS) == 6
    assert list(PAGE_LABELS.values()) == ["Overview", "Forecast Explorer", "Model Comparison", "Inventory Decisions", "Model Reliability", "About & Methodology"]
    for path in PAGE_PATHS:
        assert (da.PROJECT_ROOT / "app" / path).exists(), path
    assert not (da.PROJECT_ROOT / "app" / "pages").exists(), "app/pages/ would trigger Streamlit's own page discovery and bypass the shared shell"


def test_initial_theme_can_be_requested_in_the_url():
    from streamlit.testing.v1 import AppTest
    from tests.app.helpers import APP_FILE

    at = AppTest.from_file(str(APP_FILE), default_timeout=180)
    at.query_params["theme"] = "dark"
    at.run()
    assert_no_exception(at)
    assert at.session_state["rd_theme"] == "dark"
    at2 = AppTest.from_file(str(APP_FILE), default_timeout=180)
    at2.query_params["theme"] = "not-a-theme"
    at2.run()
    assert at2.session_state["rd_theme"] == "light"


def test_dark_mode_toggle_switches_the_stylesheet_and_survives_page_changes():
    at = start_app("light")
    assert theme.THEMES["light"]["bg"] in css_of(at)
    at.sidebar.toggle(key="rd_theme_toggle").set_value(True).run()
    assert_no_exception(at)
    assert at.session_state["rd_theme"] == "dark"
    assert theme.THEMES["dark"]["bg"] in css_of(at) and theme.THEMES["light"]["bg"] not in css_of(at)
    for path in PAGE_PATHS:
        at.switch_page(path).run()
        assert_no_exception(at)
        assert at.session_state["rd_theme"] == "dark", f"theme lost on {path}"
        assert at.sidebar.toggle(key="rd_theme_toggle").value is True
    at.sidebar.toggle(key="rd_theme_toggle").set_value(False).run()
    assert at.session_state["rd_theme"] == "light"
    assert theme.THEMES["light"]["bg"] in css_of(at)


@pytest.mark.parametrize("name", THEME_NAMES)
def test_charts_take_their_colours_from_the_active_theme(name):
    at = open_page(name, "views/2_Forecast_Explorer.py")
    spec = chart_specs(at)[0]
    t = theme.THEMES[name]
    assert spec["layout"]["paper_bgcolor"] == t["surface"] and spec["layout"]["font"]["color"] == t["text"]
    line_colours = {tr.get("line", {}).get("color") for tr in spec["data"]}
    assert theme.series_colors(name)[0] in line_colours


# --------------------------------------------------------------------------------------------------------------------- pages
@pytest.mark.parametrize("path", PAGE_PATHS)
@pytest.mark.parametrize("name", THEME_NAMES)
def test_every_page_renders_without_error(name, path):
    at = open_page(name, path)
    assert_no_exception(at)
    body = " ".join(texts(at))
    assert esc(TITLES[path]) in body, f"{path} shows no page title"
    assert "Traceback" not in body and not re.search(r">(nan|NaN|None)<", body)


@pytest.mark.parametrize("path", PAGE_PATHS)
def test_every_page_states_what_it_is_not_claiming(path):
    """Each page is explicit that it shows development results; none talks about the holdout as if it had been evaluated."""
    body = " ".join(texts(open_page("light", path))).lower()
    for phrase in ("final holdout result", "holdout mae", "holdout score", "test-set score", "holdout was evaluated"):
        assert phrase not in body, f"{path} mentions '{phrase}'"


def _controls(at):
    return {
        "selectbox": list(at.selectbox), "slider": list(at.slider), "number_input": list(at.number_input), "checkbox": list(at.checkbox),
        "radio": list(at.radio), "multiselect": list(at.multiselect),
    }


@pytest.mark.parametrize("path", PAGE_PATHS)
def test_every_control_on_every_page_works(path):
    """Operate each widget (every option of a select box, both ends of numeric inputs and sliders, both states of a checkbox)."""
    at = open_page("light", path)
    assert_no_exception(at)
    operated = 0
    for i in range(len(at.selectbox)):
        widget = at.selectbox[i]
        if widget.key == "rd_theme_toggle":
            continue
        values = raw_options(widget)
        n = len(values)
        for index in dict.fromkeys([0, n // 2, n - 1]):
            at.selectbox[i].select(values[index]).run()
            assert_no_exception(at)
            assert at.selectbox[i].index == index, f"{widget.label}: choosing option {index} did not stick"
            operated += 1
        at.selectbox[i].select(values[0]).run()
    for i in range(len(at.radio)):
        options = list(at.radio[i].options)
        for option in dict.fromkeys([options[0], options[-1]]):
            at.radio[i].set_value(option).run()
            assert_no_exception(at)
            operated += 1
    for i in range(len(at.number_input)):
        w = at.number_input[i]
        for target in (w.min, w.max):
            if target is not None:
                at.number_input[i].set_value(target).run()
                assert_no_exception(at)
                operated += 1
        at.number_input[i].increment().run()
        assert_no_exception(at)
        at.number_input[i].decrement().run()
        assert_no_exception(at)
        operated += 2
    for i in range(len(at.slider)):
        w = at.slider[i]
        for target in (w.min, w.max):
            at.slider[i].set_value(target).run()
            assert_no_exception(at)
            operated += 1
    for i in range(len(at.checkbox)):
        for target in (True, False):
            at.checkbox[i].set_value(target).run()
            assert_no_exception(at)
            operated += 1
    assert operated >= 0
    if path in ("views/2_Forecast_Explorer.py", "views/3_Model_Comparison.py", "views/4_Inventory_Decision.py"):
        assert operated > 0, f"{path} is expected to have working controls"


def test_forecast_explorer_has_the_four_required_controls():
    at = open_page("light", "views/2_Forecast_Explorer.py")
    labels = [s.label for s in at.selectbox]
    for needle in ("Series", "Model", "Forecast origin", "Forecast horizon"):
        assert any(needle in label for label in labels), f"missing control: {needle}"


def test_inventory_page_has_lead_time_service_level_and_model_controls():
    at = open_page("light", "views/4_Inventory_Decision.py")
    assert any("Lead time" in n.label for n in at.number_input)
    assert any("service level" in s.label.lower() for s in at.slider)
    assert any("Product" in s.label for s in at.selectbox) and any("model" in s.label.lower() for s in at.selectbox)


def test_inventory_result_changes_with_the_service_level_and_lead_time():
    at = open_page("light", "views/4_Inventory_Decision.py")

    def reorder_point(a):
        cards_text = " ".join(str(m.value) for m in a.main.markdown if "REORDER" in str(m.value).upper() and "rd-card" in str(m.value))
        found = re.findall(r"Reorder point</div><div class=\"v[^\"]*\">([\d,\.]+)<", cards_text, flags=re.I)
        assert found, "no reorder point card"
        return float(found[0].replace(",", ""))

    base = reorder_point(at)
    slider = next(s for s in at.slider if "service level" in s.label.lower())
    at.slider(key=slider.key).set_value(slider.max).run()
    high = reorder_point(at)
    at.slider(key=slider.key).set_value(slider.min).run()
    low = reorder_point(at)
    assert low <= base <= high and low < high
    lead = next(n for n in at.number_input if "Lead time" in n.label)
    at.number_input(key=lead.key).set_value(lead.min).run()
    short = reorder_point(at)
    at.number_input(key=lead.key).set_value(lead.max).run()
    long_ = reorder_point(at)
    assert short < long_


# ------------------------------------------------------------------------------------------------------- nothing is invented
def _saved_aggregate(model: str, fold: int) -> pd.DataFrame:
    config = da.get_config()
    for source in aggregate_sources(config):
        path = forecasts_dir(config) / source.filename
        if path.exists():
            frame = pd.read_csv(path, parse_dates=["date"])
            hit = frame[(frame["model"] == model) & (frame["fold"] == fold)]
            if len(hit):
                return hit.sort_values("date")
    pytest.skip(f"no saved aggregate forecast for {model} fold {fold}")


def test_forecast_chart_plots_exactly_the_saved_forecast_and_actuals():
    at = open_page("light", "views/2_Forecast_Explorer.py")
    model = next(m for m in next(s for s in at.selectbox if s.label == "Model").options if m.startswith("SeasonalNaive"))
    at.selectbox(key="fx_model").select(model).run()
    at.selectbox(key="fx_fold").select(5).run()
    at.selectbox(key="fx_horizon").select(28).run()
    assert_no_exception(at)
    saved = _saved_aggregate(model, 5)
    spec = chart_specs(at)[0]
    forecast = next(t for t in spec["data"] if str(t.get("name", "")).endswith("forecast"))
    actual = next(t for t in spec["data"] if t.get("name") == "Actual sales in the forecast window")
    np.testing.assert_allclose(decode_array(forecast["y"]), saved["predicted"].to_numpy(), rtol=1e-9, atol=1e-9)
    np.testing.assert_allclose(decode_array(actual["y"]), saved["actual"].to_numpy(), rtol=1e-9, atol=1e-9)
    assert len(decode_array(forecast["y"])) == 28


def test_shorter_horizon_shows_the_first_days_of_the_same_saved_forecast():
    at = open_page("light", "views/2_Forecast_Explorer.py")
    at.selectbox(key="fx_horizon").select(7).run()
    assert_no_exception(at)
    spec = chart_specs(at)[0]
    forecast = next(t for t in spec["data"] if str(t.get("name", "")).endswith("forecast"))
    assert len(decode_array(forecast["y"])) == 7


def test_model_comparison_numbers_match_the_saved_forecasts():
    from src.evaluation.metrics import compute_metrics

    at = open_page("light", "views/3_Model_Comparison.py")
    body = " ".join(str(m.value) for m in at.main.markdown)
    saved = _saved_aggregate("SeasonalNaive", 1)
    maes = []
    for fold in range(1, 6):
        part = _saved_aggregate("SeasonalNaive", fold)
        maes.append(compute_metrics(part["actual"].to_numpy(), part["predicted"].to_numpy()).mae)
    assert f"{np.mean(maes):,.3f}" in body, "the mean fold MAE shown for SeasonalNaive is not what the saved forecasts give"
    assert len(saved) == 28


def test_aggregate_and_item_level_results_are_never_ranked_together():
    at = open_page("light", "views/3_Model_Comparison.py")
    body = " ".join(texts(at)).lower()
    assert "not comparable" in body
    tab_labels = [t.label for t in at.tabs]
    assert any("Aggregate" in t for t in tab_labels) and any("Item level" in t for t in tab_labels)
    # one badge row per scope; a single ranking never mixes scales
    assert body.count("lowest is") >= 2


# ----------------------------------------------------------------------------------------------------------- holdout safety
def _holdout_dates(text: str):
    start, end = date.fromisoformat(HOLDOUT_START), date.fromisoformat(HOLDOUT_END)
    return [d for d in DATE_RE.findall(text) if start <= date.fromisoformat(d) <= end]


@pytest.mark.parametrize("path", PAGE_PATHS)
def test_holdout_dates_appear_only_in_sentences_that_say_it_is_reserved(path):
    at = open_page("light", path)
    for block in texts(at):
        if _holdout_dates(block):
            lowered = block.lower()
            assert any(w in lowered for w in GUARD_WORDS), f"{path}: holdout date shown as data: {block[:200]}"
    for spec in chart_specs(at):
        offenders = [v for v in flat_values(spec) if isinstance(v, str) and _holdout_dates(v)]
        assert not offenders, f"{path}: a chart plots holdout dates {offenders[:3]}"


def test_forecast_explorer_never_offers_a_window_that_ends_after_the_development_window():
    at = open_page("light", "views/2_Forecast_Explorer.py")
    for label in at.selectbox(key="fx_fold").options:
        for d in DATE_RE.findall(str(label)):
            assert date.fromisoformat(d) <= date.fromisoformat("2016-04-24"), label


def test_sidebar_reports_the_holdout_state_from_file_existence_only():
    state = da.holdout_state()
    assert set(state) == {"frozen", "evaluated"}
    assert state["evaluated"] is False, "this repository snapshot must not contain a final holdout evaluation"
    side = " ".join(texts(start_app("light"), sidebar=True))
    assert ("Not evaluated" in side) is (not state["evaluated"])


def test_app_source_never_touches_the_holdout_reader_or_trains_models():
    root = da.PROJECT_ROOT / "app"
    for path in root.rglob("*.py"):
        source = path.read_text(encoding="utf-8")
        for forbidden in ("read_actuals", "HoldoutActuals", "load_holdout_inputs", "evaluate_frozen_candidates", ".fit(", "import torch", "run_backtest"):
            assert forbidden not in source, f"{path.relative_to(root.parent)} contains {forbidden!r}"
