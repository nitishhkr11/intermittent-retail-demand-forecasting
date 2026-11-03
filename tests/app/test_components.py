"""Reusable dashboard components: HTML is escaped, status is never colour-only, missing values are labelled, empty states show commands."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from components import cards, tables
from streamlit.testing.v1 import AppTest


# ------------------------------------------------------------------------------------------------------------------- tables
def test_table_escapes_html_in_headers_cells_and_caption():
    df = pd.DataFrame({"<b>col</b>": ["<script>alert(1)</script>", "a & b"], "n": [1, 2]})
    out = tables.html_table(df, caption="<img src=x onerror=1>")
    assert "<script>" not in out and "<img" not in out and "<b>col</b>" not in out
    assert "&lt;script&gt;" in out and "a &amp; b" in out and "&lt;img" in out


def test_table_shows_missing_values_as_na_never_blank_or_nan():
    df = pd.DataFrame({"model": ["a", "b", "c"], "x": [1.5, float("nan"), None], "when": [pd.Timestamp("2016-01-03"), pd.NaT, None]})
    out = tables.html_table(df)
    assert all(f">{bad}<" not in out.lower() for bad in ("nan", "nat", "none")) and "></td>" not in out
    assert out.count("n/a") >= 3
    assert "2016-01-03" in out


def test_table_best_row_is_marked_with_text_not_only_colour():
    df = pd.DataFrame({"model": ["slow", "fast", "mid"], "MAE": [3.0, 1.0, 2.0]})
    out = tables.html_table(df, best={"MAE": "min"})
    assert out.count("(lowest MAE)") == 1
    row = [r for r in out.split("<tr") if "fast" in r][0]
    assert "(lowest MAE)" in row and 'class="best"' in row
    assert "(highest" not in out
    assert "(highest MAE)" in tables.html_table(df, best={"MAE": "max"})


def test_table_best_marker_is_skipped_when_every_value_is_missing():
    df = pd.DataFrame({"model": ["a", "b"], "MAE": [np.nan, np.nan]})
    assert "(lowest" not in tables.html_table(df, best={"MAE": "min"})


def test_table_formats_numbers_booleans_and_custom_formats():
    df = pd.DataFrame({"model": ["a"], "count": [12345], "value": [0.123456], "big": [123456.789], "ok": [True], "pct": [0.5]})
    out = tables.html_table(df, formats={"pct": "{:.1%}"})
    assert "12,345" in out and "0.123" in out and "123,456.8" in out and ">yes<" in out and "50.0%" in out


def test_table_has_scoped_header_cells_for_screen_readers():
    out = tables.html_table(pd.DataFrame({"a": [1]}))
    assert '<th scope="col"' in out and "<thead>" in out and "<tbody>" in out


def test_empty_table_does_not_crash():
    out = tables.html_table(pd.DataFrame({"a": [], "b": []}), best={"a": "min"})
    assert "<table" in out and "<tbody></tbody>" in out


# --------------------------------------------------------------------------------------------------------------------- cards
@pytest.mark.parametrize("kind,icon,word", [("ok", "✔", "OK"), ("warn", "▲", "Caution"), ("error", "✖", "Problem"), ("info", "ℹ", "Note")])
def test_badge_carries_an_icon_and_a_word(kind, icon, word):
    html = cards.badge(kind)
    assert icon in html and word in html


def test_badge_unknown_kind_falls_back_and_escapes_text():
    html = cards.badge("nonsense", "<i>hi</i>")
    assert "rd-badge info" in html and "<i>" not in html and "&lt;i&gt;" in html


def test_kpi_card_escapes_and_shrinks_long_values():
    short = cards.kpi_card_html("Label <b>", "123", "note & more")
    assert "&lt;b&gt;" in short and "note &amp; more" in short and 'class="v"' in short
    assert 'class="v sm"' in cards.kpi_card_html("L", "a very long value indeed")


def _empty_state_app():
    from components import cards as c
    import theme
    import streamlit as st

    st.session_state.setdefault(theme.SESSION_KEY, "light")
    c.empty_state("Nothing here", "A <b>detail</b>", ["python -m jupyter nbconvert --execute notebooks/x.ipynb"], ["artifacts/forecasts/x.csv"])
    c.callout("warn", "Heads up", "<u>text</u>")


def test_empty_state_names_the_missing_file_and_the_exact_command_and_escapes_markup():
    import sys
    from pathlib import Path

    root = Path(__file__).resolve().parents[2]
    sys.path[:0] = [str(root), str(root / "app")]
    at = AppTest.from_function(_empty_state_app, default_timeout=60).run()
    assert not at.exception
    html = " ".join(str(m.value) for m in at.markdown)
    assert "artifacts/forecasts/x.csv" in html
    assert "python -m jupyter nbconvert --execute notebooks/x.ipynb" in html
    assert "PowerShell" in html
    assert "<b>detail</b>" not in html and "&lt;b&gt;detail" in html and "<u>text</u>" not in html


def test_esc_handles_non_strings_and_quotes():
    assert cards.esc(5) == "5"
    assert cards.esc('"q"<') == "&quot;q&quot;&lt;"


# ------------------------------------------------------------------------------------------------------------------- charts
@pytest.mark.parametrize("version,expected", [("1.66.1", {"width": "stretch"}), ("1.50.0", {"width": "stretch"}), ("1.49.1", {"use_container_width": True}),
                                              ("1.46.0", {"use_container_width": True})])
def test_chart_helper_uses_the_width_argument_the_installed_streamlit_understands(monkeypatch, version, expected):
    import plotly.graph_objects as go
    import streamlit as st

    from components import charts

    seen = {}
    monkeypatch.setattr(st, "__version__", version)
    monkeypatch.setattr(st, "plotly_chart", lambda fig, **kwargs: seen.update(kwargs))
    charts.show(go.Figure())
    for key, value in expected.items():
        assert seen[key] == value
    assert seen["theme"] is None, "Streamlit's own chart theme must be off so the app theme is the only one applied"
    assert ("width" in seen) != ("use_container_width" in seen)
