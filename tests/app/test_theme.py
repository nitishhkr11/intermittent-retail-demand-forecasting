"""Theme tokens: accessible contrast in both themes, a real dark/light difference, and token-driven chart styling."""

from __future__ import annotations

import re

import pytest

import theme

THEMES = theme.THEME_NAMES


def test_contrast_ratio_reference_values():
    assert theme.contrast_ratio("#000000", "#ffffff") == pytest.approx(21.0)
    assert theme.contrast_ratio("#ffffff", "#ffffff") == pytest.approx(1.0)
    assert theme.contrast_ratio("#777777", "#ffffff") == pytest.approx(theme.contrast_ratio("#ffffff", "#777777"))
    with pytest.raises(ValueError):
        theme.relative_luminance("#fff")


@pytest.mark.parametrize("name", THEMES)
@pytest.mark.parametrize("fg,bg,minimum", theme.TEXT_PAIRS)
def test_text_pairs_meet_wcag_aa(name, fg, bg, minimum):
    t = theme.THEMES[name]
    ratio = theme.contrast_ratio(t[fg], t[bg])
    assert ratio >= minimum, f"{name}: {fg} on {bg} has contrast {ratio:.2f} < {minimum}"


@pytest.mark.parametrize("name", THEMES)
def test_chart_colours_are_visible_on_the_chart_surface(name):
    t = theme.THEMES[name]
    for colour in theme.series_colors(name):
        assert theme.contrast_ratio(colour, t["surface"]) >= 3.0, f"{name}: series colour {colour} too faint on the chart background"
    assert theme.contrast_ratio(t["history"], t["surface"]) >= 3.0


@pytest.mark.parametrize("name", THEMES)
def test_series_colours_are_distinct_and_never_the_only_cue(name):
    colours = theme.series_colors(name)
    assert len(set(colours)) == len(colours) >= 5
    assert len(set(theme.LINE_DASHES)) >= 5 and len(set(theme.MARKER_SYMBOLS)) >= 5


def test_both_themes_define_the_same_tokens_and_differ():
    assert set(theme.THEMES["light"]) == set(theme.THEMES["dark"])
    assert theme.THEMES["light"]["bg"] != theme.THEMES["dark"]["bg"]
    assert theme.relative_luminance(theme.THEMES["light"]["bg"]) > 0.5 > theme.relative_luminance(theme.THEMES["dark"]["bg"])
    assert theme.THEMES["light"]["color_scheme"] == "light" and theme.THEMES["dark"]["color_scheme"] == "dark"


@pytest.mark.parametrize("value,expected", [("dark", "dark"), ("LIGHT", "light"), (" Dark ", "dark"), ("purple", "light"), (None, "light"), ("", "light")])
def test_normalise_theme(value, expected):
    assert theme.normalise_theme(value) == expected


@pytest.mark.parametrize("name", THEMES)
def test_stylesheet_themes_every_surface_the_brief_names(name):
    css = theme.build_css(name)
    t = theme.THEMES[name]
    assert t["bg"] in css and t["surface"] in css
    for needle in ("stSidebar", "stSelectbox", "stNumberInput", "stSlider", "stCheckbox", "stTooltipContent", ".rd-card", ".rd-table", "[role=\"listbox\"]",
                   "stTabs", "stExpander", "stDownloadButton", "color-scheme"):
        assert needle in css, f"stylesheet has no rule for {needle}"


def test_stylesheets_differ_between_themes():
    assert theme.build_css("light") != theme.build_css("dark")
    assert theme.THEMES["dark"]["bg"] not in theme.build_css("light")


@pytest.mark.parametrize("name", THEMES)
def test_chart_layout_follows_the_theme(name):
    t = theme.THEMES[name]
    layout = theme.chart_layout(name, height=300, title="Title")
    assert layout["paper_bgcolor"] == t["surface"] and layout["plot_bgcolor"] == t["surface"]
    assert layout["font"]["color"] == t["text"]
    assert layout["hoverlabel"]["bgcolor"] == t["surface_alt"] and layout["hoverlabel"]["font"]["color"] == t["text"]
    assert layout["height"] == 300
    assert layout["legend"]["y"] < 0, "legend must sit below the axis title, not on top of it"


def test_chart_layout_differs_between_themes():
    assert theme.chart_layout("light")["paper_bgcolor"] != theme.chart_layout("dark")["paper_bgcolor"]


def test_no_machine_paths_or_remote_assets_in_the_stylesheet():
    css = theme.build_css("dark")
    assert not re.search(r"https?://", css), "the stylesheet must not pull remote fonts or images"
    assert not re.search(r"[A-Za-z]:\\\\|/home/|/Users/", css)
