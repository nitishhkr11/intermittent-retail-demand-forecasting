"""Light / dark theme for the RiverDemand dashboard.

Streamlit has no run-time theme switch, so the toggle in the sidebar works by injecting a stylesheet built from the
token tables below. The same tokens drive the Plotly charts (`chart_layout`), so the sidebar, cards, inputs, tables,
tooltips and charts always change together. The selected theme lives in `st.session_state["rd_theme"]`, which is not a
widget key, so it survives page changes and reruns for the whole browser session.

Accessibility: every text token is checked against the surfaces it appears on (tests/app/test_theme.py asserts WCAG AA,
4.5:1 for text and 3:1 for chart marks), and nothing in the app relies on colour alone: statuses carry an icon and a
word, chart series differ by dash and marker as well as colour, and bars carry value labels.
"""

from __future__ import annotations

from typing import Any, Dict, List

DEFAULT_THEME = "light"
THEME_NAMES = ("light", "dark")
SESSION_KEY = "rd_theme"

# Token tables. "series" follows the validated categorical order (blue, orange, aqua, yellow, magenta, green, violet, red).
THEMES: Dict[str, Dict[str, Any]] = {
    "light": {
        "bg": "#f5f4f0",
        "surface": "#fcfcfb",
        "surface_alt": "#eceae4",
        "border": "#b9b7ae",
        "border_soft": "#d6d4cc",
        "text": "#0b0b0b",
        "text_secondary": "#3f3e39",
        "muted": "#55544d",
        "accent": "#1c5cab",
        "accent_soft": "#dce8f8",
        "on_accent": "#ffffff",
        "grid": "#d9d7cf",
        "good": "#0d6b0d",
        "warning": "#8a5a00",
        "critical": "#b32d2d",
        "info": "#1c5cab",
        "good_bg": "#e3f2e3",
        "warning_bg": "#fbefd2",
        "critical_bg": "#f9e1e1",
        "info_bg": "#dce8f8",
        "series": ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"],
        "history": "#3f3e39",
        "color_scheme": "light",
    },
    "dark": {
        "bg": "#111110",
        "surface": "#1a1a19",
        "surface_alt": "#262624",
        "border": "#6b6a63",
        "border_soft": "#3a3a37",
        "text": "#f0efec",
        "text_secondary": "#d2d1c8",
        "muted": "#b0afa4",
        "accent": "#86b6ef",
        "accent_soft": "#1f3350",
        "on_accent": "#0b0b0b",
        "grid": "#33332f",
        "good": "#6fd16f",
        "warning": "#f2b640",
        "critical": "#f08080",
        "info": "#86b6ef",
        "good_bg": "#173317",
        "warning_bg": "#3a2d0c",
        "critical_bg": "#401a1a",
        "info_bg": "#1f3350",
        "series": ["#3987e5", "#d95926", "#199e70", "#c98500", "#d55181", "#008300", "#9085e9", "#e66767"],
        "history": "#d2d1c8",
        "color_scheme": "dark",
    },
}

#: (foreground token, background token, minimum contrast) pairs that must hold in both themes (see tests/app/test_theme.py).
TEXT_PAIRS = (
    ("text", "bg", 4.5), ("text", "surface", 4.5), ("text_secondary", "bg", 4.5), ("text_secondary", "surface", 4.5),
    ("muted", "bg", 4.5), ("muted", "surface", 4.5), ("muted", "surface_alt", 4.5), ("text", "surface_alt", 4.5),
    ("accent", "bg", 4.5), ("accent", "surface", 4.5), ("accent", "accent_soft", 4.5),
    ("good", "good_bg", 4.5), ("warning", "warning_bg", 4.5), ("critical", "critical_bg", 4.5), ("info", "info_bg", 4.5),
    ("text", "good_bg", 4.5), ("text", "warning_bg", 4.5), ("text", "critical_bg", 4.5), ("text", "info_bg", 4.5),
    ("on_accent", "accent", 4.5),
)


# ----------------------------------------------------------------------------------------------------- contrast maths
def _channel(value: int) -> float:
    c = value / 255.0
    return c / 12.92 if c <= 0.03928 else ((c + 0.055) / 1.055) ** 2.4


def relative_luminance(hex_color: str) -> float:
    """WCAG relative luminance of `#rrggbb`."""
    h = hex_color.lstrip("#")
    if len(h) != 6:
        raise ValueError(f"expected #rrggbb, got {hex_color!r}")
    r, g, b = (int(h[i:i + 2], 16) for i in (0, 2, 4))
    return 0.2126 * _channel(r) + 0.7152 * _channel(g) + 0.0722 * _channel(b)


def contrast_ratio(foreground: str, background: str) -> float:
    """WCAG contrast ratio (1 to 21) between two `#rrggbb` colours."""
    a, b = relative_luminance(foreground), relative_luminance(background)
    lighter, darker = max(a, b), min(a, b)
    return (lighter + 0.05) / (darker + 0.05)


# ----------------------------------------------------------------------------------------------------- session state
def normalise_theme(value: Any) -> str:
    text = str(value).strip().lower() if value is not None else ""
    return text if text in THEME_NAMES else DEFAULT_THEME


def current_theme() -> str:
    """The active theme name. Falls back to the default outside a running app."""
    try:
        import streamlit as st

        return normalise_theme(st.session_state.get(SESSION_KEY, DEFAULT_THEME))
    except Exception:  # pragma: no cover - only outside a Streamlit runtime
        return DEFAULT_THEME


def tokens(theme: str | None = None) -> Dict[str, Any]:
    return THEMES[normalise_theme(theme or current_theme())]


# ----------------------------------------------------------------------------------------------------- stylesheet
def build_css(theme: str) -> str:
    """The complete stylesheet for one theme (a `<style>` block)."""
    t = THEMES[normalise_theme(theme)]
    return f"""<style>
:root {{
  --rd-bg: {t['bg']}; --rd-surface: {t['surface']}; --rd-surface-alt: {t['surface_alt']};
  --rd-border: {t['border']}; --rd-border-soft: {t['border_soft']};
  --rd-text: {t['text']}; --rd-text-2: {t['text_secondary']}; --rd-muted: {t['muted']};
  --rd-accent: {t['accent']}; --rd-accent-soft: {t['accent_soft']}; --rd-on-accent: {t['on_accent']};
  --rd-good: {t['good']}; --rd-warning: {t['warning']}; --rd-critical: {t['critical']}; --rd-info: {t['info']};
  --rd-good-bg: {t['good_bg']}; --rd-warning-bg: {t['warning_bg']}; --rd-critical-bg: {t['critical_bg']}; --rd-info-bg: {t['info_bg']};
  color-scheme: {t['color_scheme']};
}}
html, body, .stApp, [data-testid="stApp"], [data-testid="stAppViewContainer"], [data-testid="stMain"] {{
  background: var(--rd-bg) !important; color: var(--rd-text) !important;
}}
[data-testid="stHeader"] {{ background: transparent !important; }}
[data-testid="stHeader"] *, [data-testid="stToolbar"] * {{ color: var(--rd-text-2) !important; }}
[data-testid="stMain"] .block-container {{ padding-top: 2.2rem; padding-bottom: 4rem; max-width: 1280px; }}
.stApp, .stApp p, .stApp li, .stApp label, .stApp span, .stApp h1, .stApp h2, .stApp h3, .stApp h4, .stApp h5,
.stApp [data-testid="stMarkdownContainer"], .stApp [data-testid="stWidgetLabel"] p, .stApp [data-testid="stCaptionContainer"] {{
  color: var(--rd-text);
}}
.stApp [data-testid="stCaptionContainer"], .stApp small, .stApp .rd-muted {{ color: var(--rd-muted) !important; }}
.stApp a {{ color: var(--rd-accent); text-decoration: underline; text-underline-offset: 2px; }}
.stApp hr {{ border-color: var(--rd-border-soft); }}
.stApp h1 {{ font-size: 1.9rem; font-weight: 700; letter-spacing: -0.01em; }}
.stApp h2 {{ font-size: 1.35rem; font-weight: 650; margin-top: 1.6rem; }}
.stApp h3 {{ font-size: 1.1rem; font-weight: 650; }}

/* sidebar */
[data-testid="stSidebar"], [data-testid="stSidebar"] > div {{ background: var(--rd-surface) !important; }}
[data-testid="stSidebar"] {{ border-right: 1px solid var(--rd-border-soft); }}
[data-testid="stSidebar"] * {{ color: var(--rd-text); }}
[data-testid="stSidebar"] .rd-muted, [data-testid="stSidebar"] [data-testid="stCaptionContainer"] {{ color: var(--rd-muted) !important; }}
[data-testid="stSidebar"] [data-testid="stPageLink"] a, [data-testid="stSidebar"] [data-testid="stPageLink-NavLink"] {{
  border-radius: 8px; padding: 0.45rem 0.7rem; text-decoration: none; border: 1px solid transparent; color: var(--rd-text);
}}
[data-testid="stSidebar"] [data-testid="stPageLink"] a:hover, [data-testid="stSidebar"] [data-testid="stPageLink-NavLink"]:hover {{
  background: var(--rd-surface-alt); }}
[data-testid="stSidebar"] [data-testid="stPageLink"] a[aria-current="page"], [data-testid="stSidebar"] [data-testid="stPageLink-NavLink"][aria-current="page"] {{
  background: var(--rd-accent-soft); border-color: var(--rd-accent); font-weight: 650; }}
[data-testid="stSidebar"] [data-testid="stPageLink"] a *, [data-testid="stSidebar"] [data-testid="stPageLink-NavLink"] * {{ color: var(--rd-text) !important; }}

/* inputs */
[data-baseweb="select"] > div, [data-baseweb="input"], [data-baseweb="base-input"], [data-baseweb="textarea"],
[data-testid="stNumberInput"] input, [data-testid="stNumberInputContainer"] {{
  background: var(--rd-surface) !important; color: var(--rd-text) !important; border-color: var(--rd-border) !important;
}}
[data-baseweb="select"] *, [data-baseweb="input"] input, [data-baseweb="base-input"] input {{ color: var(--rd-text) !important; }}
[data-baseweb="select"] svg, [data-testid="stNumberInput"] svg {{ fill: var(--rd-text-2) !important; }}
[data-baseweb="popover"], [data-baseweb="popover"] > div, [data-baseweb="menu"], ul[role="listbox"], [data-baseweb="popover"] ul {{
  background: var(--rd-surface) !important; color: var(--rd-text) !important; border: 1px solid var(--rd-border);
}}
li[role="option"], [data-baseweb="menu"] li {{ background: var(--rd-surface) !important; color: var(--rd-text) !important; }}
li[role="option"][aria-selected="true"], li[role="option"]:hover, [data-baseweb="menu"] li:hover {{
  background: var(--rd-accent-soft) !important; }}
/* Streamlit 1.5x+ widgets (react-aria) */
[data-testid="stSelectbox"] [role="group"], [data-testid="stMultiSelect"] [role="group"], [data-testid="stNumberInput"] [role="group"],
[data-testid="stTextInput"] [role="group"], .react-aria-ComboBox [role="group"] {{
  background: var(--rd-surface) !important; border: 1px solid var(--rd-border) !important; color: var(--rd-text) !important; }}
[role="combobox"], [data-testid="stNumberInput"] input, [data-testid="stTextInput"] input {{
  background: transparent !important; color: var(--rd-text) !important; -webkit-text-fill-color: var(--rd-text) !important; caret-color: var(--rd-text); }}
.react-aria-ComboBox svg, .react-aria-ComboBox svg *, [data-testid="stSelectbox"] button svg *, [data-testid="stNumberInput"] button svg * {{
  fill: var(--rd-text-2) !important; color: var(--rd-text-2) !important; }}
.react-aria-ComboBox svg path[fill="none"], [data-testid="stNumberInput"] svg path[fill="none"], [data-testid="stSelectbox"] svg path[fill="none"] {{ fill: none !important; }}
.react-aria-ComboBox button, [data-testid="stSelectbox"] [role="group"] button {{ background: transparent !important; border: none !important; }}
[data-testid="stNumberInput"] button {{ background: var(--rd-surface-alt) !important; color: var(--rd-text) !important; border-color: var(--rd-border) !important; }}
[data-testid="stSelectboxVirtualDropdown"], [data-testid="stSelectboxVirtualDropdown"] > div, [role="listbox"],
div:has(> [data-testid="stSelectboxVirtualDropdown"]), div:has(> [role="listbox"]) {{
  background: var(--rd-surface) !important; color: var(--rd-text) !important; border-color: var(--rd-border) !important; }}
[role="option"], [role="option"] * {{ background: transparent !important; color: var(--rd-text) !important; }}
[role="option"][aria-selected="true"], [role="option"][data-focused="true"], [role="option"]:hover, [role="option"][data-hovered="true"] {{
  background: var(--rd-accent-soft) !important; }}
[data-testid="stTooltipContent"], [data-baseweb="tooltip"], [role="tooltip"] {{
  background: var(--rd-surface-alt) !important; color: var(--rd-text) !important; border: 1px solid var(--rd-border); }}
[data-testid="stTooltipContent"] *, [role="tooltip"] * {{ color: var(--rd-text) !important; }}
[data-testid="stRadio"] label, [data-testid="stToggle"] label, [data-testid="stCheckbox"] label {{ color: var(--rd-text) !important; }}
[data-testid="stToggle"] [data-baseweb="checkbox"] > div:first-child, [data-testid="stToggle"] [role="checkbox"] + div {{
  border: 1px solid var(--rd-border); }}
[data-testid="stCheckbox"] label > span + div:not([data-testid]) {{ background: var(--rd-surface) !important; border: 1.5px solid var(--rd-muted) !important; }}
[data-testid="stCheckbox"] label > span:has(input:checked) + div:not([data-testid]) {{ background: var(--rd-accent) !important; border-color: var(--rd-accent) !important; }}
[data-testid="stCheckbox"] label > span:has(input:focus-visible) + div:not([data-testid]) {{ outline: 3px solid var(--rd-focus, var(--rd-accent)); outline-offset: 2px; }}
[data-testid="stSlider"] [data-testid="stTickBar"] *, [data-testid="stSlider"] [data-testid="stThumbValue"] {{ color: var(--rd-text) !important; }}
[data-testid="stSlider"] div[role="slider"] {{ background: var(--rd-accent) !important; }}
.stApp [data-testid^="stBaseButton"], .stApp [data-testid="stDownloadButton"] button {{
  background: var(--rd-surface) !important; color: var(--rd-text) !important; border: 1px solid var(--rd-border) !important; }}
.stApp [data-testid^="stBaseButton"]:hover, .stApp [data-testid="stDownloadButton"] button:hover {{ border-color: var(--rd-accent) !important; color: var(--rd-accent) !important; }}
.stApp [data-testid^="stBaseButton"] *, .stApp [data-testid="stDownloadButton"] button * {{ color: inherit !important; }}
[data-testid="stTooltipIcon"] svg, [data-testid="stTooltipHoverTarget"] svg {{ fill: none !important; stroke: var(--rd-text-2) !important; color: var(--rd-text-2) !important; }}
[data-testid="stTooltipIcon"] svg *, [data-testid="stTooltipHoverTarget"] svg * {{ fill: none !important; stroke: var(--rd-text-2) !important; }}
.stApp button:focus-visible, .stApp a:focus-visible, .stApp input:focus-visible, [data-baseweb="select"]:focus-within {{
  outline: 3px solid var(--rd-accent) !important; outline-offset: 2px; }}
[data-testid="stTabs"] button[role="tab"] {{ background: transparent !important; border: none !important; color: var(--rd-text-2) !important; }}
[data-testid="stTabs"] button[role="tab"][aria-selected="true"] {{ color: var(--rd-accent) !important; font-weight: 650; }}
[data-testid="stTabs"] [data-baseweb="tab-highlight"] {{ background: var(--rd-accent) !important; }}
[data-testid="stTabs"] [data-baseweb="tab-border"] {{ background: var(--rd-border-soft) !important; }}
[data-testid="stExpander"] details {{ background: var(--rd-surface); border: 1px solid var(--rd-border-soft); border-radius: 10px; }}
[data-testid="stExpander"] summary, [data-testid="stExpander"] summary * {{ color: var(--rd-text) !important; }}
[data-testid="stExpander"] svg {{ fill: var(--rd-text-2) !important; }}
.stPlotlyChart, [data-testid="stPlotlyChart"] {{ background: transparent; }}
[data-testid="stVerticalBlockBorderWrapper"] {{ border-color: var(--rd-border-soft); }}

/* components */
.rd-brand {{ font-size: 1.25rem; font-weight: 750; letter-spacing: -0.01em; margin: 0.2rem 0 0.1rem 0; }}
.rd-brand span {{ color: var(--rd-accent); }}
.rd-tagline {{ font-size: 0.86rem; color: var(--rd-muted) !important; margin: 0 0 0.9rem 0; line-height: 1.35; }}
.rd-side-label {{ font-size: 0.74rem; font-weight: 700; letter-spacing: 0.08em; text-transform: uppercase; color: var(--rd-muted) !important; margin: 1rem 0 0.35rem 0; }}
.rd-scope {{ font-size: 0.85rem; line-height: 1.5; color: var(--rd-text-2) !important; }}
.rd-page-title {{ margin-bottom: 0.15rem; }}
.rd-lede {{ color: var(--rd-text-2) !important; font-size: 1.02rem; margin: 0 0 1.1rem 0; max-width: 62rem; line-height: 1.5; }}
.rd-kpis {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(190px, 1fr)); gap: 0.8rem; margin: 0.4rem 0 1.2rem 0; }}
.rd-card {{ background: var(--rd-surface); border: 1px solid var(--rd-border-soft); border-radius: 12px; padding: 0.9rem 1rem; }}
.rd-card .k {{ font-size: 0.78rem; font-weight: 650; letter-spacing: 0.03em; color: var(--rd-muted) !important; text-transform: uppercase; }}
.rd-card .v {{ font-size: 1.7rem; font-weight: 700; line-height: 1.2; margin: 0.15rem 0; color: var(--rd-text) !important; }}
.rd-card .v.sm {{ font-size: 1.2rem; overflow-wrap: anywhere; }}
.rd-card .n {{ font-size: 0.84rem; color: var(--rd-text-2) !important; line-height: 1.35; }}
.rd-badge {{ display: inline-flex; align-items: center; gap: 0.35rem; font-size: 0.82rem; font-weight: 650; padding: 0.15rem 0.6rem;
  border-radius: 999px; border: 1px solid currentColor; white-space: nowrap; }}
.rd-badge.ok {{ color: var(--rd-good) !important; background: var(--rd-good-bg); }}
.rd-badge.warn {{ color: var(--rd-warning) !important; background: var(--rd-warning-bg); }}
.rd-badge.error {{ color: var(--rd-critical) !important; background: var(--rd-critical-bg); }}
.rd-badge.info {{ color: var(--rd-info) !important; background: var(--rd-info-bg); }}
.rd-callout {{ border: 1px solid var(--rd-border-soft); border-left-width: 5px; border-radius: 10px; padding: 0.75rem 1rem; margin: 0.6rem 0 1rem 0;
  background: var(--rd-surface); color: var(--rd-text) !important; line-height: 1.5; }}
.rd-callout .t {{ font-weight: 700; margin-bottom: 0.2rem; color: var(--rd-text) !important; }}
.rd-callout * {{ color: var(--rd-text); }}
.rd-callout.ok {{ border-left-color: var(--rd-good); background: var(--rd-good-bg); }}
.rd-callout.warn {{ border-left-color: var(--rd-warning); background: var(--rd-warning-bg); }}
.rd-callout.error {{ border-left-color: var(--rd-critical); background: var(--rd-critical-bg); }}
.rd-callout.info {{ border-left-color: var(--rd-info); background: var(--rd-info-bg); }}
.rd-code {{ background: var(--rd-surface-alt); color: var(--rd-text) !important; border: 1px solid var(--rd-border-soft); border-radius: 8px;
  padding: 0.55rem 0.75rem; font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace; font-size: 0.84rem;
  overflow-x: auto; white-space: pre; margin: 0.3rem 0; }}
code {{ background: var(--rd-surface-alt) !important; color: var(--rd-text) !important; padding: 0.05rem 0.3rem; border-radius: 4px; }}
.rd-table-wrap {{ overflow: auto; border: 1px solid var(--rd-border-soft); border-radius: 10px; background: var(--rd-surface); margin: 0.4rem 0 0.8rem 0; }}
table.rd-table {{ border-collapse: collapse; width: 100%; font-size: 0.86rem; color: var(--rd-text); }}
table.rd-table caption {{ text-align: left; padding: 0.5rem 0.75rem; font-weight: 650; color: var(--rd-text-2); caption-side: top; }}
table.rd-table th {{ position: sticky; top: 0; background: var(--rd-surface-alt); color: var(--rd-text); text-align: left; font-weight: 700;
  padding: 0.5rem 0.75rem; border-bottom: 1px solid var(--rd-border); white-space: nowrap; }}
table.rd-table td {{ padding: 0.4rem 0.75rem; border-bottom: 1px solid var(--rd-border-soft); color: var(--rd-text); }}
table.rd-table td.num, table.rd-table th.num {{ text-align: right; font-variant-numeric: tabular-nums; }}
table.rd-table tr:last-child td {{ border-bottom: none; }}
table.rd-table tr.best td {{ background: var(--rd-accent-soft); font-weight: 650; }}
.rd-prov {{ font-size: 0.84rem; color: var(--rd-text-2) !important; }}
.rd-hint {{ color: var(--rd-muted) !important; font-size: 0.86rem; }}
</style>"""


def inject_theme(theme: str | None = None) -> None:
    """Write the stylesheet of `theme` (default: the session's theme) into the page."""
    import streamlit as st

    st.markdown(build_css(theme or current_theme()), unsafe_allow_html=True)


# ----------------------------------------------------------------------------------------------------- chart styling
LINE_DASHES = ("solid", "dash", "dot", "dashdot", "longdash", "longdashdot")
MARKER_SYMBOLS = ("circle", "square", "diamond", "triangle-up", "x", "cross", "star", "triangle-down")


#: The categorical slots used for LINES and BARS: the ones that keep at least 3:1 contrast with the chart surface in both
#: themes (blue, orange, green, violet, red). Aqua, yellow and magenta fall below 3:1 on the light surface and are not used.
LINE_SLOTS = (0, 1, 5, 6, 7)


def series_colors(theme: str | None = None) -> List[str]:
    """Colours for chart lines and bars, in the fixed assignment order."""
    palette = tokens(theme)["series"]
    return [palette[i] for i in LINE_SLOTS]


def chart_layout(theme: str | None = None, *, height: int = 380, title: str | None = None) -> Dict[str, Any]:
    """Plotly layout arguments that match the active theme (backgrounds, text, grid, legend, tooltips)."""
    t = tokens(theme)
    top, bottom = (56 if title else 28), 112
    legend_y = -min(0.6, 74 / max(120, height - top - bottom))  # keep the legend a fixed distance below the axis title
    axis = dict(gridcolor=t["grid"], zerolinecolor=t["grid"], linecolor=t["border"], tickfont=dict(color=t["text_secondary"], size=12),
                title=dict(font=dict(color=t["text_secondary"], size=13)), automargin=True)
    return dict(
        template="none", height=height,
        paper_bgcolor=t["surface"], plot_bgcolor=t["surface"],
        font=dict(color=t["text"], family="Inter, 'Segoe UI', system-ui, sans-serif", size=13),
        title=dict(text=title, font=dict(color=t["text"], size=15), x=0.01, xanchor="left") if title else None,
        colorway=list(t["series"]),
        margin=dict(l=56, r=24, t=top, b=bottom),
        xaxis=dict(axis), yaxis=dict(axis),
        legend=dict(font=dict(color=t["text"], size=12), bgcolor="rgba(0,0,0,0)", orientation="h", yanchor="top", y=legend_y, x=0),
        hoverlabel=dict(bgcolor=t["surface_alt"], bordercolor=t["border"], font=dict(color=t["text"], size=13)),
        hovermode="x unified",
    )


def style_figure(fig, theme: str | None = None, *, height: int = 380, title: str | None = None):
    """Apply the active theme to a Plotly figure (in place) and return it."""
    fig.update_layout(**chart_layout(theme, height=height, title=title))
    return fig
