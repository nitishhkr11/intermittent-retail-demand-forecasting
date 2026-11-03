"""Sidebar: product name, page navigation, theme toggle, data scope and holdout status."""

from __future__ import annotations

import streamlit as st

import theme
from components.cards import badge, esc

PAGES = (
    ("views/1_Overview.py", "Overview", "📊"),
    ("views/2_Forecast_Explorer.py", "Forecast Explorer", "📈"),
    ("views/3_Model_Comparison.py", "Model Comparison", "⚖️"),
    ("views/4_Inventory_Decision.py", "Inventory Decisions", "📦"),
    ("views/5_Model_Reliability.py", "Model Reliability", "🔍"),
    ("views/6_About_Methodology.py", "About & Methodology", "📖"),
)
TOGGLE_KEY = "rd_theme_toggle"


def _on_toggle() -> None:
    st.session_state[theme.SESSION_KEY] = "dark" if st.session_state.get(TOGGLE_KEY) else "light"


def init_theme(default: str = theme.DEFAULT_THEME) -> str:
    """Set the session theme once: from `?theme=dark|light` in the URL if given, else the configured default."""
    if theme.SESSION_KEY not in st.session_state:
        requested = st.query_params.get("theme") if hasattr(st, "query_params") else None
        st.session_state[theme.SESSION_KEY] = theme.normalise_theme(requested or default)
    return theme.normalise_theme(st.session_state[theme.SESSION_KEY])


def render_sidebar(overview_scope: str, holdout: dict) -> None:
    with st.sidebar:
        st.markdown('<div class="rd-brand">River<span>Demand</span></div>'
                    '<div class="rd-tagline">Forecasting intermittent retail demand: Walmart M5, one store, one department.</div>',
                    unsafe_allow_html=True)
        st.markdown('<div class="rd-side-label">Pages</div>', unsafe_allow_html=True)
        for path, label, icon in PAGES:
            st.page_link(path, label=label, icon=icon)
        st.markdown('<div class="rd-side-label">Appearance</div>', unsafe_allow_html=True)
        st.toggle("Dark mode", value=st.session_state.get(theme.SESSION_KEY) == "dark", key=TOGGLE_KEY, on_change=_on_toggle,
                  help="Switches the whole app (sidebar, cards, inputs, tables and charts) between the light and the dark theme.")
        st.markdown('<div class="rd-side-label">Data scope</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="rd-scope">{esc(overview_scope)}</div>', unsafe_allow_html=True)
        st.markdown('<div class="rd-side-label">Final holdout</div>', unsafe_allow_html=True)
        if holdout.get("evaluated"):
            st.markdown(badge("info", "Evaluated (see record)"), unsafe_allow_html=True)
        else:
            st.markdown(badge("ok", "Not evaluated"), unsafe_allow_html=True)
        st.markdown('<div class="rd-scope" style="margin-top:0.4rem">All numbers in this app are development results. '
                    "The last 28 days stay unseen until the one-time final evaluation.</div>", unsafe_allow_html=True)
