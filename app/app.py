"""RiverDemand dashboard entry point.  Start it from the project root:  streamlit run app/app.py

The entry script runs on every interaction: it sets up the page, restores the theme chosen in this browser session, draws the
sidebar (navigation, light/dark toggle, data scope) and then runs the selected page. Pages only READ saved artifacts; nothing
is trained here and no forecast beyond the last development day is ever produced.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
APP_DIR = Path(__file__).resolve().parent
for _p in (str(ROOT), str(APP_DIR)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import streamlit as st  # noqa: E402

st.set_page_config(page_title="RiverDemand | Retail demand forecasting", page_icon="📈", layout="wide", initial_sidebar_state="expanded")

import data_access as da  # noqa: E402
import theme  # noqa: E402
from components import sidebar  # noqa: E402


def scope_text() -> str:
    config = da.get_config()
    store, dept = config["scope"]["store_id"], config["scope"]["dept_id"]
    overview = da.load_overview()
    if not overview.ok:
        return f"{store} · {dept}. Data summary unavailable: see the Overview page."
    v = overview.value
    w = v["window"]
    return (f"{store} · {dept} · {v['n_items']:,} items. Development window {w.dev_start.date()} to {w.dev_end.date()} "
            f"({v['n_days']:,} days). Final {w.n_holdout_days} days ({w.holdout_start.date()} to {w.holdout_end.date()}) reserved.")


config = da.get_config()
sidebar.init_theme(config.get("dashboard", {}).get("default_theme", theme.DEFAULT_THEME))

pages = [st.Page(path, title=label, icon=icon, default=(i == 0)) for i, (path, label, icon) in enumerate(sidebar.PAGES)]
current_page = st.navigation(pages, position="hidden")     # registers the pages (the sidebar links below need that)
sidebar.render_sidebar(scope_text(), da.holdout_state())
theme.inject_theme()
current_page.run()
