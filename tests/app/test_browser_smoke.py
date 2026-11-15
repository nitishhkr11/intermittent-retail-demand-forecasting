"""Real-browser check (slow): the dark-mode toggle changes the colours the browser actually paints, in the sidebar and in the main area,
and keeps them while moving between pages. Skipped when Playwright or its Chromium is not installed.

Optional: set RIVERDEMAND_SCREENSHOT_DIR to a folder to also save a screenshot of each page in each theme.
"""

from __future__ import annotations

import os
import re
import socket
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

import pytest

pytestmark = pytest.mark.slow
playwright_sync = pytest.importorskip("playwright.sync_api", reason="playwright is not installed (pip install playwright; python -m playwright install chromium)")

ROOT = Path(__file__).resolve().parents[2]
PAGE_URLS = {"Overview": "", "Forecast Explorer": "Forecast_Explorer", "Model Comparison": "Model_Comparison", "Inventory Decisions": "Inventory_Decision",
             "Model Reliability": "Model_Reliability", "About & Methodology": "About_Methodology"}
_OPEN_CONTEXTS = []


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


@pytest.fixture(scope="module")
def server():
    port = _free_port()
    proc = subprocess.Popen([sys.executable, "-m", "streamlit", "run", "app/app.py", "--server.port", str(port), "--server.headless", "true",
                             "--browser.gatherUsageStats", "false"], cwd=ROOT, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    base = f"http://127.0.0.1:{port}"
    try:
        for _ in range(60):
            try:
                if urllib.request.urlopen(base + "/_stcore/health", timeout=2).status == 200:
                    break
            except Exception:
                time.sleep(1)
        else:
            pytest.skip("the Streamlit server did not start in time")
        yield base
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            proc.kill()


@pytest.fixture(scope="module")
def browser():
    with playwright_sync.sync_playwright() as p:
        try:
            b = p.chromium.launch()
        except Exception as exc:  # pragma: no cover - browser binary missing
            pytest.skip(f"Chromium is not available: {exc}")
        yield b
        b.close()


@pytest.fixture(autouse=True)
def close_page_contexts_after_each_browser_test():
    yield
    while _OPEN_CONTEXTS:
        context = _OPEN_CONTEXTS.pop()
        try:
            context.close()
        except Exception:
            pass


def _rgb(text: str):
    return tuple(int(v) for v in re.findall(r"\d+", text)[:3])


def _luma(rgb) -> float:
    return 0.2126 * rgb[0] + 0.7152 * rgb[1] + 0.0722 * rgb[2]


def _paint(page):
    return page.evaluate("""() => ({
        app: getComputedStyle(document.querySelector('[data-testid="stApp"]')).backgroundColor,
        side: getComputedStyle(document.querySelector('[data-testid="stSidebar"]')).backgroundColor,
        main: getComputedStyle(document.querySelector('[data-testid="stMain"]')).backgroundColor,
        header: getComputedStyle(document.querySelector('[data-testid="stHeader"]')).backgroundColor,
        container: getComputedStyle(document.querySelector('[data-testid="stMainBlockContainer"]')).backgroundColor,
        width: document.querySelector('[data-testid="stMainBlockContainer"]').getBoundingClientRect().width,
        maxWidth: getComputedStyle(document.querySelector('[data-testid="stMainBlockContainer"]')).maxWidth,
        text: getComputedStyle(document.querySelector('h1')).color })""")


def _open(browser, url, size=(1366, 768)):
    context = browser.new_context(viewport={"width": size[0], "height": size[1]})
    _OPEN_CONTEXTS.append(context)
    page = context.new_page()
    page._rd_browser_errors = []
    page._rd_framework_probe_404 = []
    page.on("pageerror", lambda exc: page._rd_browser_errors.append(str(exc)))
    def capture_console_error(msg):
        if msg.type != "error":
            return
        url = msg.location.get("url", "")
        # Streamlit 1.66.1 resolves these two framework health probes relative
        # to a hidden multipage route. They return 404, but do not affect page
        # rendering, navigation, or app JavaScript. Keep them visible as a
        # separate, narrowly scoped framework diagnostic.
        if "404 (Not Found)" in msg.text and any(url.endswith(f"/_stcore/{name}") for name in ("health", "host-config")):
            page._rd_framework_probe_404.append(url)
            return
        page._rd_browser_errors.append(f"{msg.text} @ {msg.location}")
    page.on("console", capture_console_error)
    page.goto(url, wait_until="networkidle")
    page.wait_for_selector("h1", timeout=60000)
    page.wait_for_timeout(1500)
    return page


def _select_tab(page, name: str) -> None:
    """Wait for Streamlit's client-side tab selection before checking its content."""
    page.get_by_role("tab", name=name).click()
    page.wait_for_function(
        "label => Array.from(document.querySelectorAll('[role=tab]')).some("
        "tab => tab.innerText.trim() === label && tab.getAttribute('aria-selected') === 'true')",
        arg=name,
        timeout=15000,
    )


@pytest.mark.parametrize("size", [(1440, 900), (1366, 768)])
def test_dark_and_light_paint_differently_in_the_browser(server, browser, size):
    light = _paint(_open(browser, server + "/?theme=light", size))
    dark = _paint(_open(browser, server + "/?theme=dark", size))
    assert all(light[k] == "rgb(255, 255, 255)" for k in ("app", "side", "main", "header", "container"))
    assert _luma(_rgb(light["app"])) > 200 and _luma(_rgb(dark["app"])) < 60
    assert _luma(_rgb(light["side"])) > 180 and _luma(_rgb(dark["side"])) < 70, "the sidebar does not follow the theme"
    assert _luma(_rgb(light["text"])) < 90 and _luma(_rgb(dark["text"])) > 170, "headings keep their colour across themes"
    if size[0] == 1440:
        assert light["width"] >= 1100 and light["maxWidth"] == "1280px"


def test_the_toggle_switches_the_live_page_and_the_choice_survives_navigation(server, browser):
    page = _open(browser, server + "/?theme=light")
    assert _luma(_rgb(_paint(page)["app"])) > 200
    page.get_by_text("Dark mode", exact=True).first.click()
    page.wait_for_timeout(2500)
    after = _paint(page)
    assert _luma(_rgb(after["app"])) < 60 and _luma(_rgb(after["side"])) < 70
    page.get_by_role("link", name=re.compile("Model Comparison")).first.click()
    page.wait_for_selector("h1:has-text('Model Comparison')", timeout=60000)
    page.wait_for_timeout(2000)
    assert _luma(_rgb(_paint(page)["app"])) < 60, "dark mode was lost when changing page"
    page.get_by_text("Dark mode", exact=True).first.click()
    page.wait_for_timeout(2500)
    assert _luma(_rgb(_paint(page)["app"])) > 200


def test_warning_and_information_panels_stay_white_with_semantic_accents(server, browser):
    page = _open(browser, server + "/Inventory_Decision?theme=light")
    # Inventory result generation can take longer than a normal page render because
    # it bootstraps the saved error windows. Wait for the downstream explanation.
    page.get_by_text("What this means, in plain English").wait_for(timeout=60000)
    warnings = page.locator(".rd-callout.warn")
    warning = warnings.first
    info = page.locator(".rd-callout.info").first
    assert warnings.count() == 3 and info.count() >= 1, "the illustrative-assumptions, expanded calibration, and alternative calibration warnings must be visible"
    warning_style = warning.evaluate("e => ({background:getComputedStyle(e).backgroundColor, accent:getComputedStyle(e).borderLeftColor})")
    info_style = info.evaluate("e => ({background:getComputedStyle(e).backgroundColor, accent:getComputedStyle(e).borderLeftColor})")
    assert warning_style == {"background": "rgb(255, 255, 255)", "accent": "rgb(138, 90, 0)"}
    assert info_style == {"background": "rgb(255, 255, 255)", "accent": "rgb(37, 99, 235)"}


def test_inventory_page_shows_verified_alternative_calibration_and_downloads(server, browser):
    page = _open(browser, server + "/Inventory_Decision?theme=light")
    page.get_by_text("Alternative 7-day calibration validation", exact=True).wait_for(timeout=60000)
    summary = page.locator(".rd-table-wrap").filter(has_text="Same-class normalized errors").first
    assert "92.59%" in summary.inner_text()
    assert "92.47%" in summary.inner_text()
    assert "86.72%" in summary.inner_text()
    assert "823" in summary.inner_text() and "930,248" in summary.inner_text()
    assert page.get_by_text("No calibration rule reached 95% cycle service", exact=False).count() >= 1
    with page.expect_download(timeout=15000) as download_info:
        page.get_by_role("button", name="Download this table (CSV)").first.click()
    assert download_info.value.suggested_filename == "inventory_alternative_method_summary.csv"
    assert page.locator('[data-testid="stException"]').count() == 0


def test_expanded_comparison_exposes_prophet_interval_results(server, browser):
    page = _open(browser, server + "/Model_Comparison?theme=light")
    _select_tab(page, "Expanded development study")
    page.get_by_text("Prophet seasonal interval study", exact=True).wait_for(timeout=60000)
    assert page.get_by_text("Block conformal intervals are deliberately wide", exact=False).count() == 1
    assert page.locator('[data-testid="stException"]').count() == 0


def test_expanded_comparison_exposes_complete_neural_seasonal_results(server, browser):
    page = _open(browser, server + "/Model_Comparison?theme=light")
    _select_tab(page, "Expanded development study")
    heading = page.get_by_text("LSTM/GRU seasonal comparison", exact=True)
    heading.wait_for(timeout=60000)
    heading.scroll_into_view_if_needed()
    neural_table = page.locator(".rd-table-wrap").filter(has_text="185.092")
    assert neural_table.count() >= 1
    assert "LSTM(calendar)" in neural_table.first.inner_text()
    assert page.locator('[data-testid="stException"]').count() == 0


def test_expanded_comparison_exposes_xgboost_feature_ablations(server, browser):
    page = _open(browser, server + "/Model_Comparison?theme=light")
    _select_tab(page, "Expanded development study")
    page.get_by_text("XGBoost feature ablations", exact=True).wait_for(timeout=60000)
    assert page.get_by_text("XGBoost(pooled,lags_calendar_events_poisson,poisson)", exact=False).count() >= 1
    assert page.get_by_text("XGBoost(per-item,lags_calendar_events)", exact=False).count() >= 1
    assert page.locator('[data-testid="stException"]').count() == 0


def test_expanded_comparison_exposes_alternative_inventory_calibration(server, browser):
    page = _open(browser, server + "/Model_Comparison?theme=light")
    _select_tab(page, "Expanded development study")
    page.get_by_text("Alternative inventory calibration methods", exact=True).wait_for(timeout=60000)
    assert page.locator(".rd-table-wrap").filter(has_text="class_normalized").count() >= 1
    assert page.locator(".rd-table-wrap").filter(has_text="empirical_all").count() >= 1
    assert page.locator('[data-testid="stException"]').count() == 0


def test_final_holdout_tab_shows_the_single_saved_evaluation(server, browser):
    page = _open(browser, server + "/Model_Comparison?theme=light")
    _select_tab(page, "Development vs final holdout")
    main = page.get_by_test_id("stMain")
    main.get_by_text("The final holdout was evaluated once", exact=False).wait_for(timeout=60000)
    assert "Class-selected strategy" in main.inner_text()
    assert page.get_by_text("run --confirm EVALUATE-FINAL-HOLDOUT-ONCE", exact=False).count() == 0
    assert page.locator('[data-testid="stException"]').count() == 0


def test_about_page_reports_consumed_holdout_state(server, browser):
    page = _open(browser, server + "/About_Methodology?theme=light")
    page.get_by_text("Final holdout evaluated once", exact=False).wait_for(timeout=60000)
    assert page.get_by_text("No final-period score exists yet", exact=False).count() == 0
    assert page.locator('[data-testid="stException"]').count() == 0


RESPONSIVE_SIZES = [(1920, 1080), (1440, 900), (1366, 768), (768, 1024), (390, 844)]


@pytest.mark.parametrize("name,slug", list(PAGE_URLS.items()))
@pytest.mark.parametrize("theme_name", ["light", "dark"])
@pytest.mark.parametrize("size", RESPONSIVE_SIZES)
def test_each_page_loads_in_the_browser_with_no_script_error_and_no_sideways_scroll(server, browser, name, slug, theme_name, size):
    page = _open(browser, f"{server}/{slug}?theme={theme_name}", size)
    assert page.locator('[data-testid="stException"]').count() == 0, f"{name}: Streamlit reported an exception"
    assert page.locator("h1").first.inner_text().strip().lower().startswith(name.split(" &")[0].lower()[:8])
    painted = _paint(page)
    if theme_name == "light":
        assert all(painted[k] == "rgb(255, 255, 255)" for k in ("app", "side", "main", "header", "container")), f"{name}: a light-mode surface is not pure white"
        assert _luma(_rgb(painted["text"])) < 100
    else:
        assert _luma(_rgb(painted["app"])) < 70 and _luma(_rgb(painted["side"])) < 80, f"{name}: dark surfaces changed"
    if size[0] <= 768:
        # Streamlit opens its navigation drawer on a fresh mobile session. Close it
        # so the captured responsive page shows the actual single-column content.
        collapse = page.get_by_role("button", name="Collapse sidebar")
        if collapse.count():
            collapse.click()
            page.wait_for_timeout(300)
    overflow = page.evaluate("() => document.documentElement.scrollWidth - document.documentElement.clientWidth")
    assert overflow <= 2, f"{name}: the page scrolls sideways by {overflow}px"
    assert not page._rd_browser_errors, f"{name}: browser errors: {page._rd_browser_errors}"
    assert len(page._rd_framework_probe_404) <= 2, f"{name}: unexpected Streamlit framework probe responses: {page._rd_framework_probe_404}"
    folder = os.environ.get("RIVERDEMAND_SCREENSHOT_DIR")
    if folder:
        Path(folder).mkdir(parents=True, exist_ok=True)
        page.screenshot(path=str(Path(folder) / f"{slug or 'Overview'}_{theme_name}_{size[0]}.png"), full_page=True)


@pytest.mark.parametrize("size,columns", [((1920, 1080), 3), ((1440, 900), 3), ((1366, 768), 3), ((768, 1024), 2), ((390, 844), 1)])
def test_overview_kpis_use_a_responsive_three_two_one_column_grid(server, browser, size, columns):
    page = _open(browser, server + "/?theme=light", size)
    cards = page.locator(".rd-kpis .rd-card")
    assert cards.count() == 6
    tops = page.locator(".rd-kpis .rd-card").evaluate_all("els => els.map(e => Math.round(e.getBoundingClientRect().top))")
    rows = {}
    for top in tops:
        rows[top] = rows.get(top, 0) + 1
    assert sorted(rows.values(), reverse=True) == sorted([columns] * (6 // columns) + ([6 % columns] if 6 % columns else []), reverse=True)


@pytest.mark.parametrize("name,slug", list(PAGE_URLS.items()))
def test_theme_toggle_repaints_each_page(server, browser, name, slug):
    page = _open(browser, f"{server}/{slug}?theme=light")
    assert _paint(page)["app"] == "rgb(255, 255, 255)"
    page.get_by_text("Dark mode", exact=True).first.click()
    page.wait_for_function("getComputedStyle(document.querySelector('[data-testid=stApp]')).backgroundColor !== 'rgb(255, 255, 255)'", timeout=10000)
    dark = _paint(page)
    assert dark["main"] != "rgb(255, 255, 255)" and dark["side"] != "rgb(255, 255, 255)"
